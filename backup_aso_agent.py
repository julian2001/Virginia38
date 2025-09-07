# run_langgraph_aso_agent.py
"""
LangGraph agent that optimises antisense‑oligonucleotides (ASOs) for a given
gene using VeraASOptimizer for candidate generation and ViennaRNA (RNA.fold)
for secondary‑structure + MFE scoring.

Usage:
    python run_langgraph_aso_agent.py
"""

__all__ = ["optimize_aso", "optimize_gene"]

from pathlib import Path
import subprocess

# LangChain / LangGraph
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt
from langgraph.prebuilt import tools_condition

# ASOptimizer import shim                                                    
# --------------------------------------------------------------------------- #
# We try local admet_ai4 model first, then known external packages.           
# If nothing loads, we raise, rather than fabricate a dummy model.            
# --------------------------------------------------------------------------- #
import importlib

ASOOptimizer = None

# Option 0 – Local module in admet_ai4 (preferred)
if ASOOptimizer is None:
    try:
        # Expecting `admet_ai4/aso_optimizer.py` or installed package `aso_optimizer`
        ASOOptimizer = importlib.import_module("aso_optimizer").ASOOptimizer
    except Exception:
        pass

# Option 1 – Vera fork
if ASOOptimizer is None:
    try:
        ASOOptimizer = importlib.import_module("VeraASOptimizer").ASOOptimizer
    except Exception:
        pass

# Option 2 – Spidercores implementation
if ASOOptimizer is None:
    try:
        ASOOptimizer = importlib.import_module("ASOptimizer.main").ASOOptimizer  # type: ignore
    except Exception:
        pass

# Final – hard fail (no dummy). We do NOT fabricate sequences.
if ASOOptimizer is None:
    raise ImportError(
        "ASOOptimizer not found. Ensure `aso_optimizer.ASOOptimizer` (local) or a compatible package is available."
    )

import RNA                                        # ViennaRNA Python bindings

import os
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --------------------------------------------------------------------------- #
# Tool: optimise an ASO for a gene                                            #
# --------------------------------------------------------------------------- #
@tool
def optimize_aso(gene: str) -> str:
    """
    Generate 5 ASO candidates for *gene*, fold each with ViennaRNA,
    save full table to aso_opt_<gene>.txt and return a summary string
    with the top MFE candidate.
    """
    optimizer = ASOOptimizer(target_gene=gene)
    candidates = optimizer.generate_candidates(n=5)

    results = []
    for seq in candidates:
        struct, mfe = RNA.fold(seq)
        results.append((seq, struct, mfe))

    out = Path(f"aso_opt_{gene}.txt")
    with out.open("w") as f:
        for seq, struct, mfe in results:
            f.write(f"{seq}\t{struct}\t{mfe:.2f}\n")

    # If using dummy fallback, results may be empty: explain to user
    # Explanatory comment for fallback:
    if not results:
        return "[ASOptimizer fallback] No candidates generated."

    best = min(results, key=lambda x: x[2])  # lowest MFE
    return (
        f"Top ASO for {gene}: {best[0]}  (MFE = {best[2]:.2f} kcal/mol)\n"
        f"Full results written to → {out.resolve()}"
    )
optimize_aso.__module__ = "run_langgraph_aso_agent"

# --------------------------------------------------------------------------- #
# LangChain / LangGraph scaffolding                                           #
# --------------------------------------------------------------------------- #
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    gene: str

# --------------------------------------------------------------------------- #
# Human‑in‑the‑loop helper tool                                                 #
# --------------------------------------------------------------------------- #
@tool
def human_assistance(query: str) -> str:  # noqa: D401 – imperative style is ok
    """Ask a human for help and return their answer."""
    human_response = interrupt({"query": query})  # pauses execution
    return human_response["data"]

all_tools = [optimize_aso, human_assistance]
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
llm_with_tools = llm.bind_tools(all_tools)

def llm_node(state: AgentState):
    """Core LLM step that can decide to call a tool."""
    resp = llm_with_tools.invoke(state["messages"])
    # If a tool is called we expect max 1 call, else recursion explosion
    assert len(resp.tool_calls) <= 1
    return {"messages": [resp]}

# Build LangGraph with conditional routing to tools
builder = StateGraph(AgentState)

builder.add_node("llm", llm_node)

# Re‑use LangGraph's ready‑made ToolNode wrapper
builder.add_node("tools", ToolNode(tools=all_tools))

# Decide whether to run a tool or finish after the LLM step
builder.add_conditional_edges("llm", tools_condition)

# Route tool output back to the LLM for further reasoning
builder.add_edge("tools", "llm")

# Entry and exit
builder.add_edge(START, "llm")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# --------------------------------------------------------------------------- #
# Public helper so other controllers can reuse the optimiser programmatically #
# --------------------------------------------------------------------------- #
def optimize_gene(gene: str, *, recursion_limit: int = 50) -> str:
    """
    Optimise antisense oligonucleotides for *gene* and return the assistant’s
    final textual reply (which already contains the best sequence and the path
    to the results‑file).  Other controllers can simply call this function.

    Example
    -------
    >>> from run_langgraph_aso_agent import optimize_gene
    >>> print(optimize_gene("IDO1"))
    Top ASO for IDO1: ... (MFE = ...)
    """
    seed_input = {
        "messages": [{"role": "user", "content": f"Optimise ASOs for {gene}"}],
        "gene": gene,
    }
    events = graph.stream(
        seed_input,
        {"thread_id": f"aso-{gene}", "recursion_limit": recursion_limit},
        stream_mode="values",
    )

    final_text: str | None = None
    for ev in events:
        # capture the content of the last assistant message
        if isinstance(ev, dict) and "messages" in ev and ev["messages"]:
            last = ev["messages"][-1]
            if isinstance(last, dict) and last.get("role") == "assistant":
                final_text = last["content"]
    return final_text or "[optimize_gene] No assistant response captured."
optimize_gene.__module__ = "run_langgraph_aso_agent"

# --------------------------------------------------------------------------- #
# Driver                                                                      #
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    seed_input = {
        "messages": [{"role": "user", "content": "Optimise ASOs for IDO1"}],
        "gene": "IDO1"
    }
    events = graph.stream(
        seed_input,
        {
            "thread_id": "aso-1",
            "recursion_limit": 100,
        },
        stream_mode="values",
    )
    for ev in events:
        print(ev)
