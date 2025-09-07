#!/usr/bin/env python
# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Project-local copy of Hugging Face 'agents' interfaces with safe fallbacks.

- If `transformers` is installed, we import the real modules.
- If not, we define minimal stubs so this file executes without crashing when
  `controller_agent5.py` launches `agents.py` as a script.

This module DOES NOT run anything heavy on import. When executed as a script,
it prints environment info and exits 0 so upstream controllers can capture stdout.
"""

from __future__ import annotations

import json
import logging
import re
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# -----------------------
# Pretty console formatter
# -----------------------
class CustomFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    bold_yellow = "\x1b[33;1m"
    red = "\x1b[31;20m"
    green = "\x1b[32;20m"
    bold_green = "\x1b[32;20;1m"
    bold_red = "\x1b[31;1m"
    bold_white = "\x1b[37;1m"
    orange = "\x1b[38;5;214m"
    bold_orange = "\x1b[38;5;214;1m"
    reset = "\x1b[0m"
    _fmt = "%(message)s"

    FORMATS = {
        logging.DEBUG: grey + _fmt + reset,
        logging.INFO: _fmt,
        logging.WARNING: bold_yellow + _fmt + reset,
        logging.ERROR: red + _fmt + reset,
        logging.CRITICAL: bold_red + _fmt + reset,
        31: reset + _fmt + reset,
        32: green + _fmt + reset,
        33: bold_green + _fmt + reset,
        34: bold_white + _fmt + reset,
        35: orange + _fmt + reset,
        36: bold_orange + _fmt + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno, self._fmt)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


# ---------------
# HF compat layer
# ---------------
HF_AVAILABLE = False
try:
    # Prefer absolute imports (works when transformers is installed)
    from transformers import is_torch_available  # type: ignore
    from transformers.utils import logging as transformers_logging  # type: ignore
    from transformers.utils.import_utils import is_pygments_available  # type: ignore
    from transformers.agents.agent_types import AgentAudio, AgentImage  # type: ignore
    from transformers.agents.default_tools import (  # type: ignore
        BASE_PYTHON_TOOLS,
        FinalAnswerTool,
        setup_default_tools,
    )
    from transformers.agents.llm_engine import HfApiEngine, MessageRole  # type: ignore
    from transformers.agents.monitoring import Monitor  # type: ignore
    from transformers.agents.prompts import (  # type: ignore
        DEFAULT_CODE_SYSTEM_PROMPT,
        DEFAULT_REACT_CODE_SYSTEM_PROMPT,
        DEFAULT_REACT_JSON_SYSTEM_PROMPT,
        PLAN_UPDATE_FINAL_PLAN_REDACTION,
        PROMPTS_FOR_INITIAL_PLAN,
        PROMPTS_FOR_PLAN_UPDATE,
        SUPPORTED_PLAN_TYPES,
        SYSTEM_PROMPT_FACTS,
        SYSTEM_PROMPT_FACTS_UPDATE,
        USER_PROMPT_FACTS_UPDATE,
    )
    from transformers.agents.python_interpreter import (  # type: ignore
        LIST_SAFE_MODULES,
        evaluate_python_code,
    )
    from transformers.agents.tools import (  # type: ignore
        DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
        Tool,
        get_tool_description_with_args,
        load_tool,
    )
    HF_AVAILABLE = True
except Exception as _hf_exc:
    # Minimal stubs so this file is importable without transformers.
    class _DummyLogger(logging.Logger):
        pass

    transformers_logging = logging  # fall back to std logging

    def is_torch_available() -> bool:
        return False

    def is_pygments_available() -> bool:
        return False

    class AgentImage(bytes):  # simple placeholder
        pass

    class AgentAudio(bytes):  # simple placeholder
        pass

    class Tool:
        def __init__(self, name: str = "tool", description: str = "generic tool", **kwargs):
            self.name = name
            self.description = description
            self.repo_id = None
            self.task = kwargs.get("task")

        def __call__(self, *a, **kw):
            return f"[stub:{self.name}]"

    DEFAULT_TOOL_DESCRIPTION_TEMPLATE = "{name}: {description}"
    BASE_PYTHON_TOOLS: Dict[str, Tool] = {}

    class FinalAnswerTool(Tool):
        def __init__(self):
            super().__init__(name="final_answer", description="Return the final answer")

        def __call__(self, answer=None, **_):
            return answer

    def setup_default_tools(logger: logging.Logger) -> Dict[str, Tool]:
        return {}

    class MessageRole:
        SYSTEM = "system"
        USER = "user"
        ASSISTANT = "assistant"
        TOOL_RESPONSE = "tool_response"

    class HfApiEngine:
        def __call__(self, prompt, stop_sequences=None, **_):
            # Return a minimal valid action for both JSON and code agents
            return "Code:\n```py\nprint('ok')\n```<end_action>"

    class Monitor:
        def __init__(self, *_):
            pass

        def update_metrics(self, *_a, **_k):
            return None

    DEFAULT_CODE_SYSTEM_PROMPT = "You are a code agent."
    DEFAULT_REACT_CODE_SYSTEM_PROMPT = "You are a ReAct code agent."
    DEFAULT_REACT_JSON_SYSTEM_PROMPT = "You are a ReAct JSON agent."
    PLAN_UPDATE_FINAL_PLAN_REDACTION = "Plan update:\n{plan_update}\n"
    PROMPTS_FOR_INITIAL_PLAN = {
        "sequential": {
            "system": "Initial plan system",
            "user": "Initial plan user with tools: {tool_descriptions}\nFacts: {answer_facts}",
        }
    }
    PROMPTS_FOR_PLAN_UPDATE = {
        "sequential": {"system": "Update plan system for {task}", "user": "Update plan user, remaining {remaining_steps}"}
    }
    SUPPORTED_PLAN_TYPES = ["sequential"]
    SYSTEM_PROMPT_FACTS = "List facts."
    SYSTEM_PROMPT_FACTS_UPDATE = "Update facts."
    USER_PROMPT_FACTS_UPDATE = "Please provide updated facts."

    LIST_SAFE_MODULES: List[str] = []

    def evaluate_python_code(code_action: str, static_tools=None, custom_tools=None, state=None, authorized_imports=None):
        if state is not None:
            state["print_outputs"] = "[stub] no output"
        return None

# Configure pretty logger for this module
logger = transformers_logging.get_logger(__name__) if HF_AVAILABLE else logging.getLogger(__name__)
logger.propagate = False
_ch = logging.StreamHandler()
_ch.setFormatter(CustomFormatter())
logger.addHandler(_ch)
logger.setLevel(logging.INFO)

# ----------
# Utilities
# ----------
def parse_json_blob(json_blob: str) -> Dict[str, str]:
    try:
        first_accolade_index = json_blob.find("{")
        last_accolade_index = [a.start() for a in list(re.finditer("}", json_blob))][-1]
        json_blob = json_blob[first_accolade_index : last_accolade_index + 1].replace('\\"', "'")
        json_data = json.loads(json_blob, strict=False)
        return json_data
    except json.JSONDecodeError as e:
        place = e.pos
        if json_blob[place - 1 : place + 2] == "},\n":
            raise ValueError(
                "JSON is invalid: you probably tried to provide multiple tool calls in one action. PROVIDE ONLY ONE TOOL CALL."
            )
        raise ValueError(
            f"The JSON blob you used is invalid due to the following error: {e}.\n"
            f"JSON blob was: {json_blob}, decoding failed on that specific part of the blob:\n"
            f"'{json_blob[place - 4 : place + 5]}'."
        )
    except Exception as e:
        raise ValueError(f"Error in parsing the JSON blob: {e}")


def parse_code_blob(code_blob: str) -> str:
    try:
        pattern = r"```(?:py|python)?\n(.*?)\n```"
        match = re.search(pattern, code_blob, re.DOTALL)
        return match.group(1).strip()
    except Exception as e:
        raise ValueError(
            f"""
The code blob you used is invalid: due to the following error: {e}
This means that the regex pattern {pattern} was not respected: make sure to include code with the correct pattern, for instance:
Thoughts: Your thoughts
Code:
```py
# Your python code here
```<end_action>"""
        )


def parse_json_tool_call(json_blob: str) -> Tuple[str, Dict[str, str]]:
    json_blob = json_blob.replace("```json", "").replace("```", "")
    tool_call = parse_json_blob(json_blob)
    if "action" in tool_call and "action_input" in tool_call:
        return tool_call["action"], tool_call["action_input"]
    elif "action" in tool_call:
        return tool_call["action"], None
    else:
        raise ValueError(
            f"Missing keys: {[key for key in ['action', 'action_input'] if key not in tool_call]} in blob {tool_call}"
        )


def parse_text_tool_call(text: str) -> Tuple[str, Union[str, Dict[str, str]]]:
    """
    Expects a text in the format: 'Action:', 'Action input:', 'Observation:'. 'Action input:' contains a json string with input arguments.
    """
    try:
        if "Observation:" in text:
            text = text.split("Observation:")[0]
        if "Action:" in text:
            text = text.split("Action:")[1]
        tool_name, tool_input = text.split("Action input:")
        if "{" in tool_input:
            tool_input = parse_json_blob(tool_input)
        else:
            tool_input = tool_input.strip().replace('"', "")
        return tool_name.strip().replace('"', "").replace("\\", ""), tool_input
    except Exception as e:
        raise ValueError(
            f"Error in parsing the text tool call: {e}. Be sure to provide the correct format. DO NOT repeat your previous incorrect tool call."
        )


def to_text(input: Union[List[Dict[str, str]], Dict[str, str], str]) -> str:
    if isinstance(input, list):
        return "\n".join([m["content"] for m in input])
    elif isinstance(input, dict):
        return input["content"]
    else:
        return input


HUGGINGFACE_DEFAULT_TOOLS: Dict[str, Tool] = {}
_tools_are_initialized = False


class Toolbox:
    """
    The toolbox contains all tools that the agent can perform operations with, as well as a few methods to
    manage them.
    """

    def __init__(self, tools: List[Tool], add_base_tools: bool = False):
        self._tools = {tool.name: tool for tool in tools}
        if add_base_tools:
            self.add_base_tools()
        self._load_tools_if_needed()

    def add_base_tools(self, add_python_interpreter: bool = False):
        global _tools_are_initialized
        global HUGGINGFACE_DEFAULT_TOOLS
        if not _tools_are_initialized:
            HUGGINGFACE_DEFAULT_TOOLS = setup_default_tools(logger)
            _tools_are_initialized = True
        for tool in HUGGINGFACE_DEFAULT_TOOLS.values():
            if tool.name != "python_interpreter" or add_python_interpreter:
                self.add_tool(tool)
        self._load_tools_if_needed()

    @property
    def tools(self) -> Dict[str, Tool]:
        return self._tools

    def show_tool_descriptions(self, tool_description_template: str = None) -> str:
        return "\n".join(
            [get_tool_description_with_args(tool, tool_description_template or DEFAULT_TOOL_DESCRIPTION_TEMPLATE) for tool in self._tools.values()]  # type: ignore
        )

    def add_tool(self, tool: Tool):
        if tool.name in self._tools:
            raise KeyError(f"Error: tool '{tool.name}' already exists in the toolbox.")
        self._tools[tool.name] = tool

    def remove_tool(self, tool_name: str):
        if tool_name not in self._tools:
            raise KeyError(
                f"Error: tool {tool_name} not found in toolbox for removal, should be instead one of {list(self._tools.keys())}."
            )
        del self._tools[tool_name]

    def update_tool(self, tool: Tool):
        if tool.name not in self._tools:
            raise KeyError(
                f"Error: tool {tool.name} not found in toolbox for update, should be instead one of {list(self._tools.keys())}."
            )
        self._tools[tool.name] = tool

    def clear_toolbox(self):
        self._tools = {}

    def _load_tools_if_needed(self):
        for name, tool in list(self._tools.items()):
            if not isinstance(tool, Tool):
                try:
                    task_or_repo_id = getattr(tool, "task", None) if getattr(tool, "repo_id", None) is None else tool.repo_id
                    self._tools[name] = load_tool(task_or_repo_id)  # type: ignore
                except Exception:
                    # leave as-is if loading fails
                    pass

    def __repr__(self):
        toolbox_description = "Toolbox contents:\n"
        for tool in self._tools.values():
            toolbox_description += f"\t{tool.name}: {tool.description}\n"
        return toolbox_description


# -------------------
# Error base classes
# -------------------
class AgentError(Exception):
    def __init__(self, message):
        super().__init__(message)
        self.message = message


class AgentParsingError(AgentError):
    pass


class AgentExecutionError(AgentError):
    pass


class AgentMaxIterationsError(AgentError):
    pass


class AgentGenerationError(AgentError):
    pass


# --------------
# Prompt helpers
# --------------
def format_prompt_with_tools(toolbox: Toolbox, prompt_template: str, tool_description_template: str) -> str:
    tool_descriptions = toolbox.show_tool_descriptions(tool_description_template)
    prompt = prompt_template.replace("<<tool_descriptions>>", tool_descriptions)
    if "<<tool_names>>" in prompt:
        tool_names = [f"'{tool_name}'" for tool_name in toolbox.tools.keys()]
        prompt = prompt.replace("<<tool_names>>", ", ".join(tool_names))
    return prompt


def show_agents_descriptions(managed_agents: dict) -> str:
    managed_agents_descriptions = """
You can also give requests to team members.
Calling a team member works the same as for calling a tool: simply, the only argument you can give in the call is 'request', a long string explaining your request.
Given that this team member is a real human, you should be very verbose in your request.
Here is a list of the team members that you can call:"""
    for agent in managed_agents.values():
        managed_agents_descriptions += f"\n- {agent.name}: {agent.description}"
    return managed_agents_descriptions


def format_prompt_with_managed_agents_descriptions(prompt_template, managed_agents=None) -> str:
    if managed_agents is not None:
        return prompt_template.replace("<<managed_agents_descriptions>>", show_agents_descriptions(managed_agents))
    else:
        return prompt_template.replace("<<managed_agents_descriptions>>", "")


def format_prompt_with_imports(prompt_template: str, authorized_imports: List[str]) -> str:
    if "<<authorized_imports>>" not in prompt_template:
        raise AgentError("Tag '<<authorized_imports>>' should be provided in the prompt.")
    return prompt_template.replace("<<authorized_imports>>", str(authorized_imports))


# -------------
# Base Agent(s)
# -------------
class Agent:
    def __init__(
        self,
        tools: Union[List[Tool], Toolbox],
        llm_engine: Callable = None,
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        additional_args: Dict = {},
        max_iterations: int = 6,
        tool_parser: Optional[Callable] = None,
        add_base_tools: bool = False,
        verbose: int = 0,
        grammar: Optional[Dict[str, str]] = None,
        managed_agents: Optional[List] = None,
        step_callbacks: Optional[List[Callable]] = None,
        monitor_metrics: bool = True,
    ):
        if system_prompt is None:
            system_prompt = DEFAULT_REACT_CODE_SYSTEM_PROMPT
        if tool_parser is None:
            tool_parser = parse_json_tool_call
        self.agent_name = self.__class__.__name__
        self.llm_engine = llm_engine or HfApiEngine()
        self.system_prompt_template = system_prompt
        self.tool_description_template = (
            tool_description_template if tool_description_template else DEFAULT_TOOL_DESCRIPTION_TEMPLATE
        )
        self.additional_args = additional_args
        self.max_iterations = max_iterations
        self.logger = logger
        self.tool_parser = tool_parser
        self.grammar = grammar

        self.managed_agents = None
        if managed_agents is not None:
            self.managed_agents = {agent.name: agent for agent in managed_agents}

        if isinstance(tools, Toolbox):
            self._toolbox = tools
            if add_base_tools and is_torch_available():
                self._toolbox.add_base_tools(add_python_interpreter=(self.__class__ == ReactJsonAgent))
        else:
            self._toolbox = Toolbox(tools, add_base_tools=(add_base_tools and is_torch_available()))
        self._toolbox.add_tool(FinalAnswerTool())

        self.system_prompt = format_prompt_with_tools(
            self._toolbox, self.system_prompt_template, self.tool_description_template
        )
        self.system_prompt = format_prompt_with_managed_agents_descriptions(self.system_prompt, self.managed_agents)
        self.prompt = None
        self.logs = []
        self.task = None

        if verbose == 0:
            logger.setLevel(logging.WARNING)
        elif verbose == 1:
            logger.setLevel(logging.INFO)
        elif verbose == 2:
            logger.setLevel(logging.DEBUG)

        self.step_callbacks = step_callbacks if step_callbacks is not None else []
        self.monitor = None
        if monitor_metrics:
            self.monitor = Monitor(self.llm_engine)
            self.step_callbacks.append(self.monitor.update_metrics)

    @property
    def toolbox(self) -> Toolbox:
        return self._toolbox

    def initialize_for_run(self):
        self.token_count = 0
        self.system_prompt = format_prompt_with_tools(
            self._toolbox,
            self.system_prompt_template,
            self.tool_description_template,
        )
        self.system_prompt = format_prompt_with_managed_agents_descriptions(self.system_prompt, self.managed_agents)
        if hasattr(self, "authorized_imports"):
            self.system_prompt = format_prompt_with_imports(
                self.system_prompt, list(set(LIST_SAFE_MODULES) | set(self.authorized_imports))
            )
        self.logs = [{"system_prompt": self.system_prompt, "task": self.task}]
        self.logger.log(33, "======== New task ========")
        self.logger.log(34, self.task)
        self.logger.debug("System prompt is as follows:")
        self.logger.debug(self.system_prompt)

    def write_inner_memory_from_logs(self, summary_mode: Optional[bool] = False) -> List[Dict[str, str]]:
        prompt_message = {"role": MessageRole.SYSTEM, "content": self.logs[0]["system_prompt"]}
        task_message = {"role": MessageRole.USER, "content": "Task: " + self.logs[0]["task"]}
        if summary_mode:
            memory = [task_message]
        else:
            memory = [prompt_message, task_message]
        for i, step_log in enumerate(self.logs[1:]):
            if "llm_output" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": step_log["llm_output"].strip()}
                memory.append(thought_message)
            if "facts" in step_log:
                thought_message = {"role": MessageRole.ASSISTANT, "content": "[FACTS LIST]:\n" + step_log["facts"].strip()}
                memory.append(thought_message)
            if "plan" in step_log and not summary_mode:
                thought_message = {"role": MessageRole.ASSISTANT, "content": "[PLAN]:\n" + step_log["plan"].strip()}
                memory.append(thought_message)
            if "tool_call" in step_log and summary_mode:
                tool_call_message = {
                    "role": MessageRole.ASSISTANT,
                    "content": f"[STEP {i} TOOL CALL]: " + str(step_log["tool_call"]).strip(),
                }
                memory.append(tool_call_message)
            if "task" in step_log:
                tool_call_message = {"role": MessageRole.USER, "content": "New task:\n" + step_log["task"]}
                memory.append(tool_call_message)
            if "error" in step_log or "observation" in step_log:
                if "error" in step_log:
                    message_content = (
                        f"[OUTPUT OF STEP {i}] -> Error:\n"
                        + str(step_log["error"])
                        + "\nNow let's retry: take care not to repeat previous errors! If you have retried several times, try a completely different approach.\n"
                    )
                elif "observation" in step_log:
                    message_content = f"[OUTPUT OF STEP {i}] -> Observation:\n{step_log['observation']}"
                tool_response_message = {"role": MessageRole.TOOL_RESPONSE, "content": message_content}
                memory.append(tool_response_message)
        return memory

    def get_succinct_logs(self):
        return [{key: value for key, value in log.items() if key != "agent_memory"} for log in self.logs]

    def extract_action(self, llm_output: str, split_token: str) -> str:
        try:
            split = llm_output.split(split_token)
            rationale, action = (split[-2], split[-1])
        except Exception as e:
            self.logger.error(e, exc_info=1)
            raise AgentParsingError(
                f"Error: No '{split_token}' token provided in your output.\nYour output:\n{llm_output}\n. Be sure to include an action, prefaced with '{split_token}'!"
            )
        return rationale.strip(), action.strip()

    def execute_tool_call(self, tool_name: str, arguments: Dict[str, str]) -> Any:
        available_tools = self.toolbox.tools
        if self.managed_agents is not None:
            available_tools = {**available_tools, **self.managed_agents}
        if tool_name not in available_tools:
            error_msg = f"Error: unknown tool {tool_name}, should be instead one of {list(available_tools.keys())}."
            self.logger.error(error_msg, exc_info=1)
            raise AgentExecutionError(error_msg)

        try:
            if isinstance(arguments, str):
                observation = available_tools[tool_name](arguments)
            elif isinstance(arguments, dict):
                for key, value in arguments.items():
                    if isinstance(value, str) and hasattr(self, "state") and value in self.state:
                        arguments[key] = self.state[value]
                observation = available_tools[tool_name](**arguments)
            else:
                raise AgentExecutionError(
                    f"Arguments passed to tool should be a dict or string: got a {type(arguments)}."
                )
            return observation
        except Exception as e:
            if tool_name in self.toolbox.tools:
                raise AgentExecutionError(
                    f"Error in tool call execution: {e}\nYou should only use this tool with a correct input.\n"
                )
            elif self.managed_agents and tool_name in self.managed_agents:
                raise AgentExecutionError(f"Error in calling team member: {e}")

    def log_rationale_code_action(self, rationale: str, code_action: str) -> None:
        self.logger.warning("=== Agent thoughts:")
        self.logger.log(31, rationale)
        self.logger.warning(">>> Agent is executing the code below:")
        if is_pygments_available():
            try:
                from pygments import highlight  # type: ignore
                from pygments.formatters import Terminal256Formatter  # type: ignore
                from pygments.lexers import PythonLexer  # type: ignore
                self.logger.log(31, highlight(code_action, PythonLexer(ensurenl=False), Terminal256Formatter(style="nord")))
            except Exception:
                self.logger.log(31, code_action)
        else:
            self.logger.log(31, code_action)
        self.logger.warning("====")

    def run(self, **kwargs):
        raise NotImplementedError


class CodeAgent(Agent):
    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine or HfApiEngine(),
            system_prompt=system_prompt or DEFAULT_CODE_SYSTEM_PROMPT,
            tool_description_template=tool_description_template or DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
            grammar=grammar,
            **kwargs,
        )
        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        self.system_prompt = self.system_prompt.replace("<<authorized_imports>>", str(self.authorized_imports))

    def parse_code_blob(self, result: str) -> str:
        return parse_code_blob(result)

    def run(self, task: str, return_generated_code: bool = False, **kwargs):
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        self.initialize_for_run()

        prompt_message = {"role": MessageRole.SYSTEM, "content": self.system_prompt}
        task_message = {"role": MessageRole.USER, "content": "Task: " + self.task}
        self.prompt = [prompt_message, task_message]
        self.logger.info("====Executing with this prompt====")
        self.logger.info(self.prompt)

        additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
        llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>"], **additional_args)

        if return_generated_code:
            return llm_output

        try:
            rationale, code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output as code. Error trace: {e}")
            rationale, code_action = "", llm_output

        try:
            code_action = self.parse_code_blob(code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Be sure to provide correct code"
            self.logger.error(error_msg, exc_info=1)
            return error_msg

        self.log_rationale_code_action(rationale, code_action)
        try:
            available_tools = {**BASE_PYTHON_TOOLS.copy(), **self.toolbox.tools}
            output = self.python_evaluator(
                code_action,
                static_tools=available_tools,
                custom_tools={},
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            self.logger.info(self.state.get("print_outputs", ""))
            return output
        except Exception as e:
            error_msg = f"Error in execution: {e}. Be sure to provide correct code."
            self.logger.error(error_msg, exc_info=1)
            return error_msg


class ReactAgent(Agent):
    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        plan_type: Optional[str] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine or HfApiEngine(),
            system_prompt=system_prompt or DEFAULT_REACT_CODE_SYSTEM_PROMPT,
            tool_description_template=tool_description_template or DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
            grammar=grammar,
            **kwargs,
        )
        self.planning_interval = planning_interval
        self.plan_type = plan_type or SUPPORTED_PLAN_TYPES[0]

    def provide_final_answer(self, task) -> str:
        self.prompt = [{"role": MessageRole.SYSTEM, "content": "An agent tried to answer an user query but it got stuck and failed to do so. You are tasked with providing an answer instead. Here is the agent's memory:"}]
        self.prompt += self.write_inner_memory_from_logs()[1:]
        self.prompt += [{"role": MessageRole.USER, "content": f"Based on the above, please provide an answer to the following user request:\n{task}"}]
        try:
            return self.llm_engine(self.prompt)
        except Exception as e:
            return f"Error in generating final llm output: {e}."

    def run(self, task: str, stream: bool = False, reset: bool = True, **kwargs):
        self.task = task
        if len(kwargs) > 0:
            self.task += f"\nYou have been provided with these initial arguments: {str(kwargs)}."
        self.state = kwargs.copy()
        if reset:
            self.initialize_for_run()
        else:
            self.logs.append({"task": task})
        if stream:
            return self.stream_run(task)
        else:
            return self.direct_run(task)

    def stream_run(self, task: str):
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            step_start_time = time.time()
            step_log_entry = {"iteration": iteration, "start_time": step_start_time}
            try:
                self.step(step_log_entry)
                if "final_answer" in step_log_entry:
                    final_answer = step_log_entry["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                step_log_entry["error"] = e
            finally:
                step_end_time = time.time()
                step_log_entry["step_end_time"] = step_end_time
                step_log_entry["step_duration"] = step_end_time - step_start_time
                self.logs.append(step_log_entry)
                for callback in self.step_callbacks:
                    callback(step_log_entry)
                iteration += 1
                yield step_log_entry
        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
            final_step_log["step_duration"] = 0
            for callback in self.step_callbacks:
                callback(final_step_log)
            yield final_step_log
        yield final_answer

    def direct_run(self, task: str):
        final_answer = None
        iteration = 0
        while final_answer is None and iteration < self.max_iterations:
            step_start_time = time.time()
            step_log_entry = {"iteration": iteration, "start_time": step_start_time}
            try:
                if self.planning_interval is not None and iteration % self.planning_interval == 0:
                    self.planning_step(task, is_first_step=(iteration == 0), iteration=iteration)
                self.step(step_log_entry)
                if "final_answer" in step_log_entry:
                    final_answer = step_log_entry["final_answer"]
            except AgentError as e:
                self.logger.error(e, exc_info=1)
                step_log_entry["error"] = e
            finally:
                step_end_time = time.time()
                step_log_entry["step_end_time"] = step_end_time
                step_log_entry["step_duration"] = step_end_time - step_start_time
                self.logs.append(step_log_entry)
                for callback in self.step_callbacks:
                    callback(step_log_entry)
                iteration += 1
        if final_answer is None and iteration == self.max_iterations:
            error_message = "Reached max iterations."
            final_step_log = {"error": AgentMaxIterationsError(error_message)}
            self.logs.append(final_step_log)
            self.logger.error(error_message, exc_info=1)
            final_answer = self.provide_final_answer(task)
            final_step_log["final_answer"] = final_answer
            final_step_log["step_duration"] = 0
            for callback in self.step_callbacks:
                callback(final_step_log)
        return final_answer

    def planning_step(self, task, is_first_step: bool = False, iteration: int = None):
        if is_first_step:
            message_prompt_facts = {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT_FACTS}
            message_prompt_task = {"role": MessageRole.USER, "content": f"Here is the task:\n```\n{task}\n```\nNow begin!"}
            answer_facts = self.llm_engine([message_prompt_facts, message_prompt_task])
            message_system_prompt_plan = {"role": MessageRole.SYSTEM, "content": PROMPTS_FOR_INITIAL_PLAN[self.plan_type]["system"]}
            message_user_prompt_plan = {
                "role": MessageRole.USER,
                "content": PROMPTS_FOR_INITIAL_PLAN[self.plan_type]["user"].format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(DEFAULT_TOOL_DESCRIPTION_TEMPLATE),
                    managed_agents_descriptions=(show_agents_descriptions(self.managed_agents) if self.managed_agents is not None else ""),
                    answer_facts=answer_facts,
                ),
            }
            answer_plan = self.llm_engine([message_system_prompt_plan, message_user_prompt_plan], stop_sequences=["<end_plan>"])
            final_plan_redaction = f"Here is the plan of action that I will follow to solve the task:\n```\n{answer_plan}\n```"
            final_facts_redaction = f"Here are the facts that I know so far:\n```\n{answer_facts}\n```".strip()
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.log(36, "===== Initial plan =====")
            self.logger.log(35, final_plan_redaction)
        else:
            agent_memory = self.write_inner_memory_from_logs(summary_mode=False)
            facts_update_system_prompt = {"role": MessageRole.SYSTEM, "content": SYSTEM_PROMPT_FACTS_UPDATE}
            facts_update_message = {"role": MessageRole.USER, "content": USER_PROMPT_FACTS_UPDATE}
            facts_update = self.llm_engine([facts_update_system_prompt] + agent_memory + [facts_update_message])
            plan_update_message = {"role": MessageRole.SYSTEM, "content": PROMPTS_FOR_PLAN_UPDATE[self.plan_type]["system"].format(task=task)}
            plan_update_message_user = {
                "role": MessageRole.USER,
                "content": PROMPTS_FOR_PLAN_UPDATE[self.plan_type]["user"].format(
                    task=task,
                    tool_descriptions=self._toolbox.show_tool_descriptions(DEFAULT_TOOL_DESCRIPTION_TEMPLATE),
                    managed_agents_descriptions=(show_agents_descriptions(self.managed_agents) if self.managed_agents is not None else ""),
                    facts_update=facts_update,
                    remaining_steps=(self.max_iterations - iteration),
                ),
            }
            plan_update = self.llm_engine([plan_update_message] + agent_memory + [plan_update_message_user], stop_sequences=["<end_plan>"])
            final_plan_redaction = PLAN_UPDATE_FINAL_PLAN_REDACTION.format(task=task, plan_update=plan_update)
            final_facts_redaction = f"Here is the updated list of the facts that I know:\n```\n{facts_update}\n```"
            self.logs.append({"plan": final_plan_redaction, "facts": final_facts_redaction})
            self.logger.log(36, "===== Updated plan =====")
            self.logger.log(35, final_plan_redaction)


class ReactJsonAgent(ReactAgent):
    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine or HfApiEngine(),
            system_prompt=system_prompt or DEFAULT_REACT_JSON_SYSTEM_PROMPT,
            tool_description_template=tool_description_template or DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )

    def step(self, log_entry: Dict[str, Any]):
        agent_memory = self.write_inner_memory_from_logs()
        self.prompt = agent_memory
        self.logger.debug("===== New step =====")
        log_entry["agent_memory"] = agent_memory.copy()
        self.logger.info("===== Calling LLM with this last message: =====")
        self.logger.info(self.prompt[-1])
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"], **additional_args)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug("===== Output message of the LLM: =====")
        self.logger.debug(llm_output)
        log_entry["llm_output"] = llm_output
        self.logger.debug("===== Extracting action =====")
        rationale, action = self.extract_action(llm_output=llm_output, split_token="Action:")
        try:
            tool_name, arguments = parse_json_tool_call(action)
        except Exception as e:
            raise AgentParsingError(f"Could not parse the given action: {e}.")
        log_entry["rationale"] = rationale
        log_entry["tool_call"] = {"tool_name": tool_name, "tool_arguments": arguments}
        self.logger.warning("=== Agent thoughts:")
        self.logger.log(31, rationale)
        self.logger.warning(f">>> Calling tool: '{tool_name}' with arguments: {arguments}")
        if tool_name == "final_answer":
            if isinstance(arguments, dict):
                if "answer" in arguments:
                    answer = arguments["answer"]
                    if isinstance(answer, str) and hasattr(self, "state") and answer in self.state:
                        answer = self.state[answer]
                else:
                    answer = arguments
            else:
                answer = arguments
            log_entry["final_answer"] = answer
            return answer
        else:
            if arguments is None:
                arguments = {}
            observation = self.execute_tool_call(tool_name, arguments)
            if isinstance(observation, (AgentImage, AgentAudio)):
                observation_name = "image.png" if isinstance(observation, AgentImage) else "audio.mp3"
                self.state[observation_name] = observation
                updated_information = f"Stored '{observation_name}' in memory."
            else:
                updated_information = str(observation).strip()
            self.logger.info(updated_information)
            log_entry["observation"] = updated_information
            return log_entry


class ReactCodeAgent(ReactAgent):
    def __init__(
        self,
        tools: List[Tool],
        llm_engine: Optional[Callable] = None,
        system_prompt: Optional[str] = None,
        tool_description_template: Optional[str] = None,
        grammar: Optional[Dict[str, str]] = None,
        additional_authorized_imports: Optional[List[str]] = None,
        planning_interval: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            tools=tools,
            llm_engine=llm_engine or HfApiEngine(),
            system_prompt=system_prompt or DEFAULT_REACT_CODE_SYSTEM_PROMPT,
            tool_description_template=tool_description_template or DEFAULT_TOOL_DESCRIPTION_TEMPLATE,
            grammar=grammar,
            planning_interval=planning_interval,
            **kwargs,
        )
        self.python_evaluator = evaluate_python_code
        self.additional_authorized_imports = additional_authorized_imports if additional_authorized_imports else []
        self.authorized_imports = list(set(LIST_SAFE_MODULES) | set(self.additional_authorized_imports))
        self.system_prompt = self.system_prompt.replace("<<authorized_imports>>", str(self.authorized_imports))
        self.custom_tools: Dict[str, Tool] = {}

    def step(self, log_entry: Dict[str, Any]):
        agent_memory = self.write_inner_memory_from_logs()
        self.prompt = agent_memory.copy()
        self.logger.debug("===== New step =====")
        log_entry["agent_memory"] = agent_memory.copy()
        self.logger.info("===== Calling LLM with these last messages: =====")
        self.logger.info(self.prompt[-2:])
        try:
            additional_args = {"grammar": self.grammar} if self.grammar is not None else {}
            llm_output = self.llm_engine(self.prompt, stop_sequences=["<end_action>", "Observation:"], **additional_args)
        except Exception as e:
            raise AgentGenerationError(f"Error in generating llm output: {e}.")
        self.logger.debug("=== Output message of the LLM:")
        self.logger.debug(llm_output)
        log_entry["llm_output"] = llm_output
        self.logger.debug("=== Extracting action ===")
        try:
            rationale, raw_code_action = self.extract_action(llm_output=llm_output, split_token="Code:")
        except Exception as e:
            self.logger.debug(f"Error in extracting action, trying to parse the whole output. Error trace: {e}")
            rationale, raw_code_action = llm_output, llm_output
        try:
            code_action = parse_code_blob(raw_code_action)
        except Exception as e:
            error_msg = f"Error in code parsing: {e}. Make sure to provide correct code"
            raise AgentParsingError(error_msg)
        log_entry["rationale"] = rationale
        log_entry["tool_call"] = {"tool_name": "code interpreter", "tool_arguments": code_action}
        self.log_rationale_code_action(rationale, code_action)
        try:
            static_tools = {**BASE_PYTHON_TOOLS.copy(), **self.toolbox.tools}
            if self.managed_agents is not None:
                static_tools = {**static_tools, **self.managed_agents}
            result = self.python_evaluator(
                code_action,
                static_tools=static_tools,
                custom_tools=self.custom_tools,
                state=self.state,
                authorized_imports=self.authorized_imports,
            )
            self.logger.warning("Print outputs:")
            self.logger.log(32, self.state.get("print_outputs", ""))
            observation = "Print outputs:\n" + str(self.state.get("print_outputs", ""))
            if result is not None:
                self.logger.warning("Last output from code snippet:")
                self.logger.log(32, str(result))
                observation += "Last output from code snippet:\n" + str(result)[:100000]
            log_entry["observation"] = observation
        except Exception as e:
            error_msg = f"Code execution failed due to the following error:\n{str(e)}"
            if "'dict' object has no attribute 'read'" in str(e):
                error_msg += "\nYou get this error because you passed a dict as input for one of the arguments instead of a string."
            raise AgentExecutionError(error_msg)
        for line in code_action.split("\n"):
            if line[: len("final_answer")] == "final_answer":
                self.logger.log(33, "Final answer:")
                self.logger.log(32, result if "result" in locals() else "")
                log_entry["final_answer"] = result if "result" in locals() else None
        return log_entry.get("final_answer", None)


LENGTH_TRUNCATE_REPORTS = 1000


class ManagedAgent:
    def __init__(self, agent, name, description, additional_prompting=None, provide_run_summary=False):
        self.agent = agent
        self.name = name
        self.description = description
        self.additional_prompting = additional_prompting
        self.provide_run_summary = provide_run_summary

    def write_full_task(self, task):
        full_task = f"""You're a helpful agent named '{self.name}'.
You have been submitted this task by your manager.
---
Task:
{task}
---
You're helping your manager solve a wider task: so make sure to not provide a one-line answer, but give as much information as possible so that they have a clear understanding of the answer.

Your final_answer WILL HAVE to contain these parts:
### 1. Task outcome (short version):
### 2. Task outcome (extremely detailed version):
### 3. Additional context (if relevant):

Put all these in your final_answer tool, everything that you do not pass as an argument to final_answer will be lost.
And even if your task resolution is not successful, please return as much context as possible, so that your manager can act upon this feedback.
<<additional_prompting>>"""
        if self.additional_prompting:
            full_task = full_task.replace("\n<<additional_prompting>>", self.additional_prompting).strip()
        else:
            full_task = full_task.replace("\n<<additional_prompting>>", "").strip()
        return full_task

    def __call__(self, request, **kwargs):
        full_task = self.write_full_task(request)
        output = self.agent.run(full_task, **kwargs)
        if self.provide_run_summary:
            answer = f"Here is the final answer from your managed agent '{self.name}':\n"
            answer += str(output)
            answer += f"\n\nFor more detail, find below a summary of this agent's work:\nSUMMARY OF WORK FROM AGENT '{self.name}':\n"
            for message in self.agent.write_inner_memory_from_logs(summary_mode=True):
                content = message["content"]
                if len(str(content)) < LENGTH_TRUNCATE_REPORTS or "[FACTS LIST]" in str(content):
                    answer += "\n" + str(content) + "\n---"
                else:
                    answer += "\n" + str(content)[:LENGTH_TRUNCATE_REPORTS] + "\n(...Step was truncated because too long)...\n---"
            answer += f"\nEND OF SUMMARY OF WORK FROM AGENT '{self.name}'."
            return answer
        else:
            return output


# --------------------------
# Script entry (safe no-op)
# --------------------------
def _print_env():
    print("[agents.py] HF_AVAILABLE =", HF_AVAILABLE)
    try:
        import transformers  # type: ignore
        print("[agents.py] transformers version:", getattr(transformers, "__version__", "unknown"))
    except Exception as e:
        print("[agents.py] transformers import failed:", e)


if __name__ == "__main__":
    # When controller runs this file as a subprocess, just emit environment info and exit.
    _print_env()
    # Print out a tiny tools list so upstream logs are informative
    tb = Toolbox(tools=[], add_base_tools=False)
    print("[agents.py] Toolbox:", tb.tools)
    sys.exit(0)
