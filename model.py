from pathlib import Path
from typing import List, Dict
from api.runner import BaseModelPlugin

class Model(BaseModelPlugin):
    """
    TEMPLATE — copy this folder to models/<your_model>/ and implement:
     - load(): initialize your model from files in model_dir
     - chat(): run inference on messages
    """
    def load(self, model_dir: Path):
        self.name = model_dir.name
        # TODO: load tokenizer/weights from model_dir / "assets"
        self.system_hint = (model_dir / "config.yaml").read_text() if (model_dir / "config.yaml").exists() else ""

    def chat(self, messages: List[Dict], max_tokens: int, temperature: float) -> str:
        # TODO: replace with your inference.
        # This is just a placeholder that echoes the last user prompt.
        user_last = next((m["content"] for m in reversed(messages) if m.get("role")=="user"), "")
        return f"[{self.name}] (stub) You asked: {user_last[:200]}..."
