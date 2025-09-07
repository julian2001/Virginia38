import importlib.util
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Any

class BaseModelPlugin:
    """
    Your model plugin should implement:
      - def load(self, model_dir: Path): ...
      - def chat(self, messages: List[dict], max_tokens: int, temperature: float) -> str
    """
    def load(self, model_dir: Path):
        raise NotImplementedError

    def chat(self, messages: List[dict], max_tokens: int, temperature: float) -> str:
        raise NotImplementedError

def _pip_install_requirements(req_file: Path):
    if req_file.exists():
        print(f"[boot] Installing model requirements: {req_file}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", str(req_file)])

def _load_module_from(model_dir: Path):
    module_path = model_dir / "model.py"
    if not module_path.exists():
        raise FileNotFoundError(f"No model.py in {model_dir}")
    spec = importlib.util.spec_from_file_location(f"model_{model_dir.name}", str(module_path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod

class ModelRegistry:
    def __init__(self, models_dir: str = "models", default_model: str = None):
        self.models_dir = Path(models_dir)
        self.models: Dict[str, BaseModelPlugin] = {}
        self.active_name = None
        # auto-install any root-level model requirements (optional per-model)
        for sub in self.models_dir.glob("*/requirements.txt"):
            try:
                _pip_install_requirements(sub)
            except subprocess.CalledProcessError as e:
                print(f"[warn] pip install failed for {sub}: {e}")
        # load default if present
        if default_model:
            self.load(default_model)

    def list_models(self) -> List[str]:
        return sorted([p.name for p in self.models_dir.iterdir() if p.is_dir() and (p / "model.py").exists()])

    def load(self, name: str):
        model_dir = self.models_dir / name
        if not model_dir.exists():
            raise FileNotFoundError(f"Model folder not found: {model_dir}")
        req = model_dir / "requirements.txt"
        try:
            _pip_install_requirements(req)
        except subprocess.CalledProcessError as e:
            print(f"[warn] install failed for {req}: {e}")
        mod = _load_module_from(model_dir)
        if not hasattr(mod, "Model"):
            raise AttributeError(f"{model_dir}/model.py must define class Model(BaseModelPlugin)")
        inst = mod.Model()
        inst.load(model_dir)
        self.models[name] = inst
        self.active_name = name
        print(f"[boot] Loaded model: {name}")

    def get(self, name: str) -> BaseModelPlugin:
        if name not in self.models:
            self.load(name)
        return self.models[name]
