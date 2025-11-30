from pathlib import Path
import yaml


def _config_paths():
    root = Path(__file__).parent.parent / "configs"
    return [
        root / "config_local.yaml",
        root / "config_submit.yaml",
    ]


def load_config() -> dict:
    """Load the first available config file (local overrides submit)."""
    for path in _config_paths():
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                data = yaml.safe_load(handle) or {}
                return data
    raise FileNotFoundError("No configuration file found in configs/ directory.")
