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
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    data = yaml.safe_load(handle) or {}
                    return data
            except yaml.YAMLError as e:
                raise ValueError(f"Failed to parse config file {path}: {e}")
            except Exception as e:
                raise IOError(f"Failed to read config file {path}: {e}")
    raise FileNotFoundError(
        f"No configuration file found. Searched paths: {[str(p) for p in _config_paths()]}"
    )
