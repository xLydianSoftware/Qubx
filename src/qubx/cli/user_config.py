"""Persistent user preferences for CLI commands, stored at ~/.qubx/init.json."""

import json
from pathlib import Path

QUBX_DIR = Path.home() / ".qubx"
INIT_CONFIG_FILE = QUBX_DIR / "init.json"


def load_init_config() -> dict:
    if INIT_CONFIG_FILE.exists():
        return json.loads(INIT_CONFIG_FILE.read_text())
    return {}


def save_init_config(updates: dict):
    config = load_init_config()
    config.update({k: v for k, v in updates.items() if v})
    QUBX_DIR.mkdir(parents=True, exist_ok=True)
    INIT_CONFIG_FILE.write_text(json.dumps(config, indent=2) + "\n")
