#!/usr/bin/env python3
"""Write the qai-hub client config from the repo's ``.env``.

qai-hub reads ``~/.qai_hub/client.ini`` (not an env var). This pulls
``QAIHUB_API_KEY`` out of ``.env`` and writes that file, so the other
scripts in this directory work without manual setup.

Usage:
    python3 scripts/qaihub/configure.py
"""

from __future__ import annotations

import configparser
import os
from pathlib import Path

API_URL = "https://workbench.aihub.qualcomm.com"
WEB_URL = "https://workbench.aihub.qualcomm.com"

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_env_key() -> str:
    env_path = REPO_ROOT / ".env"
    key = ""
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("QAIHUB_API_KEY="):
                key = line.split("=", 1)[1].strip().split("#", 1)[0].strip()
    key = key or os.environ.get("QAIHUB_API_KEY", "")
    if not key or "your_api_key_here" in key:
        raise SystemExit(
            "no QAIHUB_API_KEY found — put it in .env or the environment. "
            "Get one at https://workbench.aihub.qualcomm.com"
        )
    return key


def main() -> int:
    key = load_env_key()
    config_dir = Path.home() / ".qai_hub"
    config_dir.mkdir(exist_ok=True)
    cfg = configparser.ConfigParser()
    cfg["api"] = {
        "api_url": API_URL,
        "web_url": WEB_URL,
        "api_token": key,
    }
    config_path = config_dir / "client.ini"
    with config_path.open("w") as f:
        cfg.write(f)
    config_path.chmod(0o600)
    print(f"wrote {config_path} (api_token set, not printed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
