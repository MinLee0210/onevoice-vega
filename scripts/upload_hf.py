#!/usr/bin/env python3
"""Push the on-device model bundle (`models/`) to Hugging Face.

Mirrors the GitHub Release bundle that `scripts/pack_models.sh` builds —
same tree, hosted on HF so `setup_models.sh rebuild` (and the public) can
pull it. Reads `HF_API_KEY` from the repo `.env`.

Usage:
    python3 scripts/upload_hf.py
"""

from __future__ import annotations

from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "LakoreAI/onevoice-vega"
REPO_ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    for line in (REPO_ROOT / ".env").read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().split("#", 1)[0].strip()
    return env


def main() -> int:
    token = load_env().get("HF_API_KEY", "")
    if not token or "your" in token:
        raise SystemExit("no HF_API_KEY found in .env")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(REPO_ROOT / "models"),
        repo_id=REPO_ID,
        repo_type="model",
        # models/README.md is the git repo's internal docs, not a HF model
        # card — the real card is uploaded as README.md below.
        ignore_patterns=[".gitkeep", "**/.gitkeep", "README.md"],
    )
    # Upload the model card as the HF repo's README (replaces models/README.md,
    # which is the git repo's internal docs, not a HF model card).
    card = (REPO_ROOT / "docs" / "hf_model_card.md").read_text()
    api.upload_file(
        path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=REPO_ID, repo_type="model"
    )
    print(f"pushed models/ + model card -> https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
