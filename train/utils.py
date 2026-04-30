"""
utils.py

Shared helpers for the train/ scripts. Currently:
- resolve_template(): locate or download the qwen3_soft_switch.jinja2 chat
  template OneRec uses for the recommendation task.
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path
from typing import Optional

# Override via env var if the upstream layout ever moves.
DEFAULT_TEMPLATE_URL = os.environ.get(
    "ONEREC_TEMPLATE_URL",
    "https://raw.githubusercontent.com/Kuaishou-OneRec/OpenOneRec/main/"
    "benchmarks/benchmark/tasks/v1_0/qwen3_soft_switch.jinja2",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROJECT_TEMPLATE = PROJECT_ROOT / "oneRec" / "qwen3_soft_switch.jinja2"
CACHE_DIR = Path.home() / ".cache" / "onerec_template"
CACHE_TEMPLATE = CACHE_DIR / "qwen3_soft_switch.jinja2"


def resolve_template(path: Optional[str] = None,
                     url: Optional[str] = None) -> Path:
    """
    Find the qwen3_soft_switch.jinja2 chat template.

    Resolution order:
      1. `path` (explicit CLI flag), if it exists
      2. <project_root>/oneRec/qwen3_soft_switch.jinja2
      3. ~/.cache/onerec_template/qwen3_soft_switch.jinja2
      4. Download from `url` (or ONEREC_TEMPLATE_URL env var,
         or DEFAULT_TEMPLATE_URL) into (3) and use it.

    Returns the resolved Path. Raises RuntimeError with actionable advice
    if all four steps fail.
    """
    if path:
        p = Path(path).expanduser()
        if p.exists():
            return p
        print(f"[template] explicit --template not found: {p}; falling back ...")

    if PROJECT_TEMPLATE.exists():
        return PROJECT_TEMPLATE

    if CACHE_TEMPLATE.exists():
        return CACHE_TEMPLATE

    src = url or DEFAULT_TEMPLATE_URL
    print(f"[template] not found locally; downloading from\n  {src}")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    try:
        urllib.request.urlretrieve(src, CACHE_TEMPLATE)
    except Exception as e:
        raise RuntimeError(
            f"Could not download chat template from {src}: {e}\n"
            f"Place qwen3_soft_switch.jinja2 manually at one of:\n"
            f"  {PROJECT_TEMPLATE}\n"
            f"  {CACHE_TEMPLATE}\n"
            f"or pass an explicit path via --template, "
            f"or set ONEREC_TEMPLATE_URL to a working URL."
        )
    print(f"[template] cached to {CACHE_TEMPLATE}")
    return CACHE_TEMPLATE
