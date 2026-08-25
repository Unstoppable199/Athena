"""Small, environment-overridable settings for Athena.

Defaults preserve the current local setup. A cloned repository can override
them in ``.env`` without editing source files; ``.env`` itself is ignored by
Git so machine-specific paths and preferences are not published.
"""

import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def _integer(name: str, default: int) -> int:
    """Read a positive integer, falling back safely on invalid input."""

    try:
        value = int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default

    return value if value > 0 else default


# Local web server. Keeping the loopback default is part of Athena's privacy
# boundary; changing it exposes the unauthenticated interface to a network.
ATHENA_HOST = os.environ.get("ATHENA_HOST", "127.0.0.1")
ATHENA_PORT = _integer("ATHENA_PORT", 8000)

# Files created by Athena. The default keeps the existing project layout;
# ATHENA_DATA_DIR lets a packaged installation place mutable data elsewhere.
DATA_DIR = Path(os.environ.get("ATHENA_DATA_DIR", PROJECT_ROOT)).expanduser()
CONVERSATION_DIR = DATA_DIR / "conversations"
WORKSPACE_DIR = DATA_DIR / "workspace"

MAX_FILE_SIZE = _integer("ATHENA_MAX_FILE_SIZE", 10 * 1024 * 1024)
OLLAMA_TIMEOUT_SECONDS = _integer("ATHENA_OLLAMA_TIMEOUT_SECONDS", 180)

# Output ceilings keep a malformed or looping local generation from occupying
# the interface until the HTTP timeout. These are deliberately generous: 1,536
# model tokens is roughly a thousand words, while generated programs get a
# much larger allowance because a presentation builder can legitimately be
# long. They are safety ceilings, not target answer lengths.
RESPONSE_MAX_TOKENS = _integer("ATHENA_RESPONSE_MAX_TOKENS", 1536)
PLANNER_MAX_TOKENS = _integer("ATHENA_PLANNER_MAX_TOKENS", 1024)
ROUTER_MAX_TOKENS = _integer("ATHENA_ROUTER_MAX_TOKENS", 64)
CODE_MAX_TOKENS = _integer("ATHENA_CODE_MAX_TOKENS", 4096)
# Repairing generated code is paid only when a script actually fails. Two
# attempts are enough to recover from a bad first rewrite without allowing a
# broken generator to loop indefinitely.
CODE_REPAIR_ATTEMPTS = _integer("ATHENA_CODE_REPAIR_ATTEMPTS", 2)

# Model names are configurable because Ollama tags and available hardware vary.
BALANCED_MODEL = os.environ.get("ATHENA_BALANCED_MODEL", "gemma3:12b")
FAST_MODEL = os.environ.get("ATHENA_FAST_MODEL", "qwen3:8b")
FAST_VISION_MODEL = os.environ.get("ATHENA_FAST_VISION_MODEL", "gemma3:4b")
EMBED_MODEL = os.environ.get("ATHENA_EMBED_MODEL", "nomic-embed-text")
