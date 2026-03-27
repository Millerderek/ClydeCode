import os
from pathlib import Path

# User identity
CLYDE_USER = os.environ.get("CLYDE_USER", "derek")

# Paths
CLYDE_DATA_DIR = Path(os.environ.get("CLYDE_DATA_DIR", str(Path.home() / ".clyde-memory")))
CLYDE_STATE_DIR = Path(os.environ.get("CLYDE_STATE_DIR", str(CLYDE_DATA_DIR / "state")))
CLYDE_LOG_DIR = Path(os.environ.get("CLYDE_LOG_DIR", "/var/log"))

# Claude Code project dirs (colon-separated list)
_raw_dirs = os.environ.get("CLYDE_CLAUDE_DIRS", str(Path.home() / ".claude/projects"))
CLAUDE_PROJECT_DIRS = [Path(d) for d in _raw_dirs.split(":") if d]

# Session markdown dir (for OpenClaw session_ingest)
SESSION_MEMORY_DIR = Path(os.environ.get("CLYDE_SESSION_MEMORY_DIR", str(CLYDE_DATA_DIR / "sessions")))

# OpenRouter API key (for Haiku extraction)
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_KEY_FILE = Path(os.environ.get("OPENROUTER_KEY_FILE", str(CLYDE_DATA_DIR / "openrouter.env")))

# Database
PG_HOST = os.environ.get("CLYDE_PG_HOST", "127.0.0.1")
PG_PORT = int(os.environ.get("CLYDE_PG_PORT", "5432"))
PG_USER = os.environ.get("CLYDE_PG_USER", "openclaw")
PG_DB = os.environ.get("CLYDE_PG_DB", "openclaw_memory")
PG_PASSWORD = os.environ.get("CLYDE_PG_PASSWORD", "")

REDIS_HOST = os.environ.get("CLYDE_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("CLYDE_REDIS_PORT", "6379"))

# Mem0 / Vector store
QDRANT_HOST = os.environ.get("CLYDE_QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("CLYDE_QDRANT_PORT", "6333"))

# Agent type: "claude-code", "openclaw", or "auto"
AGENT_TYPE = os.environ.get("CLYDE_AGENT_TYPE", "auto")

# User-defined named entities for context scoring (pipe-separated)
USER_ENTITIES = [e for e in os.environ.get("CLYDE_USER_ENTITIES", "").split("|") if e]

# User-defined topic labels (JSON dict: {"topic_name": "regex_pattern|pattern2"})
USER_TOPICS = {}
_raw_topics = os.environ.get("CLYDE_USER_TOPICS", "")
if _raw_topics:
    import json
    try:
        USER_TOPICS = json.loads(_raw_topics)
    except Exception:
        pass

# Ensure state dir exists
CLYDE_STATE_DIR.mkdir(parents=True, exist_ok=True)
