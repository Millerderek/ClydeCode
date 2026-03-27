# ClydeMemory

Persistent memory stack for AI agents — compatible with Claude Code and OpenClaw-style agents.

ClydeMemory gives your AI agent a long-term memory that persists across sessions, compactions, and restarts. Facts flow automatically from conversation logs into a hybrid vector + structured store, then get injected back into context when relevant.

---

## Architecture

```
[Agent Sessions]
      │
      ▼
┌─────────────────────────────────────────────────────┐
│  Ingestion Layer                                     │
│  conversation_digest.py  — JSONL → facts via Haiku  │
│  compaction_watcher.py   — hot-watch, every 5 min   │
│  session_ingest.py       — markdown session files   │
└────────────────────┬────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────┐
│  memo_daemon.py  — persistent Unix socket server    │
│  (warm Mem0 instance, skips ~7s cold start)         │
└────────┬──────────────┬──────────────┬──────────────┘
         │              │              │
         ▼              ▼              ▼
    [Qdrant]       [PostgreSQL]    [Redis]
    vectors        keyscores       entity cache
    semantic       TTLs, decay     pre-filter
    search         access counts   fast lookup
```

**Qdrant** — vector store for semantic similarity search (cosine distance over embeddings).

**PostgreSQL** — structured store for memory metadata: keyscores, confidence levels, access counts, TTL expiry, contradiction tracking.

**Redis** — cache layer for entity→memory mappings and keyscore pre-filter. Serves hot-path lookups in microseconds before the vector search runs.

**Mem0** — the retrieval API layer, wrapping Qdrant with an add/search/delete interface. ClydeMemory adds scoring, gating, and topic classification on top.

---

## Key Components

### `context_gate.py`
Heuristic classifier that decides whether a prompt needs memory context. No LLM call — pure pattern matching. Returns a score from 0.0 (self-contained) to 1.0 (needs context). Threshold at 0.40.

Also classifies topics and detects store-worthy events in assistant responses.

### `conversation_digest.py`
Reads Claude Code JSONL session files, extracts facts via OpenRouter (claude-haiku), deduplicates against existing memories, and stores via daemon socket. Runs every 4 hours via cron.

### `compaction_watcher.py`
Hot-watch version of conversation_digest. Runs every 5 minutes, only processes sessions modified in the last 30 minutes. Closes the gap where context compaction fires mid-session before the full digest runs.

### `session_ingest.py`
Ingests markdown session files (OpenClaw-style agent sessions). Parses `assistant:` / `user:` blocks, extracts facts, stores via daemon.

### `memo_daemon.py`
Persistent Unix socket server that holds a warm Mem0 Memory instance. Handles search, add, delete, and topic operations. Callers connect via socket and get responses in milliseconds instead of waiting for cold initialization.

### `openclaw_memo.py`
CLI and library interface. Handles vault-based or environment-variable-based API key loading, impact detection, and boosted search display.

### `db.py`
Native psycopg2 + redis-py connections to PostgreSQL and Redis. Replaces subprocess `docker exec` calls.

### `config.py`
Centralized configuration via environment variables with sensible defaults. All paths, credentials, and user identity loaded here.

---

## Quick Start

### 1. Start the infrastructure

```bash
# Copy and configure environment
cp config.example.env .env
# Edit .env — set OPENAI_API_KEY, ANTHROPIC_API_KEY, CLYDE_PG_PASSWORD at minimum
source .env

# Start Docker services
docker compose up -d

# Verify all three are healthy
docker compose ps
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Start the daemon

```bash
python3 memo_daemon.py start

# Verify it's running
python3 memo_daemon.py status
```

### 4. Test the memory

```bash
# Add a fact
python3 openclaw_memo.py add "myapp serves on port 5100 behind nginx"

# Search
python3 openclaw_memo.py search "what port does myapp use"

# List all memories
python3 openclaw_memo.py list
```

---

## Usage with Claude Code Agents

### Automatic ingestion via cron

Set up the cron schedule from `crontab.example`:

```bash
crontab -e
# Paste the contents of crontab.example, update paths
```

This runs:
- `compaction_watcher.py` every 5 minutes (hot-ingest active sessions)
- `conversation_digest.py` every 4 hours (deep extraction of all recent sessions)

### Context injection

In your agent's system prompt or pre-prompt hook, call the daemon to retrieve relevant context:

```python
from daemon_client import memo_search

results = memo_search(user_prompt, user_id="user", limit=5)
if results:
    context = "\n".join(r["memory"] for r in results)
    # Prepend context to prompt
```

The context gate (`context_gate.py`) is applied automatically by the daemon — self-contained prompts skip retrieval entirely.

### Manual memory addition

```bash
python3 openclaw_memo.py add "staging server is at 10.0.1.100, running nginx 1.25"
python3 openclaw_memo.py add "prefer httpx over requests for async-compatible code" --impact high
```

---

## Usage with OpenClaw Agents

OpenClaw agents write session summaries as markdown files. Point `session_ingest.py` at your session directory:

```bash
export CLYDE_SESSION_MEMORY_DIR=/path/to/your/sessions
python3 session_ingest.py
```

Or set it in `.env` and run via cron every 2 minutes.

---

## Memory Pipeline

Facts flow through three paths depending on timing:

### Path 1: Compaction Watcher (hot, ~5 min lag)
1. Agent session file is written/updated
2. `compaction_watcher.py` detects change (tail hash diff)
3. Haiku extracts 3-6 facts from recent transcript
4. Facts stored tagged with topic bucket
5. Available for retrieval within one cron cycle

### Path 2: Conversation Digest (deep, ~4 hour lag)
1. Same JSONL files, full tail read
2. Haiku extraction with stronger deduplication
3. Runs topic compaction after each batch
4. Handles sessions that compaction_watcher missed

### Path 3: Session Ingest (OpenClaw sessions, ~2 min lag)
1. Agent writes markdown session summary
2. `session_ingest.py` parses `assistant:` blocks
3. Facts extracted and stored
4. Topic classification applied

### Retrieval
1. Prompt arrives at agent
2. `context_gate.py` scores it (0.0–1.0)
3. If score >= 0.40: semantic search via Qdrant
4. Results boosted by entity cache (Redis) and keyscores (PostgreSQL)
5. Top N results injected into context

---

## Configuration Reference

All configuration is via environment variables. See `config.example.env` for the full list.

| Variable | Default | Description |
|---|---|---|
| `CLYDE_USER` | `user` | User identity for memory scoping |
| `CLYDE_DATA_DIR` | `~/.clyde-memory` | Root data directory |
| `CLYDE_STATE_DIR` | `~/.clyde-memory/state` | State files for ingestion tracking |
| `CLYDE_LOG_DIR` | `/var/log` | Log file directory |
| `CLYDE_CLAUDE_DIRS` | `~/.claude/projects` | Colon-separated list of Claude Code project dirs |
| `CLYDE_SESSION_MEMORY_DIR` | `~/.clyde-memory/sessions` | OpenClaw session markdown files |
| `CLYDE_AGENT_TYPE` | `auto` | `claude-code`, `openclaw`, or `auto` |
| `OPENAI_API_KEY` | — | For text-embedding-3-small |
| `ANTHROPIC_API_KEY` | — | For Haiku LLM reasoning |
| `OPENROUTER_API_KEY` | — | For extraction via OpenRouter |
| `CLYDE_PG_HOST` | `127.0.0.1` | PostgreSQL host |
| `CLYDE_PG_PORT` | `5432` | PostgreSQL port |
| `CLYDE_PG_USER` | `clyde` | PostgreSQL user |
| `CLYDE_PG_DB` | `clyde_memory` | PostgreSQL database name |
| `CLYDE_PG_PASSWORD` | — | PostgreSQL password |
| `CLYDE_REDIS_HOST` | `127.0.0.1` | Redis host |
| `CLYDE_REDIS_PORT` | `6379` | Redis port |
| `CLYDE_QDRANT_HOST` | `localhost` | Qdrant host |
| `CLYDE_QDRANT_PORT` | `6333` | Qdrant port |
| `CLYDE_SOCK_PATH` | `/tmp/clyde-memo.sock` | Daemon Unix socket path |
| `CLYDE_USER_ENTITIES` | — | Pipe-separated entity names that boost search scores |
| `CLYDE_USER_TOPICS` | — | JSON dict of custom topic patterns |

### Custom entity scoring

Add your project/service/client names to `CLYDE_USER_ENTITIES` and they will boost memory search scores when mentioned in prompts:

```bash
export CLYDE_USER_ENTITIES="myproject|myclient|myservice|myhostname"
```

### Custom topic patterns

Override or extend topic classification:

```bash
export CLYDE_USER_TOPICS='{"myapp": "myapp|my-app|myservice", "infra": "k8s|terraform|ansible"}'
```

---

## Gap-Patching Strategies

Three overlapping strategies ensure facts are captured even when the agent session is interrupted mid-compaction:

**1. Behavioral (agent-side):** The agent explicitly calls `clyde-memo add` at key decision points — configuration changes, fixes, deployments. This is the highest-fidelity path.

**2. Cron frequency (compaction_watcher):** Running every 5 minutes catches sessions that were active within the hot window. Even if compaction fires and the full transcript is lost, the last 200KB tail is processed before the next compaction.

**3. Deep digest (conversation_digest):** The 4-hour cycle processes all sessions in the last 7 days. Catches anything the hot-watcher missed and runs topic compaction to consolidate related facts.

---

## License

MIT
