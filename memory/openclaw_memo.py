#!/usr/bin/env python3
"""
clyde-memo — Mem0 Integration for ClydeMemory

Persistent memory layer connecting:
  - Qdrant (local Docker) for vector storage
  - OpenAI text-embedding-3-small for embeddings (API)
  - Anthropic Claude for LLM reasoning (API)
  - Optional: vault integration for secure credential access

Usage:
  # As a library
  from openclaw_memo import get_memory
  m = get_memory()
  m.add("the user prefers PowerShell for automation", user_id="user")
  results = m.search("what scripting language does the user prefer", user_id="user")

  # CLI for testing
  python3 openclaw_memo.py add "myapp serves on port 5100" --user user
  python3 openclaw_memo.py search "where does myapp serve" --user user
  python3 openclaw_memo.py list --user user
  python3 openclaw_memo.py status
"""

import argparse
import json
import os
import sys
import threading
from pathlib import Path

# Kill Mem0 telemetry before anything imports it (~0.5s saved)
os.environ.setdefault("MEM0_TELEMETRY", "false")

from config import CLYDE_USER, QDRANT_HOST, QDRANT_PORT

# ── Fix: Anthropic API rejects temperature + top_p together ──
# Mem0's _get_common_params sends both by default. Patch to filter None values.
def _patch_mem0_anthropic():
    """Remove top_p from Mem0's Anthropic LLM params to avoid API rejection."""
    try:
        from mem0.llms.anthropic import AnthropicLLM
        _orig = AnthropicLLM._get_common_params
        def _filtered_common_params(self, **kwargs):
            params = _orig(self, **kwargs)
            params.pop("top_p", None)
            return params
        AnthropicLLM._get_common_params = _filtered_common_params
    except (ImportError, AttributeError):
        pass

_patch_mem0_anthropic()

# ===============================================================================
# Vault integration -- optional, falls back to env vars
# ===============================================================================

VAULT_ENV = Path("/etc/clyde/vault.env")
VAULT_FILE = Path("/etc/clyde/vault.enc")

# ClawVault API (primary key source)
CLAWVAULT_ENV = Path("/etc/openclaw/vault.env")
CLAWVAULT_URL = "http://127.0.0.1:7777"

# Cache fetched keys for the process lifetime (avoid repeated HTTP calls)
_key_cache = {}


def _get_clawvault_token() -> str:
    """Read machine token from ClawVault env file."""
    if not CLAWVAULT_ENV.exists():
        return ""
    for line in CLAWVAULT_ENV.read_text().splitlines():
        line = line.strip()
        if line.startswith("VAULT_MACHINE_TOKEN="):
            return line.split("=", 1)[1].strip()
    return ""


def _fetch_from_clawvault(key_name: str) -> str:
    """Fetch a key from ClawVault API. Returns empty string on failure."""
    if key_name in _key_cache:
        return _key_cache[key_name]

    token = _get_clawvault_token()
    if not token:
        return ""

    try:
        import urllib.request
        req = urllib.request.Request(
            f"{CLAWVAULT_URL}/api/keys/{key_name}",
            headers={"Authorization": f"Bearer {token}"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            val = data.get("value", "")
            if val:
                _key_cache[key_name] = val
            return val
    except Exception:
        return ""


def load_vault_env():
    """Load master key and other vars from vault.env (optional)."""
    if not VAULT_ENV.exists():
        return None  # Vault not configured — use env vars directly
    env = {}
    for line in VAULT_ENV.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            env[key.strip()] = val.strip()
    return env


def get_vault_key(key_name: str) -> str:
    """
    Retrieve a key from ClawVault API, legacy vault, or environment variables.

    Priority:
      1. ClawVault API (http://127.0.0.1:7777) via machine token
      2. Legacy vault (/etc/clyde/vault.env + vault.enc)
      3. Environment variables
    """
    # 1. Try ClawVault API first
    val = _fetch_from_clawvault(key_name)
    if val:
        return val

    # 2. Try legacy vault
    try:
        vault_env = load_vault_env()
        if vault_env is not None:
            from cryptography.fernet import Fernet
            master = vault_env.get("VAULT_MASTER_KEY", "")
            if master and VAULT_FILE.exists():
                fernet = Fernet(master.encode())
                vault_data = json.loads(fernet.decrypt(VAULT_FILE.read_bytes()))
                if key_name in vault_data:
                    return vault_data[key_name]
    except Exception:
        pass  # Vault failed, fall through to env vars

    # 3. Fall back to environment variables
    val = os.environ.get(key_name, "")
    if val:
        return val

    raise RuntimeError(
        f"{key_name} not found in ClawVault, legacy vault, or environment. "
        f"Check ClawVault at {CLAWVAULT_URL} or set {key_name} env var."
    )

# ===============================================================================
# Mem0 configuration and initialization
# ===============================================================================

def get_memo_config() -> dict:
    """Build Mem0 config using vault or environment credentials."""
    openai_key = get_vault_key("OPENAI_API_KEY")
    anthropic_key = get_vault_key("ANTHROPIC_API_KEY")

    return {
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "host": QDRANT_HOST,
                "port": QDRANT_PORT,
                "collection_name": os.environ.get("QDRANT_COLLECTION", "openclaw_memories"),
            },
        },
        "embedder": {
            "provider": "openai",
            "config": {
                "api_key": openai_key,
                "model": "text-embedding-3-small",
            },
        },
        "llm": {
            "provider": "anthropic",
            "config": {
                "api_key": anthropic_key,
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 2000,
            },
        },
        "version": "v1.1",
    }

_memory_instance = None

def get_memory():
    """Initialize and return a Mem0 Memory instance (singleton)."""
    global _memory_instance
    if _memory_instance is None:
        from mem0 import Memory
        config = get_memo_config()
        _memory_instance = Memory.from_config(config)
    return _memory_instance

# ===============================================================================
# Daemon client
# ===============================================================================

DAEMON_SOCK = os.environ.get("CLYDE_SOCK_PATH", "/tmp/clyde-memo.sock")

def _daemon_request(method, params):
    """Try daemon socket. Returns response dict or None if unavailable."""
    import socket as _sock
    try:
        s = _sock.socket(_sock.AF_UNIX, _sock.SOCK_STREAM)
        s.settimeout(90)
        s.connect(DAEMON_SOCK)
        s.sendall(json.dumps({"method": method, "params": params}).encode() + b"\n")
        data = b""
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
        s.close()
        resp = json.loads(data)
        return resp if resp.get("ok") else None
    except Exception:
        return None

# ===============================================================================
# CLI commands
# ===============================================================================

GREEN  = "\033[92m"
CYAN   = "\033[96m"
YELLOW = "\033[93m"
RED    = "\033[91m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
RESET  = "\033[0m"

IMPACT_LEVELS = {"low": 0.25, "normal": 0.5, "high": 0.75, "critical": 1.0}

# Patterns that auto-detect high-impact memories
IMPACT_PATTERNS = {
    "critical": [
        r"(?i)\b(credential|secret|api.?key|password)\b.*\b(changed|rotated|compromised|leaked)\b",
        r"(?i)\b(data.?loss|outage|security.?breach|incident)\b",
    ],
    "high": [
        r"(?i)\b(deployed|shipped|released|launched)\b",
        r"(?i)\b(fixed|resolved|debugged|root.?cause)\b.*\b(bug|issue|error|crash|failure)\b",
        r"(?i)\b(bug|issue|error|crash|failure)\b.*\b(fixed|resolved|debugged|root.?cause)\b",
        r"(?i)\b(architecture|design.?decision|chose|decided|switched.?to)\b",
        r"(?i)\b(wrong|incorrect|stale|outdated)\b.*\b(path|config|file|setting)\b",
        r"(?i)\b(path|config|file|setting)\b.*\b(wrong|incorrect|stale|outdated|should.?be)\b",
        r"(?i)\b(migration|schema.?change|breaking.?change)\b",
        r"(?i)\b(container|docker)\b.*\b(build|rebuild|image)\b.*\b(from|source|path)\b",
        r"(?i)\b(mount|volume|bind)\b.*\b(correct|actual|real)\b",
    ],
}

def _detect_impact(text: str) -> str:
    """Auto-detect impact level from content patterns."""
    import re
    for level in ("critical", "high"):
        for pattern in IMPACT_PATTERNS[level]:
            if re.search(pattern, text):
                return level
    return "normal"

def _set_impact_pg(memory_id: str, impact: str):
    """Set impact_category on memories table and impact_score on keyscores via PostgreSQL.
    First syncs the memory from Qdrant->PG if it doesn't exist yet."""
    from db import pg_execute
    score = IMPACT_LEVELS.get(impact, 0.5)
    try:
        # Upsert memory record
        pg_execute(
            "INSERT INTO memories (qdrant_point_id, collection, content_hash, summary, source, confidence, impact_category) "
            "VALUES (%s, 'clyde_memories', md5(%s), '', 'explicit', 'direct', %s) "
            "ON CONFLICT (qdrant_point_id) DO UPDATE SET impact_category = %s",
            (memory_id, memory_id, impact, impact),
        )
        # Upsert keyscore with composite recalc
        pg_execute(
            "INSERT INTO keyscores (memory_id, impact_score, composite_score) "
            "SELECT id, %s, %s FROM memories WHERE qdrant_point_id = %s "
            "ON CONFLICT (memory_id) DO UPDATE SET "
            "impact_score = %s, "
            "composite_score = compute_composite_score_v2("
            "  keyscores.recency_score, keyscores.frequency_score, keyscores.authority_score, "
            "  keyscores.entity_boost, %s)",
            (score, score, memory_id, score, score),
        )
    except Exception:
        pass


def _track_access(qdrant_point_ids: list):
    """Increment access_count and update last_accessed for searched memories."""
    if not qdrant_point_ids:
        return
    from db import pg_execute
    try:
        # Use ANY(%s) with a list parameter instead of building an IN clause
        pg_execute(
            "UPDATE memories SET access_count = access_count + 1, last_accessed = NOW() "
            "WHERE qdrant_point_id = ANY(%s)",
            (qdrant_point_ids,),
        )
    except Exception:
        pass


def _display_boosted(boosted_list):
    """Display boosted search results."""
    for i, item in enumerate(boosted_list, 1):
        if isinstance(item, dict):
            final = item.get("final", 0)
            boost = item.get("boost", 0)
            lexical = item.get("lexical", 0)
            memory = item.get("memory", "")
            mem_id = item.get("mem_id", "")
        else:
            final, _orig, boost, memory, mem_id = item
            lexical = 0
        tags = []
        if boost > 0:
            tags.append(f"{GREEN}+{boost:.2f}\u2191{RESET}")
        if lexical > 0:
            tags.append(f"{CYAN}fts:{lexical:.2f}{RESET}")
        tag_str = f" {' '.join(tags)}" if tags else ""
        print(f"  {CYAN}{i}.{RESET} [{final:.3f}] {memory} {DIM}({mem_id}){RESET}{tag_str}")


def cmd_add(args):
    """Add a memory with optional impact scoring."""
    impact = getattr(args, "impact", None)
    if not impact or impact == "auto":
        impact = _detect_impact(args.text)

    # Try daemon first
    dresp = _daemon_request("add", {
        "text": args.text, "user_id": args.user, "impact": impact
    })
    if dresp is not None:
        result = dresp.get("result", {})
    else:
        m = get_memory()
        result = m.add(args.text, user_id=args.user)

    impact_icon = {"critical": "\U0001f534", "high": "\U0001f7e0", "normal": "\U0001f7e2", "low": "\u26aa"}.get(impact, "\U0001f7e2")

    if result and result.get("results"):
        for r in result["results"]:
            event = r.get("event", "unknown")
            memory = r.get("memory", "")
            mem_id = r.get("id", "")
            print(f"  {GREEN}\u2713{RESET} [{event.upper()}] {memory}")
            # Set impact in PostgreSQL (daemon does this server-side, but handle fallback)
            if dresp is None and mem_id and impact != "normal":
                _set_impact_pg(mem_id, impact)
            if impact != "normal":
                print(f"  {impact_icon} Impact: {impact} (score: {IMPACT_LEVELS[impact]})")
    else:
        print(f"  {GREEN}\u2713{RESET} Memory added {impact_icon} [{impact}]")
    return 0


def cmd_search(args):
    """Search memories with entity-boosted scoring."""
    # Try daemon first -- returns pre-boosted results
    search_params = {
        "query": args.query, "user_id": args.user, "limit": args.limit
    }
    # Pass session context from env vars (set by clydecodebot) for ML retrieval tagging
    sid = os.environ.get("OPENCLAW_SESSION_ID")
    turn = os.environ.get("OPENCLAW_TURN_NUMBER")
    if sid:
        search_params["session_id"] = sid
    if turn:
        try:
            search_params["turn_number"] = int(turn)
        except ValueError:
            pass
    dresp = _daemon_request("search", search_params)
    if dresp is not None:
        boosted = dresp.get("results", [])
        if not boosted:
            print(f"  {YELLOW}No memories found{RESET}")
            return 0
        _display_boosted(boosted)
        return 0

    # Fallback: local Memory with entity boost
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from entity_boost import get_boost
    m = get_memory()
    results = m.search(args.query, user_id=args.user, limit=args.limit)
    if not results.get("results"):
        print(f"  {YELLOW}No memories found{RESET}")
        return 0
    boosted = []
    result_ids = []
    for r in results["results"]:
        memory = r.get("memory", "")
        score = r.get("score", 0)
        mem_id = r.get("id", "")
        short_id = mem_id[:8]
        boost = get_boost(args.query, short_id)
        boosted.append((score + boost, score, boost, memory, short_id))
        if mem_id:
            result_ids.append(mem_id)
    boosted.sort(key=lambda x: x[0], reverse=True)
    _display_boosted(boosted)

    if result_ids:
        _track_access(result_ids)
    return 0


def cmd_format(args):
    """Search + format as structured context block for prompt injection."""
    search_params = {
        "query": args.query, "user_id": args.user, "limit": args.limit,
        "skip_gate": True,  # Formatter does its own filtering
    }
    sid = os.environ.get("OPENCLAW_SESSION_ID")
    turn = os.environ.get("OPENCLAW_TURN_NUMBER")
    if sid:
        search_params["session_id"] = sid
    if turn:
        try:
            search_params["turn_number"] = int(turn)
        except ValueError:
            pass
    dresp = _daemon_request("search", search_params)
    if dresp is None or not dresp.get("ok"):
        return 0

    try:
        from prompt_formatter import PackagedContext
        pkg = PackagedContext.from_search_response(dresp, query=args.query)
        block = pkg.to_prompt_block()
        if block:
            print(block)
        # Log which bucket each memory was packaged into (for context_decay)
        try:
            from context_decay import log_packaged_buckets
            log_packaged_buckets(dresp.get("results", []), pkg.classified)
        except Exception:
            pass  # non-fatal
    except Exception as e:
        # Fallback: just show flat results
        boosted = dresp.get("results", [])
        if boosted:
            _display_boosted(boosted)
    return 0


def cmd_list(args):
    """List all memories for a user."""
    # Try daemon first
    dresp = _daemon_request("get_all", {"user_id": args.user})
    if dresp is not None:
        memories = dresp.get("results", [])
    else:
        m = get_memory()
        results = m.get_all(user_id=args.user)
        memories = results.get("results", [])

    if not memories:
        print(f"  {YELLOW}No memories stored for user: {args.user}{RESET}")
        return 0

    print(f"  {BOLD}Memories for {args.user}:{RESET}\n")
    for i, r in enumerate(memories, 1):
        memory = r.get("memory", "")
        mem_id = r.get("id", "")[:8]
        created = r.get("created_at", "")[:19]
        print(f"  {CYAN}{i:3d}.{RESET} {memory}")
        print(f"       {DIM}id: {mem_id}  created: {created}{RESET}")

    print(f"\n  {BOLD}Total: {len(memories)} memories{RESET}")
    return 0


def cmd_delete(args):
    """Delete a specific memory by ID."""
    dresp = _daemon_request("delete", {"memory_id": args.memory_id})
    if dresp is None:
        m = get_memory()
        m.delete(args.memory_id)
    print(f"  {GREEN}\u2713{RESET} Memory deleted: {args.memory_id}")
    return 0


def cmd_reset(args):
    """Delete all memories for a user."""
    if not args.yes:
        try:
            confirm = input(f"  {RED}Delete ALL memories for user '{args.user}'? [y/N]{RESET} ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\n  Aborted.")
            return 1
        if confirm != "y":
            print("  Aborted.")
            return 1

    dresp = _daemon_request("delete_all", {"user_id": args.user})
    if dresp is None:
        m = get_memory()
        m.delete_all(user_id=args.user)
    print(f"  {GREEN}\u2713{RESET} All memories deleted for user: {args.user}")
    return 0


def cmd_status(args):
    """Check Mem0 system status."""
    print(f"\n{CYAN}{BOLD}{'=' * 50}")
    print(f"  ClydeMemory -- Status")
    print(f"{'=' * 50}{RESET}\n")

    # Check daemon
    dresp = _daemon_request("status", {})
    if dresp is not None:
        uptime = dresp.get("uptime_s", 0)
        print(f"  {GREEN}\u2713{RESET} Daemon running (uptime: {uptime:.0f}s)")
    else:
        print(f"  {YELLOW}-{RESET} Daemon not running (using fallback)")

    # Check API keys
    for key_name in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        try:
            val = get_vault_key(key_name)
            masked = val[:8] + "..." + val[-4:]
            print(f"  {GREEN}\u2713{RESET} {key_name}: {masked}")
        except Exception as e:
            print(f"  {RED}\u2717{RESET} {key_name}: {e}")

    # Check Qdrant
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(QDRANT_HOST, port=QDRANT_PORT)
        collections = client.get_collections().collections
        coll_names = [c.name for c in collections]
        print(f"  {GREEN}\u2713{RESET} Qdrant: {len(collections)} collection(s) {coll_names}")
    except Exception as e:
        print(f"  {RED}\u2717{RESET} Qdrant: {e}")

    # Check Mem0 initialization (skip if daemon is running -- already proven)
    if dresp is None:
        try:
            m = get_memory()
            print(f"  {GREEN}\u2713{RESET} Mem0 initialized")
        except Exception as e:
            print(f"  {RED}\u2717{RESET} Mem0: {e}")
    else:
        print(f"  {GREEN}\u2713{RESET} Mem0 initialized (via daemon)")

    print()
    return 0


def cmd_pin(args):
    """Pin or unpin a memory (pinned memories never decay)."""
    mem_id = args.memory_id
    unpin = getattr(args, "unpin", False)
    pinned_val = "FALSE" if unpin else "TRUE"
    action = "Unpinned" if unpin else "Pinned"

    # Update memories.pinned in PG
    try:
        from db import get_pg
        conn = get_pg()
        if conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE memories SET pinned = %s WHERE qdrant_point_id LIKE %s",
                (not unpin, mem_id + "%")
            )
            if cur.rowcount == 0:
                print(f"  {RED}✗{RESET} No memory found matching ID: {mem_id}")
                return 1
            # If pinning, also set recency_score = 1.0
            if not unpin:
                cur.execute("""
                    UPDATE keyscores k SET recency_score = 1.0, computed_at = NOW()
                    FROM memories m WHERE k.memory_id = m.id
                    AND m.qdrant_point_id LIKE %s
                """, (mem_id + "%",))
            print(f"  {GREEN}✓{RESET} {action}: {mem_id}")
            return 0
    except Exception:
        pass

    # Fallback: use db module directly
    import db as _db
    result = _db.pg_query(
        "UPDATE memories SET pinned = %s WHERE qdrant_point_id LIKE %s RETURNING id",
        (pinned_val, mem_id + "%")
    )
    if not result:
        print(f"  {RED}✗{RESET} No memory found matching ID: {mem_id}")
        return 1
    if not unpin:
        _db.pg_execute(
            "UPDATE keyscores k SET recency_score = 1.0, computed_at = NOW() "
            "FROM memories m WHERE k.memory_id = m.id "
            "AND m.qdrant_point_id LIKE %s",
            (mem_id + "%",)
        )
    print(f"  {GREEN}✓{RESET} {action}: {mem_id}")
    return 0


def cmd_search_debug(args):
    """Search with full salience breakdown per result."""
    import time as _time
    t0 = _time.time()

    # Try daemon first (skip gate so we see all results)
    debug_params = {
        "query": args.query, "user_id": args.user,
        "limit": args.limit, "skip_gate": True
    }
    sid = os.environ.get("OPENCLAW_SESSION_ID")
    turn = os.environ.get("OPENCLAW_TURN_NUMBER")
    if sid:
        debug_params["session_id"] = sid
    if turn:
        try:
            debug_params["turn_number"] = int(turn)
        except ValueError:
            pass
    dresp = _daemon_request("search", debug_params)

    if dresp is not None:
        results = dresp.get("results", [])
        gate_score = dresp.get("gate_score", "?")
        gate_stats = dresp.get("gate_stats", {})
        complexity = dresp.get("query_complexity", "?")
        ca_context = dresp.get("ca_context", {})
    else:
        # Fallback: direct salience engine
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            from salience_engine import salience_score
            m = get_memory()
            raw = m.search(args.query, user_id=args.user, limit=max(args.limit * 2, 10))
            results = salience_score(args.query, raw.get("results", []), limit=args.limit)
        except Exception:
            m = get_memory()
            raw = m.search(args.query, user_id=args.user, limit=args.limit)
            results = []
            for r in raw.get("results", [])[:args.limit]:
                results.append({
                    "final": r.get("score", 0), "score": r.get("score", 0),
                    "memory": r.get("memory", ""), "mem_id": r.get("id", "")[:8],
                    "id": r.get("id", ""), "breakdown": {},
                })
        gate_score = "N/A (no daemon)"
        gate_stats = {}
        complexity = "?"
        ca_context = {}

    elapsed = (_time.time() - t0) * 1000

    if not results:
        print(f"  {YELLOW}No results{RESET}")
        return 0

    # ── Header ──
    print(f"\n  {BOLD}Debug: '{args.query}'{RESET}")
    print(f"  {DIM}limit={args.limit}  elapsed={elapsed:.0f}ms  "
          f"gate={gate_score}  complexity={complexity}{RESET}")
    if gate_stats:
        print(f"  {DIM}proposals={gate_stats.get('proposed', '?')}  "
              f"admitted={gate_stats.get('admitted', '?')}  "
              f"tokens={gate_stats.get('tokens_used', '?')}{RESET}")
    print()

    # ── Column header ──
    print(f"  {'#':>2}  {'Final':>6}  {'Sem':>5}  {'Rec':>5}  {'Goal':>5}  "
          f"{'OQ':>4}  {'Narr':>5}  {'WM':>4}  {'Freq':>5}  {'Ent':>4}  Memory")
    print(f"  {'-'*110}")

    # ── Results with full breakdown ──
    for i, r in enumerate(results, 1):
        bd = r.get("breakdown", {})
        mem_preview = r.get("memory", "")[:42]
        mem_id = r.get("mem_id", r.get("id", "?")[:8])

        # Use breakdown values if available, fall back to top-level keys
        sem   = bd.get("semantic", r.get("score", 0))
        rec   = bd.get("recency", 0)
        goal  = bd.get("goal_prox", 0)
        oq    = bd.get("oq_boost", 0)
        narr  = bd.get("narrative", 0)
        wm    = bd.get("working_mode", 0)
        freq  = bd.get("frequency", 0)
        ent   = bd.get("entity_boost", r.get("boost", 0))

        print(
            f"  {i:2d}  {r.get('final', 0):6.4f}  "
            f"{sem:.3f}  {rec:.3f}  {goal:.3f}  "
            f"{oq:.2f}  {narr:.3f}  {wm:.2f}  "
            f"{freq:.3f}  {ent:.2f}  "
            f"{mem_preview}  {DIM}({mem_id}){RESET}"
        )

    # ── CA context summary ──
    if ca_context:
        print(f"\n  {BOLD}CA Context injected:{RESET}")
        for key, val in ca_context.items():
            preview = val[:80].replace("\n", " ") if isinstance(val, str) else str(val)[:80]
            print(f"    {CYAN}{key}{RESET}: {preview}...")

    # ── Weights reference ──
    print(f"\n  {DIM}Weights: sem=0.35  rec=0.15  goal=0.15  oq=0.10  "
          f"narr=0.10  wm=0.10  freq=0.05  (ent=additive){RESET}\n")
    return 0


def cmd_ingest(args):
    """Ingest a markdown file as memories (one fact per line/section)."""
    filepath = Path(args.file)

    if not filepath.exists():
        print(f"  {RED}\u2717{RESET} File not found: {filepath}")
        return 1

    content = filepath.read_text().strip()
    if not content:
        print(f"  {YELLOW}File is empty{RESET}")
        return 0

    # Split into meaningful chunks -- by headers or double newlines
    chunks = []
    current_chunk = []

    for line in content.splitlines():
        line = line.strip()
        if not line:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            continue
        if line.startswith("#"):
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            continue
        if line.startswith("- "):
            line = line[2:]
        if line.startswith("* "):
            line = line[2:]
        current_chunk.append(line)

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    chunks = [c for c in chunks if len(c) > 10]

    print(f"  {CYAN}\u2192{RESET} Ingesting {len(chunks)} chunks from {filepath.name}\n")

    added = 0
    for i, chunk in enumerate(chunks, 1):
        try:
            # Try daemon first for each chunk
            dresp = _daemon_request("add", {"text": chunk, "user_id": args.user, "impact": "auto"})
            if dresp is not None:
                result = dresp.get("result", {})
            else:
                if not hasattr(cmd_ingest, '_m'):
                    cmd_ingest._m = get_memory()
                result = cmd_ingest._m.add(chunk, user_id=args.user)

            events = result.get("results", []) if result else []
            event_types = [r.get("event", "?") for r in events]
            print(f"  {GREEN}\u2713{RESET} [{i}/{len(chunks)}] {chunk[:80]}{'...' if len(chunk) > 80 else ''}")
            if event_types:
                print(f"    {DIM}events: {', '.join(event_types)}{RESET}")
            added += 1
        except Exception as e:
            print(f"  {RED}\u2717{RESET} [{i}/{len(chunks)}] Failed: {e}")

    print(f"\n  {BOLD}Ingested {added}/{len(chunks)} chunks{RESET}")
    return 0

# ===============================================================================
# CLI entry point
# ===============================================================================

def main():
    # Only preload mem0 if daemon isn't available (avoids thread abort at exit)
    if not os.path.exists(DAEMON_SOCK):
        threading.Thread(target=lambda: __import__('mem0'), daemon=True).start()

    parser = argparse.ArgumentParser(
        prog="clyde-memo",
        description="ClydeMemory Mem0 Integration -- Persistent Memory Layer",
    )
    sub = parser.add_subparsers(dest="command")

    # add
    p_add = sub.add_parser("add", help="Add a memory")
    p_add.add_argument("text", help="Memory text to add")
    p_add.add_argument("--user", default=CLYDE_USER, help=f"User ID (default: {CLYDE_USER})")
    p_add.add_argument("--impact", choices=["low", "normal", "high", "critical", "auto"],
                       default="auto", help="Impact level (default: auto-detect)")

    # search
    p_search = sub.add_parser("search", help="Search memories")
    p_search.add_argument("query", help="Search query")
    p_search.add_argument("--user", default=CLYDE_USER, help=f"User ID (default: {CLYDE_USER})")
    p_search.add_argument("--limit", type=int, default=5, help="Max results")
    p_search.add_argument("--debug", action="store_true", help="Show raw vs re-ranked scores")

    # list
    p_list = sub.add_parser("list", help="List all memories")
    p_list.add_argument("--user", default=CLYDE_USER, help=f"User ID (default: {CLYDE_USER})")

    # delete
    p_del = sub.add_parser("delete", help="Delete a memory by ID")
    p_del.add_argument("memory_id", help="Memory ID to delete")

    # reset
    p_reset = sub.add_parser("reset", help="Delete all memories for a user")
    p_reset.add_argument("--user", default=CLYDE_USER, help=f"User ID (default: {CLYDE_USER})")
    p_reset.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # status
    sub.add_parser("status", help="Check system status")

    # pin
    p_pin = sub.add_parser("pin", help="Pin a memory (immune to recency decay)")
    p_pin.add_argument("memory_id", help="Memory ID (or prefix) to pin")
    p_pin.add_argument("--unpin", action="store_true", help="Unpin instead of pin")

    # format — structured context block for prompt injection
    p_format = sub.add_parser("format", help="Search + format as structured context block")
    p_format.add_argument("query", help="Search query")
    p_format.add_argument("--user", default=CLYDE_USER, help=f"User ID (default: {CLYDE_USER})")
    p_format.add_argument("--limit", type=int, default=10, help="Max results to classify")

    # ingest
    p_ingest = sub.add_parser("ingest", help="Ingest a markdown file as memories")
    p_ingest.add_argument("file", help="Path to markdown file")
    p_ingest.add_argument("--user", default=CLYDE_USER, help=f"User ID (default: {CLYDE_USER})")

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return 1

    cmds = {
        "add": cmd_add,
        "search": lambda a: cmd_search_debug(a) if getattr(a, "debug", False) else cmd_search(a),
        "format": cmd_format,
        "list": cmd_list,
        "delete": cmd_delete,
        "reset": cmd_reset,
        "status": cmd_status,
        "pin": cmd_pin,
        "ingest": cmd_ingest,
    }
    return cmds[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
