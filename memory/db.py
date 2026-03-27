"""
Direct database connections for ClydeMemory stack.

Replaces `docker exec` subprocess calls with native Python drivers:
  - psycopg2 for PostgreSQL
  - redis-py for Redis

Connection pooling keeps a persistent connection per process.
Falls back to subprocess pipe if native drivers aren't available.
"""

import os
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

from config import PG_HOST, PG_PORT, PG_USER, PG_DB, PG_PASSWORD, REDIS_HOST, REDIS_PORT

# ═══════════════════════════════════════════════════════════════════════════════
# Connection singletons
# ═══════════════════════════════════════════════════════════════════════════════

_pg_conn = None
_redis_conn = None


_PG_PASS_FILE = Path("/root/openclaw-memory/secrets/pg_password.txt")


def _get_pg_password() -> str:
    """Return the PostgreSQL password from config, env var, or shared password file."""
    if PG_PASSWORD:
        return PG_PASSWORD
    if _PG_PASS_FILE.exists():
        return _PG_PASS_FILE.read_text().strip()
    return ""


def get_pg():
    """Get or create a persistent PostgreSQL connection."""
    global _pg_conn
    try:
        import psycopg2
    except ImportError:
        return None

    if _pg_conn is not None:
        try:
            # Check if connection is still alive
            _pg_conn.cursor().execute("SELECT 1")
            return _pg_conn
        except Exception:
            try:
                _pg_conn.close()
            except Exception:
                pass
            _pg_conn = None

    try:
        _pg_conn = psycopg2.connect(
            host=PG_HOST, port=PG_PORT, user=PG_USER,
            dbname=PG_DB, password=_get_pg_password(),
        )
        _pg_conn.autocommit = True
        return _pg_conn
    except Exception:
        return None


def get_redis():
    """Get or create a persistent Redis connection."""
    global _redis_conn
    try:
        import redis
    except ImportError:
        return None

    if _redis_conn is not None:
        try:
            _redis_conn.ping()
            return _redis_conn
        except Exception:
            _redis_conn = None

    try:
        _redis_conn = redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT,
            decode_responses=True, socket_timeout=10,
        )
        _redis_conn.ping()
        return _redis_conn
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Query helpers (drop-in replacements for subprocess versions)
# ═══════════════════════════════════════════════════════════════════════════════

def pg_query(sql: str, params=None) -> str:
    """Execute SQL and return results as pipe-delimited text (matches subprocess pg() output format).

    Args:
        sql: SQL query string. Use %s placeholders for parameters.
        params: Optional tuple/list of parameter values for safe interpolation.
    """
    conn = get_pg()
    if conn is None:
        return None  # Caller should fall back to subprocess

    cur = conn.cursor()
    cur.execute(sql, params)

    if cur.description is None:
        # Non-SELECT (INSERT, UPDATE, DELETE)
        return ""

    rows = cur.fetchall()
    if not rows:
        return ""

    # Format like psql -t -A: pipe-delimited, one row per line
    lines = []
    for row in rows:
        parts = []
        for v in row:
            if v is None:
                parts.append("")
            elif isinstance(v, list):
                # Format Python lists as PG array format: {val1,val2,...}
                parts.append("{" + ",".join(str(x) for x in v) + "}")
            else:
                parts.append(str(v))
        lines.append("|".join(parts))
    return "\n".join(lines)


def pg_execute(sql: str, params=None):
    """Execute SQL without returning results (INSERT, UPDATE, DELETE).

    Args:
        sql: SQL statement. Use %s placeholders for parameters.
        params: Optional tuple/list of parameter values for safe interpolation.
    """
    conn = get_pg()
    if conn is None:
        return None
    cur = conn.cursor()
    cur.execute(sql, params)


def pg_execute_many(sql: str):
    """Execute multiple SQL statements in one call."""
    conn = get_pg()
    if conn is None:
        return None
    cur = conn.cursor()
    # Split on semicolons but handle edge cases
    for stmt in sql.split(";"):
        stmt = stmt.strip()
        if stmt:
            cur.execute(stmt)


def redis_cmd(cmd: str, *args) -> str:
    """Execute a Redis command directly. Returns string result."""
    r = get_redis()
    if r is None:
        return None

    method = getattr(r, cmd.lower(), None)
    if method is None:
        return None
    result = method(*args)
    if isinstance(result, (list, set)):
        return "\n".join(str(x) for x in result)
    return str(result) if result is not None else ""


def redis_pipeline(commands: list) -> int:
    """Execute multiple Redis commands in a pipeline (atomic batch).

    Args:
        commands: list of tuples, e.g. [("SET", "k", "v"), ("SADD", "s", "m1")]

    Returns number of commands executed.
    """
    r = get_redis()
    if r is None:
        return None

    pipe = r.pipeline()
    for cmd in commands:
        op = cmd[0].upper()
        args = cmd[1:]
        # Handle SET with EX/PX options: ("SET", key, val, "EX", 600)
        if op == "SET" and len(args) >= 2:
            kwargs = {}
            key, val = args[0], args[1]
            i = 2
            while i < len(args) - 1:
                flag = str(args[i]).upper()
                if flag == "EX":
                    kwargs["ex"] = int(args[i + 1])
                    i += 2
                elif flag == "PX":
                    kwargs["px"] = int(args[i + 1])
                    i += 2
                else:
                    i += 1
            pipe.set(key, val, **kwargs)
        else:
            method = getattr(pipe, op.lower(), None)
            if method:
                method(*args)
    pipe.execute()
    return len(commands)


def close_all():
    """Close all connections. Call at process exit if needed."""
    global _pg_conn, _redis_conn
    if _pg_conn:
        try:
            _pg_conn.close()
        except Exception:
            pass
        _pg_conn = None
    if _redis_conn:
        try:
            _redis_conn.close()
        except Exception:
            pass
        _redis_conn = None
