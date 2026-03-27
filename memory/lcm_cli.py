#!/usr/bin/env python3
"""
openclaw-lcm — CLI for LCM agent-facing history tools.

Thin wrapper that sends requests to the memo daemon socket.

Usage:
  openclaw-lcm grep "query" [--session ID] [--limit N]
  openclaw-lcm describe --session ID [--turns START-END]
  openclaw-lcm recall "query" [--session ID] [--budget N]
  openclaw-lcm pressure --session ID [--compact]
  openclaw-lcm stats
"""

import json
import os
import socket
import sys

SOCK_PATH = os.environ.get("CLYDE_SOCK_PATH", "/tmp/clyde-memo.sock")


def daemon_request(method: str, params: dict) -> dict:
    """Send a request to the memo daemon and return the response."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(30)
        s.connect(SOCK_PATH)
        s.sendall(json.dumps({"method": method, "params": params}).encode() + b"\n")

        # Read until newline (daemon sends JSON + \n)
        buf = bytearray()
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf.extend(chunk)
            if b"\n" in buf:
                break
        s.close()

        line = buf.split(b"\n", 1)[0]
        return json.loads(line.decode())
    except FileNotFoundError:
        return {"ok": False, "error": f"Daemon socket not found at {SOCK_PATH}. Start with: memo_daemon.py start"}
    except ConnectionRefusedError:
        return {"ok": False, "error": "Daemon not running. Start with: memo_daemon.py start"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def cmd_grep(args):
    if not args:
        print("Usage: openclaw-lcm grep \"query\" [--session ID] [--limit N]")
        sys.exit(1)

    query = args[0]
    params = {"query": query, "limit": 10}

    if "--session" in args:
        idx = args.index("--session")
        if idx + 1 < len(args):
            params["session_id"] = args[idx + 1]
    if "--limit" in args:
        idx = args.index("--limit")
        if idx + 1 < len(args):
            params["limit"] = int(args[idx + 1])

    resp = daemon_request("lcm_grep", params)
    if not resp.get("ok"):
        print(f"Error: {resp.get('error')}")
        sys.exit(1)

    results = resp.get("results", [])
    if not results:
        print("No results.")
        return

    print(f"Found {len(results)} result(s):\n")
    for r in results:
        role = r.get("role", "?")
        sid = r.get("session_id", "?")[:12]
        turn = r.get("turn_index", "?")
        rank = r.get("rank", 0)
        content = r.get("content", "")[:150]
        print(f"  [{sid}] turn {turn} ({role}) rank={rank:.3f}")
        print(f"    {content}")
        print()


def cmd_describe(args):
    params = {}
    if "--session" in args:
        idx = args.index("--session")
        if idx + 1 < len(args):
            params["session_id"] = args[idx + 1]
    elif args and not args[0].startswith("--"):
        params["session_id"] = args[0]

    if not params.get("session_id"):
        print("Usage: openclaw-lcm describe --session ID [--turns START-END]")
        sys.exit(1)

    if "--turns" in args:
        idx = args.index("--turns")
        if idx + 1 < len(args):
            parts = args[idx + 1].split("-")
            params["start_turn"] = int(parts[0])
            params["end_turn"] = int(parts[1]) if len(parts) > 1 else int(parts[0])

    resp = daemon_request("lcm_describe", params)
    if not resp.get("ok"):
        print(f"Error: {resp.get('error')}")
        sys.exit(1)

    if resp.get("summaries"):
        for s in resp["summaries"]:
            print(f"[{s['node_id']}] depth={s['depth']} turns={s['turn_range']}")
            print(f"  {s['summary']}")
            print()
    elif resp.get("raw_context"):
        print(resp["raw_context"][:3000])

    if resp.get("large_files"):
        print("\nLarge files in range:")
        for lf in resp["large_files"]:
            print(f"  [{lf['file_hint']}] turn={lf['turn_index']}")
            print(f"    {lf['summary'][:200]}")


def cmd_recall(args):
    if not args:
        print("Usage: openclaw-lcm recall \"query\" [--session ID] [--budget N]")
        sys.exit(1)

    query = args[0]
    params = {"query": query, "token_budget": 4000}

    if "--session" in args:
        idx = args.index("--session")
        if idx + 1 < len(args):
            params["session_id"] = args[idx + 1]
    if "--budget" in args:
        idx = args.index("--budget")
        if idx + 1 < len(args):
            params["token_budget"] = int(args[idx + 1])

    resp = daemon_request("lcm_recall", params)
    if not resp.get("ok"):
        print(f"Error: {resp.get('error')}")
        sys.exit(1)

    text = resp.get("text", "")
    if text:
        print(text)
        print(f"\n--- {resp.get('tokens', 0)} tokens from {resp.get('source', '?')} ---")
    else:
        print("No relevant history found.")


def cmd_pressure(args):
    params = {}
    if "--session" in args:
        idx = args.index("--session")
        if idx + 1 < len(args):
            params["session_id"] = args[idx + 1]
    elif args and not args[0].startswith("--"):
        params["session_id"] = args[0]

    if not params.get("session_id"):
        print("Usage: openclaw-lcm pressure --session ID [--compact]")
        sys.exit(1)

    params["auto_compact"] = "--compact" in args

    resp = daemon_request("lcm_pressure", params)
    if not resp.get("ok"):
        print(f"Error: {resp.get('error')}")
        sys.exit(1)

    if resp.get("compacted"):
        print(f"Compacted: {resp['leaves_created']} leaves, {resp['rollups_created']} rollups")
        print(f"Pressure: {resp['pressure_before']:.1%} → {resp['pressure_after']:.1%}")
    else:
        ratio = resp.get("ratio", 0)
        tokens = resp.get("tokens", 0)
        limit = resp.get("limit", 0)
        print(f"Pressure: {ratio:.1%} ({tokens:,} / {limit:,} tokens)")
        print(f"Needs compaction: {resp.get('needs_compaction', False)}")


def cmd_stats():
    resp = daemon_request("lcm_stats", {})
    if not resp.get("ok"):
        print(f"Error: {resp.get('error')}")
        sys.exit(1)

    print(f"\n  LCM Statistics")
    print(f"  {'=' * 40}")
    for key in ("active_sessions", "total_messages", "total_tokens",
                "summary_nodes", "leaf_summaries", "rollup_summaries", "max_depth"):
        val = resp.get(key, 0)
        label = key.replace("_", " ").title()
        if "token" in key:
            print(f"  {label + ':':<25} {val:,}")
        else:
            print(f"  {label + ':':<25} {val}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "grep":
        cmd_grep(args)
    elif cmd == "describe":
        cmd_describe(args)
    elif cmd == "recall":
        cmd_recall(args)
    elif cmd == "pressure":
        cmd_pressure(args)
    elif cmd == "stats":
        cmd_stats()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
