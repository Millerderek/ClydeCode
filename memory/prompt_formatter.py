#!/usr/bin/env python3
"""
prompt_formatter.py — Structured Context Packaging for LLM Prompts.

Converts raw retrieval results (memo search + ca_context + constraints)
into a structured prompt block ordered for model attention:

    ENTITIES → CONSTRAINTS → PROCEDURES → FACTS → HISTORICAL → UNCERTAINTIES

Every memory gets classified into a bucket, scored for confidence,
and formatted as terse bullets. The model gets structured, actionable
context instead of a flat dump.

Usage:
    from prompt_formatter import PackagedContext

    pkg = PackagedContext.from_search_response(daemon_response)
    prompt_block = pkg.to_prompt_block()
    # → "[RECOVERED_CONTEXT]\n\nENTITIES\n- Docker\n..."

CLI:
    python3 prompt_formatter.py format "configure the bridge deployment"
    python3 prompt_formatter.py classify "Docker container is set to restart=always"
"""

import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ═══════════════════════════════════════════════════════════════════════════════
# Bucket definitions
# ═══════════════════════════════════════════════════════════════════════════════

BUCKETS = ["CONSTRAINTS", "PROCEDURES", "FACTS", "HISTORICAL", "UNCERTAINTIES"]

# ═══════════════════════════════════════════════════════════════════════════════
# Bucket classifier — pattern-based with confidence scoring
# ═══════════════════════════════════════════════════════════════════════════════

# Constraint patterns: decisions, requirements, hard rules
_CONSTRAINT_PATTERNS = [
    (re.compile(r'\b(must|required|requires?|mandatory|critical|necessary)\b', re.I), 0.90),
    (re.compile(r'\b(always|never|do not|don\'t|cannot|must not)\b', re.I), 0.85),
    (re.compile(r'\b(chose|chosen|decided|selected|switched to|migrated to)\b', re.I), 0.88),
    (re.compile(r'\b(constraint|restriction|limitation|requirement)\b', re.I), 0.92),
    (re.compile(r'\b(set to|configured? (?:to|as|for)|bound to|pinned to)\b', re.I), 0.80),
    (re.compile(r'\b(IP (?:is|address)|port \d+|uses? port)\b', re.I), 0.82),
    (re.compile(r'\b(depends? on|dependency|prerequisite)\b', re.I), 0.78),
]

# Procedure patterns: actionable steps, commands, instructions
_PROCEDURE_PATTERNS = [
    (re.compile(r'\b(run|execute|restart|deploy|install|start|stop|kill)\b', re.I), 0.80),
    (re.compile(r'`[^`]+`', re.I), 0.75),  # Inline code
    (re.compile(r'\b(step \d|first|then|next|finally|after that)\b', re.I), 0.82),
    (re.compile(r'\b(docker[- ]compose|systemctl|pip install|npm|curl|scp|ssh)\b', re.I), 0.85),
    (re.compile(r'\b(to fix|to resolve|to clear|to update|to build|to push)\b', re.I), 0.78),
    (re.compile(r'\b(OTA|firmware|update|upgrade|rebuild)\b', re.I), 0.72),
]

# Historical patterns: past state, legacy, old versions
_HISTORICAL_PATTERNS = [
    (re.compile(r'\b(originally|previously|used to|was formerly|legacy|deprecated)\b', re.I), 0.80),
    (re.compile(r'\b(old|former|prior|replaced by|superseded|obsolete)\b', re.I), 0.75),
    (re.compile(r'\b(v1|version 1|initial|first version|prototype)\b', re.I), 0.72),
    (re.compile(r'\b(removed|dropped|no longer|discontinued)\b', re.I), 0.78),
    (re.compile(r'\b(migrated from|moved from|switched from)\b', re.I), 0.76),
]

# Uncertainty patterns: unknowns, unconfirmed, maybes
_UNCERTAINTY_PATTERNS = [
    (re.compile(r'\b(unclear|unknown|unconfirmed|not confirmed|uncertain)\b', re.I), 0.90),
    (re.compile(r'\b(may|might|possibly|perhaps|could be|not sure)\b', re.I), 0.75),
    (re.compile(r'\b(TODO|TBD|pending|needs? investigation|needs? verification)\b', re.I), 0.85),
    (re.compile(r'\b(not accessible|exception|error|failed|broken)\b', re.I), 0.70),
    (re.compile(r'\b(no .* found|not found|missing|absent)\b', re.I), 0.78),
]


@dataclass
class ClassifiedMemory:
    """A memory classified into a bucket with confidence."""
    text: str
    bucket: str                  # CONSTRAINTS, PROCEDURES, FACTS, HISTORICAL, UNCERTAINTIES
    confidence: float            # 0.0-1.0 within bucket
    mem_id: str = ""
    source: str = ""             # memo_search, graph_walk, constraint, etc.
    entity_refs: list = field(default_factory=list)

    def to_bullet(self, include_id: bool = False) -> str:
        """Format as a terse bullet point."""
        # Clean up the text: remove XML tags, trim whitespace
        text = re.sub(r'<[^>]+>', '', self.text).strip()
        # Truncate long memories
        if len(text) > 150:
            text = text[:147] + "..."
        suffix = f" [{self.mem_id}]" if include_id and self.mem_id else ""
        return f"- {text}{suffix}"


def classify_memory(text: str, source: str = "", existing_type: str = "") -> ClassifiedMemory:
    """Classify a memory text into a bucket with confidence score.

    Checks patterns in priority order: constraint > procedure > historical > uncertainty > fact.
    """
    if not text or not text.strip():
        return ClassifiedMemory(text=text, bucket="FACTS", confidence=0.50, source=source)

    # If it came from the constraints system, it's a constraint
    if existing_type in ("decision", "requirement", "config", "convention"):
        return ClassifiedMemory(
            text=text, bucket="CONSTRAINTS",
            confidence=0.95 if existing_type == "decision" else 0.85,
            source=source,
        )

    best_bucket = "FACTS"
    best_confidence = 0.60  # Default fact confidence

    # Check each bucket's patterns
    for patterns, bucket_name in [
        (_CONSTRAINT_PATTERNS, "CONSTRAINTS"),
        (_PROCEDURE_PATTERNS, "PROCEDURES"),
        (_HISTORICAL_PATTERNS, "HISTORICAL"),
        (_UNCERTAINTY_PATTERNS, "UNCERTAINTIES"),
    ]:
        for pattern, conf in patterns:
            if pattern.search(text):
                if conf > best_confidence:
                    best_bucket = bucket_name
                    best_confidence = conf
                break  # Take first match per bucket (patterns are priority-ordered)

    return ClassifiedMemory(
        text=text, bucket=best_bucket,
        confidence=round(best_confidence, 3),
        source=source,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PackagedContext — the main formatter
# ═══════════════════════════════════════════════════════════════════════════════

class PackagedContext:
    """Structured context package ready for prompt injection."""

    def __init__(self):
        self.entities: list[str] = []
        self.classified: list[ClassifiedMemory] = []
        self.query: str = ""

    @classmethod
    def from_search_response(cls, response: dict, query: str = "") -> "PackagedContext":
        """Build from a daemon search response.

        Args:
            response: dict from memo_daemon handle_search(), containing:
                - results: list of scored memo results
                - ca_context: dict of context blocks
            query: the original query text
        """
        pkg = cls()
        pkg.query = query

        # Extract entities via NodeResolver (canonical names) with fallback
        try:
            from node_resolver import resolve_seeds
            seeds = resolve_seeds(query, max_seeds=8)
            if seeds:
                # Deduplicate: if "PostgreSQL" and "Postgres" both appear, keep the longer one
                raw_names = [s.name for s in seeds]
                deduped = []
                for name in sorted(raw_names, key=len, reverse=True):
                    if not any(name.lower() in existing.lower() or existing.lower() in name.lower()
                              for existing in deduped):
                        deduped.append(name)
                pkg.entities = sorted(deduped)
            else:
                from entity_boost import extract_entities_from_text
                pkg.entities = sorted(extract_entities_from_text(query))
        except Exception:
            try:
                from entity_boost import extract_entities_from_text
                pkg.entities = sorted(extract_entities_from_text(query))
            except Exception:
                pass

        # Classify memo search results
        results = response.get("results", [])
        for r in results:
            text = r.get("memory", "")
            if not text:
                continue
            cm = classify_memory(text, source=r.get("_source", "memo_search"))
            cm.mem_id = r.get("mem_id", r.get("id", ""))[:8]
            cm.entity_refs = r.get("_entities", [])
            # Use salience score to boost confidence
            salience = r.get("final", r.get("score", 0.5))
            cm.confidence = round(min(1.0, cm.confidence * (0.7 + 0.3 * salience)), 3)
            pkg.classified.append(cm)

        # Add constraints from ca_context
        ca = response.get("ca_context", {})
        for key, content in ca.items():
            if not content or not isinstance(content, str):
                continue
            # Strip XML tags for classification
            clean = re.sub(r'<[^>]+>', '', content).strip()
            if not clean:
                continue

            # Constraints block gets special handling
            if "<CONSTRAINTS>" in content:
                for line in clean.split("\n"):
                    line = line.strip()
                    if line and len(line) > 5:
                        cm = classify_memory(line, source="constraint")
                        cm.bucket = "CONSTRAINTS"
                        cm.confidence = max(cm.confidence, 0.85)
                        pkg.classified.append(cm)
            # Skip raw graph/narrative dumps — they're internal context
            # The classified memories already capture their signal

        # Check for constraint system results
        try:
            from constraints import check_query_constraints
            constraints = check_query_constraints(query)
            for c in constraints:
                cm = ClassifiedMemory(
                    text=c["content"],
                    bucket="CONSTRAINTS",
                    confidence=c["confidence"],
                    source="constraint_system",
                )
                if not any(existing.text == cm.text for existing in pkg.classified):
                    pkg.classified.append(cm)
        except Exception:
            pass

        # Generate constraint statements from graph relationships
        # When entities are known, pull DEPENDS_ON/CONFIGURED_BY/RUNS_ON edges
        # and render them as structured constraints even if no memory says "must"
        try:
            from node_resolver import resolve_seeds
            import db as _db

            seeds = resolve_seeds(query, max_seeds=5)
            seed_ids = [s.entity_id for s in seeds] if seeds else []
            if seed_ids:
                placeholders = ", ".join(["%s"] * len(seed_ids))
                rel_result = _db.pg_query(
                    f"""SELECT e1.name, r.predicate, e2.name,
                               COALESCE(r.observation_count, 1) as obs
                        FROM relationships r
                        JOIN entities e1 ON e1.id = r.source_id
                        JOIN entities e2 ON e2.id = r.target_id
                        WHERE r.source_id IN ({placeholders})
                        AND r.predicate IN ('DEPENDS_ON', 'CONFIGURED_BY', 'RUNS_ON', 'USES')
                        AND COALESCE(r.decay_score, 1.0) > 0.3
                        ORDER BY obs DESC
                        LIMIT 8;""",
                    tuple(seed_ids)
                )
                if rel_result:
                    for line in rel_result.split("\n"):
                        if "|" not in line:
                            continue
                        parts = line.split("|")
                        if len(parts) < 4:
                            continue
                        src = parts[0].strip()
                        pred = parts[1].strip()
                        tgt = parts[2].strip()
                        obs = int(parts[3].strip() or "1")

                        # Render as natural constraint
                        pred_text = pred.lower().replace("_", " ")
                        text = f"{src} {pred_text} {tgt}"
                        conf = min(0.85, 0.70 + obs * 0.02)

                        cm = ClassifiedMemory(
                            text=text,
                            bucket="CONSTRAINTS",
                            confidence=conf,
                            source="graph_relationship",
                        )
                        if not any(existing.text == cm.text for existing in pkg.classified):
                            pkg.classified.append(cm)
        except Exception:
            pass

        return pkg

    def to_prompt_block(self, include_ids: bool = False,
                        max_per_bucket: int = 5,
                        max_total_tokens: int = 800) -> str:
        """Render the structured prompt block.

        Returns a formatted string like:
            [RECOVERED_CONTEXT]

            ENTITIES
            - Docker
            - anenji-bridge-tcp

            CONSTRAINTS
            - Must use host networking for the TCP bridge.
            ...
        """
        if not self.classified and not self.entities:
            return ""

        sections = []

        # ENTITIES — always first
        if self.entities:
            sections.append("ENTITIES\n" + "\n".join(f"- {e}" for e in self.entities[:10]))

        # Group classified memories by bucket, sort by confidence within each
        buckets = {}
        for cm in self.classified:
            buckets.setdefault(cm.bucket, []).append(cm)

        for bucket_name in buckets:
            buckets[bucket_name].sort(key=lambda x: x.confidence, reverse=True)

        # Render in priority order
        token_est = sum(len(s) // 4 for s in sections)

        for bucket_name in BUCKETS:
            items = buckets.get(bucket_name, [])
            if not items:
                continue

            # Deduplicate within bucket (fuzzy — >70% word overlap)
            deduped = self._dedup_bucket(items)

            # Take top N per bucket
            selected = deduped[:max_per_bucket]

            # Token check
            bucket_text = "\n".join(cm.to_bullet(include_ids) for cm in selected)
            bucket_tokens = len(bucket_text) // 4
            if token_est + bucket_tokens > max_total_tokens:
                # Trim to fit
                remaining_tokens = max_total_tokens - token_est
                if remaining_tokens < 20:
                    break
                trimmed = []
                for cm in selected:
                    bullet = cm.to_bullet(include_ids)
                    if token_est + len(bullet) // 4 > max_total_tokens:
                        break
                    trimmed.append(bullet)
                    token_est += len(bullet) // 4
                if trimmed:
                    if bucket_name == "CONSTRAINTS":
                        directive = ("If the user's request conflicts with any constraint below, "
                                     "state the conflict and the risk before proceeding.")
                        sections.append(f"{bucket_name}\n{directive}\n" + "\n".join(trimmed))
                    else:
                        sections.append(f"{bucket_name}\n" + "\n".join(trimmed))
                break

            # Inline dissonance directive: when constraints are present,
            # instruct the model to flag conflicts before proceeding
            if bucket_name == "CONSTRAINTS":
                directive = ("If the user's request conflicts with any constraint below, "
                             "state the conflict and the risk before proceeding.")
                sections.append(f"{bucket_name}\n{directive}\n" + bucket_text)
            else:
                sections.append(f"{bucket_name}\n" + bucket_text)
            token_est += bucket_tokens

        if not sections:
            return ""

        return "[RECOVERED_CONTEXT]\n\n" + "\n\n".join(sections)

    def _dedup_bucket(self, items: list) -> list:
        """Remove near-duplicate items within a bucket."""
        if len(items) <= 1:
            return items

        kept = [items[0]]
        for candidate in items[1:]:
            is_dup = False
            cand_words = set(candidate.text.lower().split())
            for existing in kept:
                exist_words = set(existing.text.lower().split())
                if not cand_words or not exist_words:
                    continue
                overlap = len(cand_words & exist_words) / min(len(cand_words), len(exist_words))
                if overlap > 0.70:
                    is_dup = True
                    break
            if not is_dup:
                kept.append(candidate)
        return kept

    def to_dict(self) -> dict:
        """Serialize for JSON output / debugging."""
        buckets = {}
        for cm in self.classified:
            buckets.setdefault(cm.bucket, []).append({
                "text": cm.text[:100],
                "confidence": cm.confidence,
                "source": cm.source,
                "mem_id": cm.mem_id,
            })
        return {
            "query": self.query,
            "entities": self.entities,
            "buckets": buckets,
            "total_classified": len(self.classified),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: format a query end-to-end
# ═══════════════════════════════════════════════════════════════════════════════

def format_query_context(query: str, limit: int = 10) -> str:
    """One-shot: search + classify + format.

    Connects to the daemon, runs search, classifies results,
    and returns the formatted prompt block.
    """
    try:
        from openclaw_memo import _daemon_request, CLYDE_USER
        response = _daemon_request("search", {
            "query": query,
            "user_id": CLYDE_USER,
            "limit": limit,
            "skip_gate": True,  # Formatter does its own filtering
        })
        if not response or not response.get("ok"):
            return ""

        pkg = PackagedContext.from_search_response(response, query=query)
        return pkg.to_prompt_block()
    except Exception as e:
        return f"[context formatting error: {e}]"


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RED = "\033[91m"
RESET = "\033[0m"

BUCKET_COLORS = {
    "CONSTRAINTS": RED,
    "PROCEDURES": YELLOW,
    "FACTS": CYAN,
    "HISTORICAL": DIM,
    "UNCERTAINTIES": YELLOW,
}

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenClaw Prompt Formatter — Structured Context Packaging"
    )
    sub = parser.add_subparsers(dest="command")

    p_fmt = sub.add_parser("format", help="Format context for a query")
    p_fmt.add_argument("query", help="Query text")
    p_fmt.add_argument("--limit", type=int, default=10)
    p_fmt.add_argument("--ids", action="store_true", help="Include memory IDs")
    p_fmt.add_argument("--json", action="store_true", help="JSON output")
    p_fmt.add_argument("--raw", action="store_true", help="Show raw prompt block only")

    p_cls = sub.add_parser("classify", help="Classify a single memory text")
    p_cls.add_argument("text", help="Memory text to classify")

    p_batch = sub.add_parser("batch", help="Format multiple queries")

    args = parser.parse_args()

    if args.command == "format":
        from openclaw_memo import _daemon_request, CLYDE_USER
        t0 = time.time()
        response = _daemon_request("search", {
            "query": args.query,
            "user_id": CLYDE_USER,
            "limit": args.limit,
            "skip_gate": True,
        })
        elapsed = (time.time() - t0) * 1000

        if not response or not response.get("ok"):
            print(f"  {RED}Search failed{RESET}")
            sys.exit(1)

        pkg = PackagedContext.from_search_response(response, query=args.query)

        if args.json:
            print(json.dumps(pkg.to_dict(), indent=2))
        elif args.raw:
            block = pkg.to_prompt_block(include_ids=args.ids)
            print(block)
        else:
            print(f"\n  {BOLD}Query: \"{args.query}\"{RESET}")
            print(f"  {DIM}({len(pkg.classified)} memories classified in {elapsed:.0f}ms){RESET}")
            print(f"  {DIM}Entities: {', '.join(pkg.entities) or '(none)'}{RESET}\n")

            # Show classification breakdown
            buckets = {}
            for cm in pkg.classified:
                buckets.setdefault(cm.bucket, []).append(cm)

            for bucket_name in BUCKETS:
                items = buckets.get(bucket_name, [])
                if not items:
                    continue
                color = BUCKET_COLORS.get(bucket_name, RESET)
                print(f"  {color}{BOLD}{bucket_name}{RESET} ({len(items)})")
                for cm in sorted(items, key=lambda x: x.confidence, reverse=True)[:5]:
                    print(f"    {cm.confidence:.2f}  {cm.text[:70]}  "
                          f"{DIM}({cm.source}){RESET}")
                print()

            # Show formatted prompt block
            block = pkg.to_prompt_block(include_ids=args.ids)
            print(f"  {BOLD}{'─' * 60}{RESET}")
            print(f"  {BOLD}Formatted prompt block:{RESET}\n")
            for line in block.split("\n"):
                print(f"  {line}")
            print()

    elif args.command == "classify":
        cm = classify_memory(args.text)
        color = BUCKET_COLORS.get(cm.bucket, RESET)
        print(f"\n  Text: \"{args.text[:80]}\"")
        print(f"  Bucket: {color}{cm.bucket}{RESET}")
        print(f"  Confidence: {cm.confidence:.3f}\n")

    elif args.command == "batch":
        queries = [
            "restart the bridge",
            "set up the database for the inverter bridge",
            "how do I fix the bridge error",
            "what does the bridge depend on",
            "why postgres here",
        ]

        from openclaw_memo import _daemon_request, CLYDE_USER

        for q in queries:
            response = _daemon_request("search", {
                "query": q, "user_id": CLYDE_USER, "limit": 8, "skip_gate": True,
            })
            if not response or not response.get("ok"):
                print(f"  {RED}Failed: {q}{RESET}")
                continue

            pkg = PackagedContext.from_search_response(response, query=q)
            block = pkg.to_prompt_block()

            bucket_counts = {}
            for cm in pkg.classified:
                bucket_counts[cm.bucket] = bucket_counts.get(cm.bucket, 0) + 1

            print(f"\n  {BOLD}Q: \"{q}\"{RESET}")
            print(f"  Entities: {', '.join(pkg.entities[:5]) or '(none)'}")
            print(f"  Buckets: {bucket_counts}")
            print(f"  Block size: {len(block)} chars (~{len(block)//4} tokens)")
            print()
            for line in block.split("\n"):
                print(f"    {line}")
            print()

    else:
        parser.print_help()
