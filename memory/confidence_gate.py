#!/usr/bin/env python3
"""
confidence_gate.py -- Confidence-Gated Context Injection for OpenClaw.

Every token must earn its place. This gate sits between all context
producers (memo search, graph, narratives, open questions, peripheral
nudges, patterns, identity) and the prompt assembler. Each subsystem's
output is a *proposal*, not an injection.

The gate enforces:
  1. Minimum confidence thresholds per source type
  2. Token budget (total + per-source caps)
  3. Redundancy suppression (word-overlap dedup)
  4. Adaptive thresholds (query complexity, session state)
  5. Earn-back mechanism (within-session learning)

Usage:
    from confidence_gate import ConfidenceGate, ContextProposal, classify_query_complexity

    gate = ConfidenceGate()
    proposals = [
        ContextProposal("memo_search", text, confidence=0.72, token_estimate=80),
        ContextProposal("graph", graph_text, confidence=0.55, token_estimate=350),
        ...
    ]
    admitted = gate.gate(proposals, query_complexity="moderate", session_context={})
"""

import logging
import re
import time

log = logging.getLogger("confidence_gate")

# ═══════════════════════════════════════════════════════════════════════════════
# ContextProposal
# ═══════════════════════════════════════════════════════════════════════════════

SOURCE_TAGS = {
    "memo_search":      "MEMORIES",
    "graph":            "GRAPH_CONTEXT",
    "narrative":        "NARRATIVE_CONTEXT",
    "open_question":    "OPEN_QUESTIONS",
    "peripheral":       "AWARENESS",
    "patterns":         "PATTERNS",
    "identity":         "IDENTITY",
    "session_buffer":   "SESSION_BUFFER",
    "relevant_context": "RELEVANT_CONTEXT",
}


class ContextProposal:
    """A proposed context block from any subsystem."""

    __slots__ = ("source_type", "content", "confidence", "token_estimate",
                 "entity_refs", "metadata")

    def __init__(self, source_type, content, confidence=0.0,
                 token_estimate=None, entity_refs=None, metadata=None):
        self.source_type = source_type
        self.content = content
        self.confidence = confidence
        # Estimate tokens as len/4 (closer to actual LLM tokens than word count)
        self.token_estimate = token_estimate or max(1, len(content) // 4)
        self.entity_refs = entity_refs or []
        self.metadata = metadata or {}

    def to_injection(self):
        """Format for prompt injection with source attribution."""
        tag = SOURCE_TAGS.get(self.source_type, self.source_type.upper())
        return f"<{tag}>\n{self.content}\n</{tag}>"

    def __repr__(self):
        return (f"ContextProposal({self.source_type!r}, "
                f"conf={self.confidence:.3f}, "
                f"tokens={self.token_estimate})")


# ═══════════════════════════════════════════════════════════════════════════════
# Session earn-back state
# ═══════════════════════════════════════════════════════════════════════════════

class SessionGateState:
    """Tracks within-session threshold adjustments from outcome feedback."""

    def __init__(self):
        self.threshold_adjustments = {}  # {source_type: float offset}
        self.outcomes = {}  # {source_type: {"useful": N, "ignored": N, "harmful": N}}

    def record_outcome(self, source_type, was_useful):
        """Adjust threshold based on whether injected context was used."""
        current = self.threshold_adjustments.get(source_type, 0.0)
        if was_useful:
            # Lower threshold slightly (easier admission next turn)
            self.threshold_adjustments[source_type] = max(-0.15, current - 0.03)
            label = "useful"
        else:
            # Raise threshold slightly (harder admission next turn)
            self.threshold_adjustments[source_type] = min(0.15, current + 0.05)
            label = "ignored"

        if source_type not in self.outcomes:
            self.outcomes[source_type] = {"useful": 0, "ignored": 0, "harmful": 0}
        self.outcomes[source_type][label] += 1

    def get_adjusted_threshold(self, source_type, base_threshold):
        offset = self.threshold_adjustments.get(source_type, 0.0)
        return max(0.15, min(0.95, base_threshold + offset))


# ═══════════════════════════════════════════════════════════════════════════════
# Query complexity classifier (heuristic — ML replaces later)
# ═══════════════════════════════════════════════════════════════════════════════

_COMPARISON_WORDS = frozenset([
    "compare", "vs", "versus", "difference", "between",
    "should i", "which", "pros and cons", "trade-off", "tradeoff",
])

_PLANNING_WORDS = frozenset([
    "plan", "design", "architect", "strategy", "how should",
    "what if", "approach", "roadmap", "migrate", "refactor",
])


def classify_query_complexity(message):
    """
    Classify query complexity to set context budget.

    Returns: "trivial" | "simple" | "moderate" | "complex"
    """
    if not message or not message.strip():
        return "trivial"

    text = message.strip()
    tokens = len(text.split())
    is_question = text.endswith("?")
    lower = text.lower()

    has_comparison = any(w in lower for w in _COMPARISON_WORDS)
    has_planning = any(w in lower for w in _PLANNING_WORDS)

    # Count entity-like capitalized words (rough heuristic)
    entity_mentions = len(re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text))

    # Trivial: greetings, acks, very short non-questions
    # But not if it contains technical/infrastructure terms (likely a real query)
    has_technical = bool(re.search(
        r'\b(docker|nginx|mqtt|vps|server|deploy|api|config|cron|daemon|'
        r'postgres|redis|qdrant|inverter|modbus|esphome|mqtt|graph|memory)\b',
        lower,
    ))
    if tokens <= 3 and not is_question and entity_mentions < 2 and not has_technical:
        return "trivial"

    # Complex: multi-entity, comparison, planning, or long-form
    if has_comparison or has_planning:
        return "complex"
    if entity_mentions >= 2 and is_question:
        return "complex"
    if tokens >= 30:
        return "complex"

    # Simple: short questions, single-entity lookups
    if tokens <= 10 and is_question:
        return "simple"
    if tokens <= 8:
        return "simple"

    # Moderate: everything else
    return "moderate"


# ═══════════════════════════════════════════════════════════════════════════════
# ConfidenceGate
# ═══════════════════════════════════════════════════════════════════════════════

class ConfidenceGate:
    """
    Central gating mechanism. Controls what enters the context window.
    Every token must earn its place.
    """

    # Base thresholds — the bar each source must clear
    BASE_THRESHOLDS = {
        "memo_search":      0.45,
        "graph":            0.50,
        "narrative":        0.40,
        "open_question":    0.35,
        "peripheral":       0.55,
        "patterns":         0.45,
        "identity":         0.30,   # Lightweight, low bar
        "session_buffer":   0.70,
        "relevant_context": 0.30,
    }

    # Priority order for budget allocation (lower = higher priority)
    PRIORITY_ORDER = {
        "memo_search":      1,
        "graph":            2,
        "narrative":        3,
        "open_question":    4,
        "relevant_context": 5,
        "patterns":         6,
        "peripheral":       7,
        "identity":         8,
        "session_buffer":   9,
    }

    # Per-source token caps
    TOKEN_CAPS = {
        "memo_search":      500,
        "graph":            600,
        "narrative":        400,
        "open_question":    200,
        "relevant_context": 500,
        "patterns":         300,
        "peripheral":       200,
        "identity":         150,
        "session_buffer":   200,
    }

    # Total budget by query complexity
    TOTAL_BUDGETS = {
        "trivial":  1000,
        "simple":   1500,
        "moderate": 2000,
        "complex":  3000,
    }

    def __init__(self):
        self.session_state = SessionGateState()
        self.injection_log = []
        self._total_proposed = 0
        self._total_admitted = 0
        self._total_tokens_used = 0
        self._rejection_reasons = {
            "below_threshold": 0,
            "over_budget": 0,
            "redundant": 0,
        }

    def gate(self, proposals, query_complexity="moderate", session_context=None):
        """
        Main entry point. Takes all proposals, returns only those
        that earned their tokens.

        Args:
            proposals: list of ContextProposal
            query_complexity: "trivial" | "simple" | "moderate" | "complex"
            session_context: dict with turn_number, nearest_deadline_days, etc.

        Returns:
            list of ContextProposal that passed the gate
        """
        if not proposals:
            return []

        session_context = session_context or {}
        t0 = time.time()

        # Step 1: Compute adaptive thresholds for this turn
        thresholds = self._adaptive_thresholds(query_complexity, session_context)

        # Step 2: Confidence filter
        cleared = []
        for p in proposals:
            threshold = self.session_state.get_adjusted_threshold(
                p.source_type,
                thresholds.get(p.source_type, 0.50),
            )
            if p.confidence >= threshold:
                cleared.append(p)
            else:
                self._rejection_reasons["below_threshold"] += 1
                log.debug(
                    "gate: rejected %s (conf=%.3f < threshold=%.3f)",
                    p.source_type, p.confidence, threshold,
                )

        # Step 3: Sort by priority, then by confidence within tier
        cleared.sort(key=lambda p: (
            self.PRIORITY_ORDER.get(p.source_type, 99),
            -p.confidence,
        ))

        # Step 4: Apply token budget
        total_budget = self.TOTAL_BUDGETS.get(query_complexity, 2000)
        budgeted = self._apply_budget(cleared, total_budget)

        # Step 5: Redundancy suppression
        deduplicated = self._dedup(budgeted)

        # Step 6: Log stats
        tokens_used = sum(p.token_estimate for p in deduplicated)
        elapsed_ms = (time.time() - t0) * 1000

        self._total_proposed += len(proposals)
        self._total_admitted += len(deduplicated)
        self._total_tokens_used += tokens_used

        entry = {
            "proposed": len(proposals),
            "cleared_confidence": len(cleared),
            "cleared_budget": len(budgeted),
            "final": len(deduplicated),
            "tokens_used": tokens_used,
            "tokens_budget": total_budget,
            "query_complexity": query_complexity,
            "elapsed_ms": round(elapsed_ms, 2),
            "sources": [p.source_type for p in deduplicated],
        }
        self.injection_log.append(entry)

        if len(proposals) > len(deduplicated):
            log.info(
                "gate: %d/%d proposals admitted, %d/%d tokens used (%s, %.1fms)",
                len(deduplicated), len(proposals),
                tokens_used, total_budget,
                query_complexity, elapsed_ms,
            )

        return deduplicated

    def _adaptive_thresholds(self, complexity, context):
        """Adjust thresholds based on query and session state."""
        thresholds = dict(self.BASE_THRESHOLDS)

        # Complex queries → lower bar (wider net)
        if complexity == "complex":
            for key in thresholds:
                thresholds[key] = max(0.20, thresholds[key] - 0.10)

        # Trivial queries → higher bar (almost nothing gets in)
        elif complexity == "trivial":
            for key in thresholds:
                thresholds[key] = min(0.90, thresholds[key] + 0.20)

        # First turn → lower bar (context priming)
        if context.get("turn_number", 1) == 1:
            for key in thresholds:
                thresholds[key] = max(0.20, thresholds[key] - 0.15)

        # Deadline pressure → lower bar for goal-related sources
        deadline_days = context.get("nearest_deadline_days")
        if deadline_days is not None and deadline_days <= 3:
            for key in ("narrative", "open_question", "peripheral"):
                if key in thresholds:
                    thresholds[key] = max(0.20, thresholds[key] - 0.15)

        # High correction rate recently → raise bar (be more selective)
        if context.get("correction_rate_7d", 0) > 0.10:
            for key in thresholds:
                thresholds[key] = min(0.90, thresholds[key] + 0.05)

        return thresholds

    def _apply_budget(self, proposals, total_budget):
        """Admit proposals until budget is exhausted."""
        admitted = []
        remaining = total_budget
        source_usage = {}

        for p in proposals:
            cap = self.TOKEN_CAPS.get(p.source_type, 300)
            used = source_usage.get(p.source_type, 0)

            # Per-source cap
            if used + p.token_estimate > cap:
                self._rejection_reasons["over_budget"] += 1
                continue

            # Total budget
            if p.token_estimate > remaining:
                self._rejection_reasons["over_budget"] += 1
                continue

            admitted.append(p)
            remaining -= p.token_estimate
            source_usage[p.source_type] = used + p.token_estimate

        return admitted

    def _dedup(self, proposals):
        """Remove proposals that overlap >80% with already-admitted ones."""
        if len(proposals) <= 1:
            return proposals

        kept = [proposals[0]]

        for candidate in proposals[1:]:
            is_redundant = False
            for existing in kept:
                overlap = _text_overlap(candidate.content, existing.content)
                if overlap > 0.80:
                    self._rejection_reasons["redundant"] += 1
                    log.debug(
                        "gate: suppressed redundant %s (overlap=%.2f with %s)",
                        candidate.source_type, overlap, existing.source_type,
                    )
                    is_redundant = True
                    break
            if not is_redundant:
                kept.append(candidate)

        return kept

    def record_outcome(self, source_type, was_useful):
        """Called by outcome logger after response is evaluated."""
        self.session_state.record_outcome(source_type, was_useful)

    def stats(self):
        """Gate performance stats for monitoring."""
        return {
            "turns": len(self.injection_log),
            "total_proposed": self._total_proposed,
            "total_admitted": self._total_admitted,
            "admission_rate": round(
                self._total_admitted / max(1, self._total_proposed), 3
            ),
            "total_tokens_used": self._total_tokens_used,
            "rejection_reasons": dict(self._rejection_reasons),
            "threshold_adjustments": dict(
                self.session_state.threshold_adjustments
            ),
            "earn_back": dict(self.session_state.outcomes),
            "recent_log": self.injection_log[-5:] if self.injection_log else [],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _text_overlap(a, b):
    """Containment similarity between two text blocks."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    smaller = min(len(words_a), len(words_b))
    return len(intersection) / smaller if smaller > 0 else 0.0


def estimate_confidence_for_ca(source_type, content, query=None):
    """
    Estimate a confidence score for a CA context block.

    Since the CA subsystems return plain strings without confidence scores,
    this function uses heuristics to assign one.

    Args:
        source_type: "graph", "narrative", "peripheral", "patterns", "identity"
        content: the context string
        query: optional query text for relevance estimation

    Returns:
        float: estimated confidence 0.0-1.0
    """
    if not content or not content.strip():
        return 0.0

    # Base confidence by source type (non-empty = some value)
    base = {
        "graph":            0.55,
        "narrative":        0.50,
        "peripheral":       0.45,
        "patterns":         0.50,
        "identity":         0.40,
        "relevant_context": 0.45,
    }.get(source_type, 0.40)

    # Boost for longer, richer content (up to +0.15)
    content_len = len(content)
    if content_len > 500:
        base += 0.10
    elif content_len > 200:
        base += 0.05

    # Boost for query word overlap (up to +0.20)
    if query:
        overlap = _text_overlap(query, content)
        base += min(0.20, overlap * 0.40)

    return min(1.0, round(base, 3))


def score_memo_results(results):
    """
    Convert salience-scored memo results into ContextProposals.

    Args:
        results: list of dicts from salience_score(), each with
                 'final', 'memory', 'mem_id', etc.

    Returns:
        list of ContextProposal (one per result, source_type="memo_search")
    """
    proposals = []
    for r in results:
        text = r.get("memory", "")
        if not text:
            continue
        proposals.append(ContextProposal(
            source_type="memo_search",
            content=text,
            confidence=r.get("final", r.get("score", 0.0)),
            entity_refs=[],
            metadata={"mem_id": r.get("mem_id", ""), "id": r.get("id", "")},
        ))
    return proposals


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    if "--complexity" in sys.argv:
        tests = [
            "hey",
            "yes",
            "how do I parse JSON?",
            "what was the config?",
            "compare Docker Compose vs Kubernetes for this deployment",
            "plan the migration from direct routing to operator connect for Dometic",
            "restart nginx",
            ("I have a complex multi-step deployment involving three containers, "
             "a reverse proxy, TLS termination, and I need to figure out the "
             "optimal routing strategy for the VoIP traffic"),
        ]
        print("\n=== Query Complexity Classification ===")
        for t in tests:
            c = classify_query_complexity(t)
            budget = ConfidenceGate.TOTAL_BUDGETS[c]
            print(f"  [{c:8s}] ({budget:4d} tokens)  {t[:70]}")
        sys.exit(0)

    # Default: simulate gate with mock proposals
    gate = ConfidenceGate()
    proposals = [
        ContextProposal("memo_search", "Derek uses VPS at 217.216.85.157 from Contabo",
                         confidence=0.72),
        ContextProposal("memo_search", "Docker container set to restart=always",
                         confidence=0.38),
        ContextProposal("graph", "Entity: Contabo → hosts → VPS\nEntity: VPS → runs → Docker",
                         confidence=0.55),
        ContextProposal("narrative", "Stale narrative from 45 days ago about old project",
                         confidence=0.25),
        ContextProposal("peripheral", "[deadline] Bill due in 2 days: Contabo VPS",
                         confidence=0.60),
        ContextProposal("patterns", "Pattern: Derek prefers Docker Compose over raw docker run",
                         confidence=0.48),
        ContextProposal("identity", "Era: Builder phase, focused on infrastructure",
                         confidence=0.40),
    ]

    print("\n=== Confidence Gate Simulation ===")
    print(f"\nProposals submitted: {len(proposals)}")
    for p in proposals:
        print(f"  {p}")

    admitted = gate.gate(proposals, query_complexity="moderate")

    print(f"\nAdmitted: {len(admitted)}")
    for p in admitted:
        print(f"  ✓ {p}")
        print(f"    → {p.content[:60]}...")

    print(f"\nStats: {gate.stats()}")
