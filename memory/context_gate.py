#!/usr/bin/env python3
"""
context_gate.py -- Heuristic classifier for memory retrieval gating.

Determines whether a prompt is self-contained (skip memory) or referential
(needs context from memory). No LLM call -- pure pattern matching.

Returns a score from 0.0 (definitely self-contained) to 1.0 (definitely
needs context). Threshold at 0.40 -- below that, skip retrieval.

Also provides:
  should_store(response)  -- detects store-worthy events in assistant output
  classify_topic(text)    -- assigns a topic bucket for memory tagging

Usage:
    from context_gate import needs_context, should_store, classify_topic
    score  = needs_context("how do I parse JSON in Python")   # ~0.05
    score  = needs_context("what was that project config?")   # ~0.85
    store  = should_store("Fixed the connection error by updating the timeout")  # True
    topic  = classify_topic("MQTT sensor added to Home Assistant")  # "home_automation"
"""

import re

# ===============================================================================
# Signal patterns
# ===============================================================================

# These push the score UP (needs context)
REFERENTIAL_SIGNALS = [
    # Pronouns without antecedent
    (r"\b(that|the|this|those)\s+(project|client|config|setup|thing|issue|bug|error|server|script|file|setting|deployment|container|service)\b", 0.30),
    (r"\bhis\s+(config|setup|environment|server|tenant)\b", 0.25),
    (r"\b(its|their)\s+(config|setup|status|state)\b", 0.20),

    # Past references
    (r"\b(we discussed|we talked about|last time|previously|earlier|remember when|you said|you mentioned)\b", 0.40),
    (r"\b(what was|where was|when did|how did we|what did we|did we decide|did we set)\b", 0.30),
    (r"\b(the same|like before|as before|again)\b", 0.20),

    # Short follow-ups (vague, need prior context)
    (r"^(yes|yeah|yep|do it|go ahead|proceed|ok|okay|sure|right)\s*[.!?]?\s*$", 0.35),
    (r"^(where|what|how|why|when)\s*[?]?\s*$", 0.30),
    (r"^(and|also|plus|but)\s", 0.20),

    # Named entities that match known projects/clients
    # Add your project/client names to CLYDE_USER_ENTITIES env var
    (r"\b(qdrant|mem0|custodian|autotuner|entity.?boost)\b", 0.20),

    # References to "my" infrastructure
    (r"\b(my (server|vps|docker|container|setup|config|cron|daemon|network))\b", 0.15),
    (r"\b(the (daemon|custodian|ingest|digest|backup|memory))\b", 0.15),
]

# These patterns FORCE score to 1.0 regardless of other signals.
# Use for things that always need memory context no matter what.
FORCE_SEARCH_PATTERNS = [
    # IP addresses — always need context to know what device this is
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    # Add your hostnames to CLYDE_USER_ENTITIES env var
    # File paths in user infrastructure
    r"/root/[a-zA-Z]",
    r"/etc/(nginx|hosts|cron|network)",
    # "remember", "store", "save this" — explicit memory requests
    r"\b(remember this|store this|save (this|that|it)|add (this|that) to (memory|your memory))\b",
    # "what do you know about X" — explicit memory retrieval
    r"\bwhat do you know about\b",
    # MAC addresses
    r"\b([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b",
]

# These push the score DOWN (self-contained)
SELFCONTAINED_SIGNALS = [
    # Tutorial / how-to questions
    (r"\b(how (do|can|to|would) (i|you|we|one))\b", -0.25),
    (r"\b(what is|what are|what does|explain|define|describe)\b", -0.20),
    (r"\b(tutorial|example|syntax|documentation)\b", -0.15),

    # Generic knowledge queries
    (r"\b(in (python|bash|powershell|javascript|go|rust|java|c\+\+))\b", -0.15),
    (r"\b(what is a|what are the|difference between)\b", -0.20),
    (r"\b(best practice|convention|standard|pattern)\b", -0.10),

    # Explicit new context provided
    (r"(here'?s (my|the|a) (code|config|error|log|output|file))", -0.30),
    (r"```", -0.25),  # Code block = self-contained context
    (r"(i have|i got|i see|i'm getting|i'm seeing)\s+(this|an?|the)\s+(error|issue|problem|warning)", -0.15),

    # System commands (action requests, not knowledge retrieval)
    (r"\b(restart|stop|start|reload|check|run|execute|install|update|upgrade|deploy)\s+(the\s+)?(nginx|docker|service|daemon|container|server|cron|script)\b", -0.20),
    (r"\b(install|pip install|npm install|apt install|brew install)\s+\S+", -0.20),
    (r"\b(ls|cd|cat|grep|find|ps|top|df|du|free|uname)\b", -0.15),

    # Long messages with lots of context are usually self-contained
    # (handled in scoring function, not regex)
]

# ===============================================================================
# Scoring
# ===============================================================================

THRESHOLD = 0.40  # Below this, skip memory retrieval

# ===============================================================================
# Store trigger patterns
# Detects assistant responses that contain store-worthy events
# ===============================================================================

# Each tuple: (pattern, event_type)
STORE_TRIGGERS = [
    # Problem resolved
    (r"\b(fixed|resolved|working now|that.?s (it|the fix)|no longer)\b", "fix"),
    (r"\b(root cause|the issue was|the problem was|turned out (to be|it was))\b", "fix"),

    # Configuration change
    (r"\b(updated|changed|modified|set|configured|tuned)\s+(the\s+)?(config|setting|value|parameter|cron|rule|policy)\b", "config"),
    (r"\b(docker compose (up|build|restart|down))\b", "deployment"),
    (r"\b(deployed|redeployed|rebuilt|container (is|now) running)\b", "deployment"),

    # New device/service/entity added
    (r"\b(added|created|registered|integrated|wired\s+up|now in HA|now in home assistant)\b", "addition"),
    (r"\b(entity|device|integration|config entry|config flow)\b.*\b(added|created|loaded)\b", "addition"),

    # Decision made
    (r"\b(decided|going with|chose|chose to|will use|best approach|recommended)\b", "decision"),
    (r"\b(instead of|rather than|over|vs\.?)\b.*\b(because|since|as it)\b", "decision"),

    # Credentials / paths discovered
    (r"\b(token|api.?key|password|credential)\b.*\b(at|in|stored|located)\b", "credential"),
    (r"\b(found (at|in|under)|lives at|stored at|located at)\b.*(/[a-z]|\d{1,3}\.\d)", "location"),
]

# Minimum pattern hits to trigger a store recommendation
STORE_THRESHOLD = 1


def should_store(response_text: str) -> tuple[bool, list[str]]:
    """
    Analyze an assistant response for store-worthy events.

    Returns:
        (bool, list[str]): Whether to store, and which event types were detected.

    Example:
        store, reasons = should_store("Fixed the connection error — root cause was missing timeout")
        # (True, ["fix"])
    """
    if not response_text or len(response_text) < 30:
        return False, []

    hits = []
    for pattern, event_type in STORE_TRIGGERS:
        if re.search(pattern, response_text, re.IGNORECASE):
            if event_type not in hits:
                hits.append(event_type)

    return len(hits) >= STORE_THRESHOLD, hits


# ===============================================================================
# Topic classifier
# ===============================================================================

TOPIC_PATTERNS = [
    ("home_automation",  r"\b(home.?assistant|HA\b|entity|automation|switch\.|sensor\.|binary_sensor|mqtt|zigbee|zwave|homebridge)\b"),
    ("networking",       r"\b(vlan|subnet|tailscale|nginx|dns|cloudflare|dnat|route|gateway|firewall)\b"),
    ("clyde_memory",     r"\b(clyde.?memory|mem0|qdrant|custodian|session.?ingest|conversation.?digest|compaction|context.?gate|memory.?system)\b"),
    ("iot_devices",      r"\b(iot|mqtt|sensor|actuator|firmware|esp32|raspberry.?pi|arduino|zigbee|zwave)\b"),
    ("transport",        r"\b(ev.?charger|evse|charging|vehicle|car|truck|fleet)\b"),
    ("vps_infra",        r"\b(vps|docker|container|compose|systemd|cron|nginx|certbot|letsencrypt)\b"),
    ("clients",          r"\b(client|tenant|operator.?connect|direct.?routing|teams)\b"),  # Add your client names to CLYDE_USER_ENTITIES
    ("api_keys",         r"\b(token|api.?key|credential|secret|password|auth|bearer|openrouter|anthropic|moonshot)\b.*\b(at|in|path|stored|located|root)\b"),
]

DEFAULT_TOPIC = "general"


def classify_topic(text: str) -> str:
    """
    Classify text into a topic bucket for memory tagging.

    Returns the first matching topic, or 'general' if none match.

    Example:
        classify_topic("MQTT sensor added to Home Assistant")
        # "home_automation"
    """
    if not text:
        return DEFAULT_TOPIC

    scores = {}
    for topic, pattern in TOPIC_PATTERNS:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        if matches > 0:
            scores[topic] = matches

    if not scores:
        return DEFAULT_TOPIC

    return max(scores, key=scores.get)


def classify_topics_multi(text: str, max_topics: int = 2) -> list[str]:
    """Return up to N matching topics sorted by match count."""
    if not text:
        return [DEFAULT_TOPIC]

    scores = {}
    for topic, pattern in TOPIC_PATTERNS:
        matches = len(re.findall(pattern, text, re.IGNORECASE))
        if matches > 0:
            scores[topic] = matches

    if not scores:
        return [DEFAULT_TOPIC]

    return sorted(scores, key=scores.get, reverse=True)[:max_topics]


def needs_context(prompt: str) -> float:
    """
    Score how much a prompt needs memory context.

    Returns:
        float: 0.0 (self-contained) to 1.0 (needs context).
               >= THRESHOLD means "search memory".
               < THRESHOLD means "skip, prompt is self-contained".
    """
    if not prompt or not prompt.strip():
        return 0.0

    text = prompt.strip()

    # Force-search patterns override everything — return 1.0 immediately
    for pattern in FORCE_SEARCH_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return 1.0

    score = 0.30  # Neutral baseline -- slight lean toward searching

    # Dynamically score user-defined entities from config
    try:
        from config import USER_ENTITIES
        for entity in USER_ENTITIES:
            if entity.lower() in text.lower():
                score += 0.25
    except ImportError:
        pass

    # Apply referential signals
    for pattern, weight in REFERENTIAL_SIGNALS:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight

    # Apply self-contained signals
    for pattern, weight in SELFCONTAINED_SIGNALS:
        if re.search(pattern, text, re.IGNORECASE):
            score += weight  # weight is negative

    # Length heuristic: very short prompts (< 5 words) are often follow-ups
    word_count = len(text.split())
    if word_count <= 3:
        score += 0.20
    elif word_count <= 6:
        score += 0.10
    elif word_count >= 50:
        score -= 0.15  # Long prompts usually carry their own context
    elif word_count >= 100:
        score -= 0.25

    # Question mark at end is slightly referential (asking about something)
    if text.rstrip().endswith("?"):
        score += 0.05

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, round(score, 3)))


def should_search(prompt: str) -> bool:
    """Convenience: returns True if memory search is warranted."""
    return needs_context(prompt) >= THRESHOLD


# ===============================================================================
# CLI for testing
# ===============================================================================

if __name__ == "__main__":
    import sys

    if "--store" in sys.argv:
        # Test store trigger detection
        test_responses = [
            "Fixed the connection error — root cause was a missing timeout value",
            "Added MQTT sensor at 10.0.0.50 to Home Assistant as Temperature Sensor",
            "Decided to use summarize_anchored instead of hybrid_extract",
            "The API token is stored at ~/.config/myapp/token.env",
            "How do I parse JSON in Python?",
            "Updated the cron schedule from */10 to */2",
            "Deployed myapp with docker compose build --no-cache",
        ]
        print("\n=== Store Trigger Detection ===")
        for r in test_responses:
            store, reasons = should_store(r)
            topic = classify_topic(r)
            flag = "STORE" if store else "skip "
            print(f"  [{flag}] [{topic:16s}]  {r[:70]}")
            if reasons:
                print(f"          reasons: {reasons}")
        sys.exit(0)

    if "--topic" in sys.argv:
        # Test topic classification
        test_texts = [
            "MQTT sensor at 10.0.0.50 added to Home Assistant",
            "VLAN 10 IoT network, gateway 10.0.0.1",
            "ClydeMemory system with Qdrant and Mem0",
            "Docker container rebuilt from source",
            "EV charger installed on dedicated 40A circuit",
            "Client operator connect migration",
            "API token at ~/.config/myapp/token.env",
        ]
        print("\n=== Topic Classification ===")
        for t in test_texts:
            topics = classify_topics_multi(t)
            print(f"  [{', '.join(topics):30s}]  {t[:70]}")
        sys.exit(0)

    # Default: test search gating
    test_prompts = [
        "how do I parse JSON in Python",
        "what was that project config?",
        "yes",
        "do it",
        "restart nginx",
        "where is the url?",
        "what is a VLAN",
        "we discussed the telephony setup last time",
        "here's my code: ```python\nprint('hello')```",
        "the daemon isn't starting",
        "how does ClydeMemory work?",
        "explain kubernetes pod lifecycle",
        "what did we decide about the scoring formula?",
        "install psycopg2-binary",
        "check the custodian cron schedule",
        "what's at 10.0.0.50?",
        "remember this: the staging server is at 10.0.1.100",
        "check /root/myapp/deploy/",
    ]

    if len(sys.argv) > 1 and sys.argv[1] not in ("--store", "--topic"):
        test_prompts = [" ".join(sys.argv[1:])]

    print("\n=== Search Gate Scoring ===")
    for prompt in test_prompts:
        score = needs_context(prompt)
        action = "SEARCH" if score >= THRESHOLD else "SKIP"
        forced = " [FORCED]" if score == 1.0 else ""
        print(f"  [{score:.3f}] {action:6s}{forced}  {prompt[:70]}")
