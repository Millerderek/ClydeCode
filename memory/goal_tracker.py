#!/usr/bin/env python3
"""
goal_tracker.py -- Goal and Open Question Management for OpenClaw.

Phase 2 of the Cognitive Architecture. Tracks active goals (objectives the
agent is working toward) and open questions (unresolved uncertainties).
Provides goal_proximity scoring for salience_engine integration.

PG tables: goals, open_questions, goal_memory_links

Usage:
    python3 goal_tracker.py add-goal "Build Phase 2" --priority high
    python3 goal_tracker.py list-goals
    python3 goal_tracker.py update-goal 1 --status completed
    python3 goal_tracker.py add-question "How should salience weights be tuned?"
    python3 goal_tracker.py list-questions
    python3 goal_tracker.py resolve-question 1 "Based on empirical testing"
    python3 goal_tracker.py proximity "scoring formula weights"
    python3 goal_tracker.py link-memory 1 <uuid> --relevance 0.8
"""

import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import db


# ═══════════════════════════════════════════════════════════════════════════════
# Goal CRUD
# ═══════════════════════════════════════════════════════════════════════════════

def add_goal(title, description=None, priority="normal", parent_goal_id=None, tags=None):
    """
    Add a new goal. Returns goal id or -1 on failure.
    """
    tags_list = tags if tags else []

    result = db.pg_query(
        "INSERT INTO goals (title, description, priority, parent_goal_id, tags) "
        "VALUES (%s, %s, %s, %s, %s::text[]) "
        "ON CONFLICT (title) DO UPDATE SET "
        "description = COALESCE(EXCLUDED.description, goals.description), "
        "priority = EXCLUDED.priority, "
        "updated_at = NOW() "
        "RETURNING id;",
        (title, description, priority,
         int(parent_goal_id) if parent_goal_id else None,
         tags_list)
    )
    if result:
        try:
            return int(result.strip().split("|")[0])
        except (ValueError, IndexError):
            pass
    return -1


def update_goal(goal_id, status=None, description=None):
    """Update a goal's status and/or description."""
    parts = ["updated_at = NOW()"]
    params = []
    if status:
        parts.append("status = %s")
        params.append(status)
        if status == "completed":
            parts.append("completed_at = NOW()")
    if description:
        parts.append("description = %s")
        params.append(description)

    params.append(int(goal_id))
    set_clause = ", ".join(parts)
    db.pg_execute(f"UPDATE goals SET {set_clause} WHERE id = %s;", tuple(params))


def get_active_goals(limit=10):
    """Get active goals ordered by priority then recency."""
    result = db.pg_query(
        "SELECT id, title, description, status, priority, "
        "created_at, updated_at, parent_goal_id, tags "
        "FROM goals WHERE status = 'active' "
        "ORDER BY "
        "CASE priority "
        "  WHEN 'critical' THEN 0 "
        "  WHEN 'high' THEN 1 "
        "  WHEN 'normal' THEN 2 "
        "  WHEN 'low' THEN 3 "
        "END, "
        "updated_at DESC "
        "LIMIT %s;",
        (int(limit),)
    )
    if not result:
        return []

    goals = []
    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 9:
            goals.append({
                "id": int(parts[0]),
                "title": parts[1],
                "description": parts[2] or None,
                "status": parts[3],
                "priority": parts[4],
                "created_at": parts[5],
                "updated_at": parts[6],
                "parent_goal_id": int(parts[7]) if parts[7] else None,
                "tags": parts[8].strip("{}").split(",") if parts[8] and parts[8] != "{}" else [],
            })
    return goals


# ═══════════════════════════════════════════════════════════════════════════════
# Open Question CRUD
# ═══════════════════════════════════════════════════════════════════════════════

def add_question(question, context=None, goal_id=None):
    """
    Add an open question. Returns question id or -1 on failure.
    """
    result = db.pg_query(
        "INSERT INTO open_questions (question, context, goal_id) "
        "VALUES (%s, %s, %s) "
        "ON CONFLICT (question) DO UPDATE SET "
        "context = COALESCE(EXCLUDED.context, open_questions.context) "
        "RETURNING id;",
        (question, context, int(goal_id) if goal_id else None)
    )
    if result:
        try:
            return int(result.strip().split("|")[0])
        except (ValueError, IndexError):
            pass
    return -1


def resolve_question(question_id, answer):
    """Mark an open question as answered."""
    db.pg_execute(
        "UPDATE open_questions SET status = 'answered', "
        "answer = %s, resolved_at = NOW() "
        "WHERE id = %s;",
        (answer, int(question_id))
    )


def get_open_questions(limit=10):
    """Get open questions ordered by creation date."""
    result = db.pg_query(
        "SELECT id, question, context, status, goal_id, created_at "
        "FROM open_questions WHERE status = 'open' "
        "ORDER BY created_at DESC "
        "LIMIT %s;",
        (int(limit),)
    )
    if not result:
        return []

    questions = []
    for line in result.split("\n"):
        if "|" not in line:
            continue
        parts = line.split("|")
        if len(parts) >= 6:
            questions.append({
                "id": int(parts[0]),
                "question": parts[1],
                "context": parts[2] or None,
                "status": parts[3],
                "goal_id": int(parts[4]) if parts[4] else None,
                "created_at": parts[5],
            })
    return questions


# ═══════════════════════════════════════════════════════════════════════════════
# Goal-Memory linking
# ═══════════════════════════════════════════════════════════════════════════════

def link_memory_to_goal(goal_id, memory_id, relevance=0.5):
    """Link a memory (UUID) to a goal."""
    db.pg_execute(
        "INSERT INTO goal_memory_links (goal_id, memory_id, relevance) "
        "VALUES (%s, %s::uuid, %s) "
        "ON CONFLICT (goal_id, memory_id) DO UPDATE SET relevance = %s;",
        (int(goal_id), str(memory_id), float(relevance), float(relevance))
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Goal Proximity scoring (FTS-based)
# ═══════════════════════════════════════════════════════════════════════════════

def get_goal_proximity(query, limit=5):
    """
    Score how relevant a query is to active goals + open questions.
    Returns float 0.0-1.0.

    Uses word-overlap matching between query words and goal titles/descriptions
    plus open question text. Normalized by total active items.
    """
    if not query or not query.strip():
        return 0.0

    # Extract meaningful words (3+ chars, no stopwords)
    stopwords = {
        "the", "and", "for", "are", "but", "not", "you", "all",
        "can", "had", "her", "was", "one", "our", "out", "has",
        "his", "how", "its", "may", "new", "now", "old", "see",
        "way", "who", "did", "get", "let", "say", "she", "too",
        "use", "this", "that", "with", "have", "from", "they",
        "been", "some", "what", "when", "will", "more", "into",
        "also", "than", "them", "very", "just", "about", "which",
        "their", "there", "would", "could", "should", "where",
    }
    words = [w.lower() for w in re.findall(r'[a-zA-Z]{3,}', query)]
    words = [w for w in words if w not in stopwords]

    if not words:
        return 0.0

    # Build SQL to match against goals + open questions using ILIKE
    # Score = fraction of query words that match any active goal/OQ text
    word_patterns = ['%' + w + '%' for w in words]

    goal_conditions = " OR ".join(
        "(title ILIKE %s OR description ILIKE %s)" for _ in words
    )
    goal_params = []
    for wp in word_patterns:
        goal_params.extend([wp, wp])

    oq_conditions = " OR ".join(
        "(question ILIKE %s OR context ILIKE %s)" for _ in words
    )
    oq_params = []
    for wp in word_patterns:
        oq_params.extend([wp, wp])

    # Count matching goals
    goal_count = db.pg_query(
        f"SELECT COUNT(*) FROM goals WHERE status = 'active' AND ({goal_conditions});",
        tuple(goal_params)
    )
    # Count matching open questions
    oq_count = db.pg_query(
        f"SELECT COUNT(*) FROM open_questions WHERE status = 'open' AND ({oq_conditions});",
        tuple(oq_params)
    )

    # Count total active items
    total_goals = db.pg_query("SELECT COUNT(*) FROM goals WHERE status = 'active';")
    total_oq = db.pg_query("SELECT COUNT(*) FROM open_questions WHERE status = 'open';")

    try:
        g_match = int(goal_count) if goal_count else 0
        oq_match = int(oq_count) if oq_count else 0
        total_g = int(total_goals) if total_goals else 0
        total_oq_n = int(total_oq) if total_oq else 0
    except ValueError:
        return 0.0

    total_items = total_g + total_oq_n
    total_matches = g_match + oq_match

    if total_items == 0:
        return 0.0

    # Also compute per-word hit rate against concatenated goal text
    # For a richer signal, check how many of the query words appear
    word_hit_count = 0
    for w in words:
        wp = '%' + w + '%'
        hit_g = db.pg_query(
            "SELECT COUNT(*) FROM goals WHERE status = 'active' "
            "AND (title ILIKE %s OR description ILIKE %s);",
            (wp, wp)
        )
        hit_oq = db.pg_query(
            "SELECT COUNT(*) FROM open_questions WHERE status = 'open' "
            "AND (question ILIKE %s OR context ILIKE %s);",
            (wp, wp)
        )
        try:
            if (int(hit_g or 0) + int(hit_oq or 0)) > 0:
                word_hit_count += 1
        except ValueError:
            pass

    word_hit_rate = word_hit_count / len(words) if words else 0.0

    # Combine: item match ratio (how many goals match) + word hit rate
    item_ratio = min(1.0, total_matches / max(1, total_items))
    proximity = 0.5 * item_ratio + 0.5 * word_hit_rate

    return round(min(1.0, proximity), 4)


# ═══════════════════════════════════════════════════════════════════════════════
# Deadline pressure (CG remaining item)
# ═══════════════════════════════════════════════════════════════════════════════

def get_nearest_deadline_days():
    """
    Return the number of days until the nearest active goal deadline.
    Returns None if no goals have deadlines set.
    """
    result = db.pg_query(
        "SELECT EXTRACT(EPOCH FROM (deadline - NOW())) / 86400.0 "
        "FROM goals "
        "WHERE status = 'active' AND deadline IS NOT NULL AND deadline > NOW() "
        "ORDER BY deadline ASC LIMIT 1;"
    )
    if result:
        try:
            return round(float(result.strip()), 1)
        except ValueError:
            pass
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

BOLD = "\033[1m"
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
DIM = "\033[2m"
RESET = "\033[0m"

PRIORITY_ICONS = {
    "critical": "\033[91m!!!\033[0m",
    "high": "\033[93m!! \033[0m",
    "normal": "\033[92m!  \033[0m",
    "low": "\033[2m.  \033[0m",
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="OpenClaw Goal & Open Question Tracker (Phase 2)"
    )
    sub = parser.add_subparsers(dest="command")

    # add-goal
    p_ag = sub.add_parser("add-goal", help="Add a new goal")
    p_ag.add_argument("title", help="Goal title")
    p_ag.add_argument("--description", "-d", help="Goal description")
    p_ag.add_argument("--priority", "-p", default="normal",
                       choices=["critical", "high", "normal", "low"])
    p_ag.add_argument("--parent", type=int, help="Parent goal ID")
    p_ag.add_argument("--tags", nargs="*", help="Tags for the goal")

    # update-goal
    p_ug = sub.add_parser("update-goal", help="Update a goal")
    p_ug.add_argument("goal_id", type=int, help="Goal ID")
    p_ug.add_argument("--status", "-s",
                       choices=["active", "completed", "paused", "abandoned"])
    p_ug.add_argument("--description", "-d", help="New description")

    # list-goals
    p_lg = sub.add_parser("list-goals", help="List active goals")
    p_lg.add_argument("--limit", type=int, default=10)

    # add-question
    p_aq = sub.add_parser("add-question", help="Add an open question")
    p_aq.add_argument("question", help="The question")
    p_aq.add_argument("--context", "-c", help="What prompted the question")
    p_aq.add_argument("--goal-id", type=int, help="Link to a goal")

    # resolve-question
    p_rq = sub.add_parser("resolve-question", help="Resolve an open question")
    p_rq.add_argument("question_id", type=int, help="Question ID")
    p_rq.add_argument("answer", help="The answer")

    # list-questions
    p_lq = sub.add_parser("list-questions", help="List open questions")
    p_lq.add_argument("--limit", type=int, default=10)

    # proximity
    p_prox = sub.add_parser("proximity", help="Score goal proximity for a query")
    p_prox.add_argument("query", help="Query text")

    # link-memory
    p_lm = sub.add_parser("link-memory", help="Link a memory to a goal")
    p_lm.add_argument("goal_id", type=int, help="Goal ID")
    p_lm.add_argument("memory_id", help="Memory UUID")
    p_lm.add_argument("--relevance", type=float, default=0.5)

    args = parser.parse_args()

    if args.command == "add-goal":
        gid = add_goal(args.title, args.description, args.priority,
                        args.parent, args.tags)
        if gid > 0:
            print(f"  {GREEN}+{RESET} Goal added: #{gid} \"{args.title}\" [{args.priority}]")
        else:
            print(f"  {YELLOW}!{RESET} Failed to add goal (duplicate or error)")

    elif args.command == "update-goal":
        update_goal(args.goal_id, args.status, args.description)
        print(f"  {GREEN}+{RESET} Goal #{args.goal_id} updated")

    elif args.command == "list-goals":
        goals = get_active_goals(args.limit)
        if not goals:
            print(f"  {DIM}No active goals{RESET}")
        else:
            print(f"\n  {BOLD}Active Goals ({len(goals)}){RESET}\n")
            for g in goals:
                icon = PRIORITY_ICONS.get(g["priority"], "?  ")
                tags = f" {DIM}[{', '.join(g['tags'])}]{RESET}" if g["tags"] else ""
                parent = f" {DIM}(sub of #{g['parent_goal_id']}){RESET}" if g["parent_goal_id"] else ""
                print(f"  {icon} #{g['id']:3d}  {g['title']}{parent}{tags}")
                if g["description"]:
                    print(f"         {DIM}{g['description'][:80]}{RESET}")

    elif args.command == "add-question":
        qid = add_question(args.question, args.context,
                            getattr(args, "goal_id", None))
        if qid > 0:
            print(f"  {GREEN}?{RESET} Question added: #{qid}")
        else:
            print(f"  {YELLOW}!{RESET} Failed to add question (duplicate or error)")

    elif args.command == "resolve-question":
        resolve_question(args.question_id, args.answer)
        print(f"  {GREEN}+{RESET} Question #{args.question_id} resolved")

    elif args.command == "list-questions":
        questions = get_open_questions(args.limit)
        if not questions:
            print(f"  {DIM}No open questions{RESET}")
        else:
            print(f"\n  {BOLD}Open Questions ({len(questions)}){RESET}\n")
            for q in questions:
                goal_ref = f" {DIM}(goal #{q['goal_id']}){RESET}" if q["goal_id"] else ""
                print(f"  {CYAN}?{RESET} #{q['id']:3d}  {q['question']}{goal_ref}")
                if q["context"]:
                    print(f"         {DIM}context: {q['context'][:80]}{RESET}")

    elif args.command == "proximity":
        score = get_goal_proximity(args.query)
        print(f"  Goal proximity for \"{args.query}\": {score:.4f}")

    elif args.command == "link-memory":
        link_memory_to_goal(args.goal_id, args.memory_id, args.relevance)
        print(f"  {GREEN}+{RESET} Linked memory {args.memory_id[:8]} to goal #{args.goal_id} (relevance={args.relevance})")

    else:
        parser.print_help()
