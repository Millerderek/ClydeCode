#!/bin/bash
# Session Hygiene — prevent token bloat across all agents
#
# Runs via cron to:
# 1. Clean up stale sessions (>48h inactive)
# 2. Log session sizes for monitoring
# 3. Compact large sessions that exceed token thresholds
#
# Cron: 0 */6 * * * /root/.openclaw/scripts/session-hygiene.sh >> /tmp/session-hygiene.log 2>&1

set -euo pipefail

LOG_TAG="[session-hygiene]"
log() { echo "$(date -Iseconds) $LOG_TAG $*"; }

log "Starting session hygiene run"

# 1. Run cleanup across all agents (removes stale sessions)
log "Running session cleanup (dry-run first)..."
DRY=$(openclaw sessions cleanup --all-agents --dry-run 2>&1) || true
echo "$DRY" | grep -E "evict|remove|compact|stale|prune" && {
    log "Actions found — enforcing cleanup"
    openclaw sessions cleanup --all-agents --enforce 2>&1 || true
} || {
    log "No cleanup actions needed"
}

# 2. Log current session stats
log "Current session state:"
openclaw sessions --all-agents 2>&1 | tail -n +3 | while read -r line; do
    log "  $line"
done

# 3. Check JSONL file sizes and warn if any are getting large
log "Session file sizes:"
for dir in /root/.openclaw/agents/*/sessions/; do
    agent=$(basename "$(dirname "$dir")")
    for f in "$dir"*.jsonl; do
        [ -f "$f" ] || continue
        size=$(stat -c%s "$f" 2>/dev/null || echo 0)
        lines=$(wc -l < "$f" 2>/dev/null || echo 0)
        size_kb=$((size / 1024))
        log "  $agent/$(basename "$f" .jsonl): ${size_kb}KB, ${lines} lines"

        # Warn if session file exceeds 500KB
        if [ "$size" -gt 512000 ]; then
            log "  ⚠️  Large session file: ${size_kb}KB — consider /new in that session"
        fi
    done
done

log "Session hygiene complete"
