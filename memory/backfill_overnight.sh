#!/bin/bash
# backfill_overnight.sh — Run full graph backfill in batches overnight
# Processes 100 memories at a time with 60s pause between batches
# Self-terminates when no more unextracted memories remain

cd /root/ClydeMemory
export PYTHONPATH="/root/openclaw-memory:$PYTHONPATH"

LOG="/root/ClydeMemory/logs/backfill_$(date +%Y%m%d).log"
mkdir -p /root/ClydeMemory/logs

echo "=== Backfill started: $(date) ===" >> "$LOG"

BATCH=1
while true; do
    echo "--- Batch $BATCH: $(date) ---" >> "$LOG"
    OUTPUT=$(python3 backfill_graph.py --limit 100 2>&1)
    echo "$OUTPUT" >> "$LOG"

    # Check if there are no more memories to process
    if echo "$OUTPUT" | grep -q "No unextracted memories"; then
        echo "=== Backfill complete: $(date) ===" >> "$LOG"
        break
    fi

    BATCH=$((BATCH + 1))

    # Safety: max 15 batches per run (1500 memories)
    if [ $BATCH -gt 15 ]; then
        echo "=== Hit batch limit, stopping: $(date) ===" >> "$LOG"
        break
    fi

    # Pause between batches to avoid rate limits
    sleep 60
done

echo "=== Final stats ===" >> "$LOG"
python3 backfill_graph.py --stats >> "$LOG" 2>&1
