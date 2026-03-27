#!/bin/bash
# Refresh Claude Code OAuth token and update OpenClaw agent auth profile
# Reads from Claude Code's credential store, refreshes via platform API,
# and writes the fresh token into OpenClaw's auth-profiles.json
#
# Sources (checked in order):
#   1. /root/.clydecodebot/claude-auth/.credentials.json (SDK session - freshest)
#   2. /root/.claude/.credentials.json (CLI fallback)
#
# Token refresh endpoint: https://platform.claude.com/v1/oauth/token
# Client ID: 9d1c250a-e61b-44d9-88ed-5944d1962f5e (Claude Code)

set -euo pipefail

AGENT_DIR="/root/.openclaw/agents/claude/agent"
AUTH_FILE="$AGENT_DIR/auth-profiles.json"
SDK_CREDS="/root/.clydecodebot/claude-auth/.credentials.json"
CLI_CREDS="/root/.claude/.credentials.json"
CLIENT_ID="9d1c250a-e61b-44d9-88ed-5944d1962f5e"
TOKEN_URL="https://platform.claude.com/v1/oauth/token"
LOG_TAG="[claude-oauth-refresh]"

log() { echo "$(date -Iseconds) $LOG_TAG $*"; }

# Pick the freshest credential source
CREDS_FILE=""
if [[ -f "$SDK_CREDS" ]]; then
    CREDS_FILE="$SDK_CREDS"
elif [[ -f "$CLI_CREDS" ]]; then
    CREDS_FILE="$CLI_CREDS"
else
    log "ERROR: No Claude credentials found"
    exit 1
fi

# Extract current tokens
CURRENT=$(python3 -c "
import json, sys
d = json.load(open('$CREDS_FILE'))
o = d.get('claudeAiOauth', {})
print(json.dumps({
    'access': o.get('accessToken',''),
    'refresh': o.get('refreshToken',''),
    'expires': o.get('expiresAt', 0),
    'sub': o.get('subscriptionType',''),
    'tier': o.get('rateLimitTier','')
}))
")

EXPIRES_AT=$(echo "$CURRENT" | python3 -c "import json,sys; print(json.load(sys.stdin)['expires'])")
NOW_MS=$(python3 -c "import time; print(int(time.time()*1000))")
REMAINING_S=$(( (EXPIRES_AT - NOW_MS) / 1000 ))

# If token still has >10 min, just update the auth profile with current token
if [[ $REMAINING_S -gt 600 ]]; then
    log "Token still valid ($REMAINING_S sec remaining), syncing to auth profile"
    ACCESS=$(echo "$CURRENT" | python3 -c "import json,sys; print(json.load(sys.stdin)['access'])")
else
    # Need to refresh
    log "Token expired or expiring soon ($REMAINING_S sec), refreshing..."
    REFRESH_TOKEN=$(echo "$CURRENT" | python3 -c "import json,sys; print(json.load(sys.stdin)['refresh'])")

    RESP=$(curl -s -X POST "$TOKEN_URL" \
        -H "Content-Type: application/x-www-form-urlencoded" \
        -d "grant_type=refresh_token&refresh_token=$REFRESH_TOKEN&client_id=$CLIENT_ID" 2>&1)

    # Check for success
    ACCESS=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('access_token',''))" 2>/dev/null || echo "")

    if [[ -z "$ACCESS" ]]; then
        log "ERROR: Refresh failed: $RESP"
        # Fall back to current token even if expired
        ACCESS=$(echo "$CURRENT" | python3 -c "import json,sys; print(json.load(sys.stdin)['access'])")
        if [[ -z "$ACCESS" ]]; then
            log "ERROR: No usable token"
            exit 1
        fi
        log "Using existing token (may be expired)"
    else
        NEW_REFRESH=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('refresh_token',''))" 2>/dev/null || echo "")
        NEW_EXPIRES=$(echo "$RESP" | python3 -c "import json,sys; d=json.load(sys.stdin); ei=d.get('expires_in',3600); print(int($(python3 -c 'import time;print(int(time.time()*1000))')+ei*1000))" 2>/dev/null || echo "$NOW_MS")

        # Update the credential files with new tokens
        python3 -c "
import json
for f in ['$SDK_CREDS', '$CLI_CREDS']:
    try:
        d = json.load(open(f))
        d['claudeAiOauth']['accessToken'] = '$ACCESS'
        if '$NEW_REFRESH':
            d['claudeAiOauth']['refreshToken'] = '$NEW_REFRESH'
        d['claudeAiOauth']['expiresAt'] = $NEW_EXPIRES
        json.dump(d, open(f, 'w'), indent=2)
    except: pass
"
        log "Token refreshed successfully (expires in $(echo "$RESP" | python3 -c "import json,sys; print(json.load(sys.stdin).get('expires_in','?'))")s)"
    fi
fi

# Write/update auth-profiles.json for the claude agent
mkdir -p "$AGENT_DIR"
python3 -c "
import json, os, time

access = '$ACCESS'
expires = int('$EXPIRES_AT') if $REMAINING_S > 600 else int(time.time()*1000) + 3600000

auth_file = '$AUTH_FILE'
if os.path.exists(auth_file):
    store = json.load(open(auth_file))
else:
    store = {'version': 1, 'profiles': {}, 'lastGood': {}, 'usageStats': {}}

store['profiles']['anthropic:claude-code-oauth'] = {
    'type': 'oauth',
    'provider': 'anthropic',
    'access': access,
    'refresh': '$(echo "$CURRENT" | python3 -c "import json,sys; print(json.load(sys.stdin)[\"refresh\"])")',
    'expires': expires
}

store['lastGood'] = store.get('lastGood', {})
store['lastGood']['anthropic'] = 'anthropic:claude-code-oauth'

json.dump(store, open(auth_file, 'w'), indent=2)
print(f'Updated {auth_file}')
"

# Also update the simple auth.json (used by some code paths)
python3 -c "
import json
auth = {
    'anthropic': {
        'type': 'api_key',
        'key': '$ACCESS'
    }
}
json.dump(auth, open('$AGENT_DIR/auth.json', 'w'), indent=2)
"

log "Auth files updated. Token: ${ACCESS:0:30}..."
