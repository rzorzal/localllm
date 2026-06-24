#!/usr/bin/env bash
# measure.sh — Measure server RAM usage and inference throughput
# Sends one fixed non-streaming request, times it, and computes tokens/sec.
# Usage: measure.sh [server_pid]
#   If server_pid is provided, sample RAM from that process.
#   Otherwise, attempt to find the localllm process via pgrep.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
SERVER_PID="${1:-}"

echo "=== localllm Performance Measurement ==="
echo "Server: $BASE_URL"

# ---------------------------------------------------------------------------
# Find server PID for RAM measurement
# ---------------------------------------------------------------------------
if [ -z "$SERVER_PID" ]; then
  SERVER_PID=$(pgrep -f "localllm" 2>/dev/null | head -1 || true)
fi

if [ -n "$SERVER_PID" ]; then
  echo "Monitoring PID: $SERVER_PID"
else
  echo "Warning: could not determine server PID; RAM measurement will be skipped"
fi

# ---------------------------------------------------------------------------
# Sample RAM before request
# ---------------------------------------------------------------------------
sample_rss_kb() {
  local pid="$1"
  if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
    ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || echo "0"
  else
    echo "0"
  fi
}

RSS_BEFORE=$(sample_rss_kb "$SERVER_PID")
echo "RSS before request: $((RSS_BEFORE / 1024)) MB ($RSS_BEFORE KB)"

# ---------------------------------------------------------------------------
# Fixed prompt
# ---------------------------------------------------------------------------
REQUEST_BODY=$(cat <<'EOF'
{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "In exactly two sentences, explain what a transformer neural network is."
    }
  ],
  "max_tokens": 80,
  "stream": false
}
EOF
)

# ---------------------------------------------------------------------------
# Timed request
# ---------------------------------------------------------------------------
echo ""
echo "Sending fixed prompt (non-streaming)..."
echo "Prompt: 'In exactly two sentences, explain what a transformer neural network is.'"
echo "(max_tokens=80)"
echo ""

START_TIME=$(date +%s)  # seconds (macOS date doesn't support %3N)

RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY" \
  --max-time 300)

END_TIME=$(date +%s)

# ---------------------------------------------------------------------------
# Sample RAM during/after request
# ---------------------------------------------------------------------------
RSS_AFTER=$(sample_rss_kb "$SERVER_PID")

# ---------------------------------------------------------------------------
# Parse response
# ---------------------------------------------------------------------------
COMPLETION_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens // 0')
PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens // 0')
TOTAL_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.total_tokens // 0')
CONTENT=$(echo "$RESPONSE" | jq -r '.choices[0].message.content // ""')

# ---------------------------------------------------------------------------
# Compute elapsed and tokens/sec
# ---------------------------------------------------------------------------
ELAPSED_SEC=$((END_TIME - START_TIME))

# tokens/sec: completion_tokens / elapsed_seconds
if [ "$ELAPSED_SEC" -gt 0 ] && [ "$COMPLETION_TOKENS" -gt 0 ]; then
  TOKENS_PER_SEC=$(echo "scale=2; $COMPLETION_TOKENS / $ELAPSED_SEC" | bc)
else
  TOKENS_PER_SEC="0 (elapsed ${ELAPSED_SEC}s)"
fi

# Peak RSS
PEAK_RSS_KB=$(( RSS_AFTER > RSS_BEFORE ? RSS_AFTER : RSS_BEFORE ))
PEAK_RSS_MB=$(( PEAK_RSS_KB / 1024 ))

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
echo "=== Measurement Results ==="
echo "Model response: $CONTENT"
echo ""
echo "--- Performance Metrics ---"
echo "Prompt tokens:      $PROMPT_TOKENS"
echo "Completion tokens:  $COMPLETION_TOKENS"
echo "Total tokens:       $TOTAL_TOKENS"
echo "Elapsed time:       ${ELAPSED_SEC}s"
echo "Tokens/sec:         ${TOKENS_PER_SEC} tok/s"
echo ""
echo "--- Memory ---"
echo "RSS before:         $((RSS_BEFORE / 1024)) MB"
echo "RSS after:          $((RSS_AFTER / 1024)) MB"
echo "Peak RSS:           ${PEAK_RSS_MB} MB"
echo ""
echo "=== Summary ==="
echo "Peak RAM: ${PEAK_RSS_MB} MB | completion_tokens: ${COMPLETION_TOKENS} | elapsed: ${ELAPSED_SEC}s | throughput: ${TOKENS_PER_SEC} tok/s"

# Also print server log tail for info
echo ""
echo "--- Server log tail ---"
tail -5 /tmp/localllm-server2.log 2>/dev/null || true
