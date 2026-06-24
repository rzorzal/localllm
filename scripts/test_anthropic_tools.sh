#!/usr/bin/env bash
# test_anthropic_tools.sh — Two-step Anthropic tool-calling acceptance test
# Tests: POST /v1/messages with get_weather tool, verify tool_use response,
#        then send tool result back and verify final answer mentions weather.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
MODEL="${MODEL:-qwen2.5-7b-instruct}"
PASS=0
FAIL=0

pass() { echo "PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== Anthropic Tool-Calling Acceptance Test ==="
echo "Server: $BASE_URL"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Send initial request with get_weather tool
# ---------------------------------------------------------------------------
echo "--- Step 1: Initial request with get_weather tool ---"

STEP1_BODY=$(cat <<'EOF'
{
  "model": "qwen2.5-7b-instruct",
  "max_tokens": 512,
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Recife? Use the get_weather tool."
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather for a given location.",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city or location to get weather for."
          }
        },
        "required": ["location"]
      }
    }
  ]
}
EOF
)

echo "Sending step 1 request (may take 30s-3min on CPU)..."
STEP1_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/messages" \
  -H "Content-Type: application/json" \
  -d "$STEP1_BODY" \
  --max-time 300)

echo "Step 1 response received."
echo "Raw response:"
echo "$STEP1_RESPONSE" | jq .

# Assert stop_reason == "tool_use"
STOP_REASON=$(echo "$STEP1_RESPONSE" | jq -r '.stop_reason')
if [ "$STOP_REASON" = "tool_use" ]; then
  pass "Step 1: stop_reason == 'tool_use'"
else
  fail "Step 1: stop_reason expected 'tool_use', got '$STOP_REASON'"
fi

# Extract tool_use block (first content block with type == "tool_use")
TOOL_USE_BLOCK=$(echo "$STEP1_RESPONSE" | jq '.content[] | select(.type == "tool_use") | {id, name, input}' | head -1)
if [ -z "$TOOL_USE_BLOCK" ] || [ "$TOOL_USE_BLOCK" = "null" ]; then
  # Try selecting first block regardless
  TOOL_USE_BLOCK=$(echo "$STEP1_RESPONSE" | jq '.content[0]')
fi

TOOL_USE_ID=$(echo "$STEP1_RESPONSE" | jq -r '[.content[] | select(.type == "tool_use")][0].id // empty')
TOOL_USE_NAME=$(echo "$STEP1_RESPONSE" | jq -r '[.content[] | select(.type == "tool_use")][0].name // empty')
TOOL_USE_INPUT=$(echo "$STEP1_RESPONSE" | jq -r '[.content[] | select(.type == "tool_use")][0].input // "{}"')

if [ -n "$TOOL_USE_ID" ]; then
  pass "Step 1: tool_use id present: $TOOL_USE_ID"
else
  fail "Step 1: tool_use id missing"
  TOOL_USE_ID="toolu_missing"
fi

if [ "$TOOL_USE_NAME" = "get_weather" ]; then
  pass "Step 1: tool name == 'get_weather'"
else
  fail "Step 1: tool name expected 'get_weather', got '$TOOL_USE_NAME'"
fi

echo ""
echo "Tool use id: $TOOL_USE_ID"
echo "Tool input: $TOOL_USE_INPUT"

# Capture the full assistant content blocks for replay
ASSISTANT_CONTENT=$(echo "$STEP1_RESPONSE" | jq '.content')

# ---------------------------------------------------------------------------
# Step 2: Send follow-up with tool result
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Follow-up with tool result ---"

# Build step 2: replay user text, assistant tool_use content, then user tool_result
STEP2_BODY=$(cat <<EOF
{
  "model": "qwen2.5-7b-instruct",
  "max_tokens": 512,
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Recife? Use the get_weather tool."
    },
    {
      "role": "assistant",
      "content": ${ASSISTANT_CONTENT}
    },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "${TOOL_USE_ID}",
          "content": "28°C, sunny"
        }
      ]
    }
  ],
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather for a given location.",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city or location to get weather for."
          }
        },
        "required": ["location"]
      }
    }
  ]
}
EOF
)

echo "Sending step 2 request (may take 30s-3min on CPU)..."
STEP2_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/messages" \
  -H "Content-Type: application/json" \
  -d "$STEP2_BODY" \
  --max-time 300)

echo "Step 2 response received."
echo "Raw response:"
echo "$STEP2_RESPONSE" | jq .

# Assert final text is non-empty
FINAL_TEXT=$(echo "$STEP2_RESPONSE" | jq -r '[.content[] | select(.type == "text")][0].text // ""')
if [ -n "$FINAL_TEXT" ] && [ "$FINAL_TEXT" != "null" ]; then
  pass "Step 2: final text content is non-empty"
else
  fail "Step 2: final text content is empty or null"
fi

# Assert mentions weather (lenient, case-insensitive)
CONTENT_LOWER=$(echo "$FINAL_TEXT" | tr '[:upper:]' '[:lower:]')
if echo "$CONTENT_LOWER" | grep -qiE '28|sunny|weather|warm|celsius'; then
  pass "Step 2: final answer mentions weather details"
else
  fail "Step 2: final answer does not mention '28', 'sunny', 'weather', 'warm', or 'celsius'"
fi

echo ""
echo "Final answer from model:"
echo "$FINAL_TEXT"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "=== Results ==="
echo "PASSED: $PASS"
echo "FAILED: $FAIL"
echo ""
if [ "$FAIL" -eq 0 ]; then
  echo "ALL TESTS PASSED"
  exit 0
else
  echo "SOME TESTS FAILED"
  exit 1
fi
