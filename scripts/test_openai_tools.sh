#!/usr/bin/env bash
# test_openai_tools.sh — Two-step OpenAI tool-calling acceptance test
# Tests: POST /v1/chat/completions with get_weather tool, verify tool_calls response,
#        then send tool result back and verify final answer mentions weather.
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
MODEL="${MODEL:-qwen2.5-7b-instruct}"
PASS=0
FAIL=0

pass() { echo "PASS: $1"; PASS=$((PASS+1)); }
fail() { echo "FAIL: $1"; FAIL=$((FAIL+1)); }

echo "=== OpenAI Tool-Calling Acceptance Test ==="
echo "Server: $BASE_URL"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Send initial request with get_weather tool
# ---------------------------------------------------------------------------
echo "--- Step 1: Initial request with get_weather tool ---"

STEP1_BODY=$(cat <<'EOF'
{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Recife? Use the get_weather tool."
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
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
    }
  ],
  "max_tokens": 256
}
EOF
)

echo "Sending step 1 request (may take 30s-3min on CPU)..."
STEP1_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$STEP1_BODY" \
  --max-time 300)

echo "Step 1 response received."
echo "Raw response:"
echo "$STEP1_RESPONSE" | jq .

# Assert finish_reason == "tool_calls"
FINISH_REASON=$(echo "$STEP1_RESPONSE" | jq -r '.choices[0].finish_reason')
if [ "$FINISH_REASON" = "tool_calls" ]; then
  pass "Step 1: finish_reason == 'tool_calls'"
else
  fail "Step 1: finish_reason expected 'tool_calls', got '$FINISH_REASON'"
fi

# Extract tool call id
TOOL_CALL_ID=$(echo "$STEP1_RESPONSE" | jq -r '.choices[0].message.tool_calls[0].id')
if [ -n "$TOOL_CALL_ID" ] && [ "$TOOL_CALL_ID" != "null" ]; then
  pass "Step 1: tool_call_id present: $TOOL_CALL_ID"
else
  fail "Step 1: tool_call_id missing or null"
  TOOL_CALL_ID="call_missing"
fi

# Extract arguments (should contain location)
TOOL_ARGS=$(echo "$STEP1_RESPONSE" | jq -r '.choices[0].message.tool_calls[0].function.arguments')
TOOL_NAME=$(echo "$STEP1_RESPONSE" | jq -r '.choices[0].message.tool_calls[0].function.name')
if [ "$TOOL_NAME" = "get_weather" ]; then
  pass "Step 1: tool name == 'get_weather'"
else
  fail "Step 1: tool name expected 'get_weather', got '$TOOL_NAME'"
fi

echo ""
echo "Tool call id: $TOOL_CALL_ID"
echo "Tool arguments: $TOOL_ARGS"

# Capture the assistant message for replay
ASSISTANT_MSG=$(echo "$STEP1_RESPONSE" | jq '.choices[0].message')

# ---------------------------------------------------------------------------
# Step 2: Send follow-up with tool result
# ---------------------------------------------------------------------------
echo ""
echo "--- Step 2: Follow-up with tool result ---"

# Build step 2 messages array: original user, assistant with tool_calls, tool result
STEP2_BODY=$(cat <<EOF
{
  "model": "qwen2.5-7b-instruct",
  "messages": [
    {
      "role": "user",
      "content": "What's the weather in Recife? Use the get_weather tool."
    },
    ${ASSISTANT_MSG},
    {
      "role": "tool",
      "tool_call_id": "${TOOL_CALL_ID}",
      "content": "28°C, sunny"
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "parameters": {
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
    }
  ],
  "max_tokens": 256
}
EOF
)

echo "Sending step 2 request (may take 30s-3min on CPU)..."
STEP2_RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$STEP2_BODY" \
  --max-time 300)

echo "Step 2 response received."
echo "Raw response:"
echo "$STEP2_RESPONSE" | jq .

# Assert final content is non-empty
FINAL_CONTENT=$(echo "$STEP2_RESPONSE" | jq -r '.choices[0].message.content // ""')
if [ -n "$FINAL_CONTENT" ] && [ "$FINAL_CONTENT" != "null" ]; then
  pass "Step 2: final content is non-empty"
else
  fail "Step 2: final content is empty or null"
fi

# Assert final content mentions weather (lenient: 28 or sunny, case-insensitive)
CONTENT_LOWER=$(echo "$FINAL_CONTENT" | tr '[:upper:]' '[:lower:]')
if echo "$CONTENT_LOWER" | grep -qiE '28|sunny|weather|warm|celsius'; then
  pass "Step 2: final answer mentions weather details"
else
  fail "Step 2: final answer does not mention '28', 'sunny', 'weather', 'warm', or 'celsius'"
fi

echo ""
echo "Final answer from model:"
echo "$FINAL_CONTENT"

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
