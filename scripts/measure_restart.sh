#!/usr/bin/env bash
# measure_restart.sh — Cold-start vs warm-after-restart KV-cache persistence benchmark.
#
# Phase 4 Task 2 verification:
#   Run A: Start server fresh (delete kv-cache-dir), send a large-prefix request,
#          measure time. Server saves prefix KV to disk. Kill server.
#   Run B: Restart server (should warm-start from disk), send SAME large-prefix
#          request, measure time. Must be faster AND log "warm-started from disk".
#
# Usage:
#   ./scripts/measure_restart.sh [binary] [port]
#
#   binary  — path to localllm binary (default: ./target/debug/localllm)
#   port    — port to use (default: 18099, avoids clashing with a running server)
set -euo pipefail

BINARY="${1:-./target/debug/localllm}"
PORT="${2:-18099}"
BASE_URL="http://localhost:${PORT}"

# KV cache dir dedicated to this benchmark (cleaned before run A)
KV_CACHE_DIR="/tmp/localllm_measure_restart_kvcache"

SERVER_LOG_A="/tmp/localllm_runA.log"
SERVER_LOG_B="/tmp/localllm_runB.log"

echo "=== measure_restart.sh: KV-cache disk persistence benchmark ==="
echo "Binary:      ${BINARY}"
echo "Port:        ${PORT}"
echo "KV dir:      ${KV_CACHE_DIR}"
echo ""

if [[ ! -f "$BINARY" ]]; then
  echo "ERROR: binary not found: $BINARY"
  echo "Run 'cargo build' first."
  exit 1
fi

# ---------------------------------------------------------------------------
# Build a large system prompt (~4-8k tokens) to simulate a Claude Code prefix.
# Repeated realistic tool schemas to hit the KV_PERSIST_MIN_TOKENS=1024 threshold.
# ---------------------------------------------------------------------------
TOOL_BLOCK=$(cat <<'EOF'
{"name":"read_file","description":"Read the contents of a file at the given path. Use this when you need to examine source code, configuration files, or any text file. Returns the full file content as a string.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute or relative path to the file to read"},"start_line":{"type":"integer","description":"Optional: first line to read (1-indexed)"},"end_line":{"type":"integer","description":"Optional: last line to read (inclusive)"}},"required":["path"]}}
{"name":"write_file","description":"Write content to a file, creating it if it does not exist or overwriting if it does. Use for creating new files, updating configuration, writing generated code, or saving results. Always show the user what you are writing first.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute or relative path to write"},"content":{"type":"string","description":"Full content to write to the file"}},"required":["path","content"]}}
{"name":"bash","description":"Execute a shell command in the project directory. Use for running tests, builds, git operations, and inspecting the filesystem. Avoid long-running interactive commands. Always quote paths with spaces.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute"},"timeout_ms":{"type":"integer","description":"Optional timeout in milliseconds (default 120000)"}},"required":["command"]}}
{"name":"search_files","description":"Search for a pattern across files in the repository using grep or ripgrep. Returns matching lines with file paths and line numbers. Use to find symbol definitions, usages, configuration keys, or any textual pattern.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"The regex or literal pattern to search for"},"path":{"type":"string","description":"Optional: directory or glob to restrict search"},"case_sensitive":{"type":"boolean","description":"Whether the search is case-sensitive (default true)"}},"required":["pattern"]}}
{"name":"list_directory","description":"List the files and subdirectories in a given directory. Use to explore project structure, find relevant files, and understand the layout before diving into specific files.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path to list"},"recursive":{"type":"boolean","description":"Whether to recurse into subdirectories (default false)"}},"required":["path"]}}
{"name":"get_diagnostics","description":"Retrieve compiler errors, linter warnings, and type-checking diagnostics for the current workspace. Use after making changes to verify correctness before committing.","parameters":{"type":"object","properties":{"severity":{"type":"string","enum":["error","warning","info","hint"],"description":"Minimum severity level to return"}},"required":[]}}
{"name":"edit_file","description":"Apply a targeted edit to an existing file, replacing a specific string with a new one. More efficient than write_file for small targeted changes. The old_string must match exactly (including whitespace) and must be unique in the file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to file to edit"},"old_string":{"type":"string","description":"Exact text to replace"},"new_string":{"type":"string","description":"Replacement text"}},"required":["path","old_string","new_string"]}}
{"name":"create_directory","description":"Create a new directory (and any missing parent directories). Use before writing files to paths that do not yet exist.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path to create"}},"required":["path"]}}
{"name":"move_file","description":"Move or rename a file or directory. Use for reorganising the codebase, renaming symbols at the file level, or archiving files.","parameters":{"type":"object","properties":{"source":{"type":"string","description":"Current path of the file or directory"},"destination":{"type":"string","description":"Target path"}},"required":["source","destination"]}}
{"name":"delete_file","description":"Permanently delete a file or directory. Use with caution. Always confirm with the user before deleting important files.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to delete"},"recursive":{"type":"boolean","description":"If true, delete directories and contents recursively"}},"required":["path"]}}
EOF
)

TOOLS_REPEATED=""
for i in $(seq 1 8); do
  TOOLS_REPEATED="${TOOLS_REPEATED}${TOOL_BLOCK}"
done

SYSTEM_PROMPT="You are Claude Code, an AI coding assistant integrated into the developer's terminal.

You help users understand, write, debug, and improve code across all programming languages and frameworks. You have access to powerful tools to read files, execute shell commands, search code, and make targeted edits.

## Core principles
- Be concise and precise. Developers value accuracy over verbosity.
- Always read relevant code before proposing changes.
- Run tests after making changes to verify correctness.
- Explain what you are doing and why, especially for non-obvious decisions.
- When in doubt, ask a clarifying question rather than guessing.

## Tool usage guidelines
- Use read_file before editing to understand the current state.
- Prefer edit_file over write_file for small targeted changes.
- Use bash to run builds, tests, and verifications after edits.
- Chain tool calls efficiently; do not read the same file twice.
- When searching for a symbol, use search_files with a precise pattern.

## Available tools (JSON schema):
<tools>
${TOOLS_REPEATED}
</tools>

When you call a tool, emit exactly:
<tool_call>{\"name\": \"<function-name>\", \"arguments\": <args-dict>}</tool_call>"

build_payload() {
  local user_msg="$1"
  python3 -c "
import json, sys
system = open('/tmp/measure_restart_sysprompt.txt').read()
user = sys.argv[1]
payload = {
    'model': 'qwen2.5-7b-instruct',
    'messages': [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': user}
    ],
    'max_tokens': 10,
    'temperature': 0.0
}
print(json.dumps(payload))
" "$user_msg"
}

# Write system prompt to file for python3 to read (avoids shell escaping issues)
echo "$SYSTEM_PROMPT" > /tmp/measure_restart_sysprompt.txt

echo "System prompt length: $(wc -c < /tmp/measure_restart_sysprompt.txt | tr -d ' ') chars"
echo ""

# ---------------------------------------------------------------------------
# Helper: wait for server to be up
# ---------------------------------------------------------------------------
wait_for_server() {
  local url="$1"
  local max_wait=120
  local i=0
  while ! curl -sf "${url}/health" > /dev/null 2>&1; do
    sleep 2
    i=$((i+2))
    if [[ $i -ge $max_wait ]]; then
      echo "ERROR: Server did not come up within ${max_wait}s"
      return 1
    fi
  done
}

# ---------------------------------------------------------------------------
# Helper: send request and time it
# ---------------------------------------------------------------------------
timed_request() {
  local user_msg="$1"
  build_payload "$user_msg" > /tmp/measure_restart_payload.json
  python3 -c "
import urllib.request, json, time

url = '${BASE_URL}/v1/chat/completions'
with open('/tmp/measure_restart_payload.json') as f:
    payload = f.read().encode()
req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})

t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as resp:
    body = resp.read()
t1 = time.perf_counter()

data = json.loads(body)
elapsed = t1 - t0
usage = data.get('usage', {})
content = data.get('choices', [{}])[0].get('message', {}).get('content', '')

print(f'elapsed={elapsed:.3f}')
print(f'prompt_tokens={usage.get(\"prompt_tokens\", 0)}')
print(f'completion_tokens={usage.get(\"completion_tokens\", 0)}')
print(f'content={content[:120]}')
"
}

# ---------------------------------------------------------------------------
# RUN A: Cold start — delete KV cache dir first
# ---------------------------------------------------------------------------
echo "============================================================"
echo "RUN A: Cold start (deleting kv-cache-dir: ${KV_CACHE_DIR})"
echo "============================================================"

rm -rf "$KV_CACHE_DIR"

echo "Starting server (Run A) ..."
RUST_LOG="localllm=info" "$BINARY" \
  --port "$PORT" \
  --kv-cache-dir "$KV_CACHE_DIR" \
  > "$SERVER_LOG_A" 2>&1 &
SERVER_PID_A=$!
echo "Server PID: $SERVER_PID_A"

echo "Waiting for server to be ready ..."
wait_for_server "$BASE_URL"
echo "Server ready."
echo ""

echo "Sending large-prefix request (Run A — cold) ..."
RUN_A_RESULT=$(timed_request "What is 2+2? Reply with just the number.")

A_ELAPSED=$(echo "$RUN_A_RESULT" | grep '^elapsed=' | cut -d= -f2)
A_PROMPT=$(echo "$RUN_A_RESULT" | grep '^prompt_tokens=' | cut -d= -f2)
A_COMP=$(echo "$RUN_A_RESULT" | grep '^completion_tokens=' | cut -d= -f2)
A_CONTENT=$(echo "$RUN_A_RESULT" | grep '^content=' | cut -d= -f2-)

echo "  Elapsed:       ${A_ELAPSED}s"
echo "  Prompt tokens: ${A_PROMPT}"
echo "  Completion:    ${A_COMP} tokens"
echo "  Answer:        ${A_CONTENT}"
echo ""

echo "Waiting 3s for KV state to be flushed to disk ..."
sleep 3

echo "KV cache dir contents after Run A:"
ls -lh "$KV_CACHE_DIR" 2>/dev/null || echo "  (empty — save may not have triggered)"
echo ""

echo "Killing server (Run A, PID $SERVER_PID_A) ..."
kill "$SERVER_PID_A" 2>/dev/null || true
wait "$SERVER_PID_A" 2>/dev/null || true
echo "Server A stopped."
echo ""

# Check KV state was saved
KV_FILE_COUNT=$(ls "$KV_CACHE_DIR"/*.kvstate 2>/dev/null | wc -l | tr -d ' ')
if [[ "$KV_FILE_COUNT" -eq 0 ]]; then
  echo "WARNING: No .kvstate files found in $KV_CACHE_DIR — warm-start will not be triggered."
  echo "         Check server log A for errors: $SERVER_LOG_A"
  echo "         Relevant log lines:"
  grep -i "kv\|prefix\|persist\|save\|warm" "$SERVER_LOG_A" || echo "  (none)"
else
  echo "KV state saved: $KV_FILE_COUNT file(s) in $KV_CACHE_DIR"
  echo "  $(ls -lh $KV_CACHE_DIR/*.kvstate)"
fi
echo ""

# ---------------------------------------------------------------------------
# RUN B: Restart — server should warm-start from disk
# ---------------------------------------------------------------------------
echo "============================================================"
echo "RUN B: Restart — warm-start from disk KV state"
echo "============================================================"

echo "Starting server (Run B) ..."
RUST_LOG="localllm=info" "$BINARY" \
  --port "$PORT" \
  --kv-cache-dir "$KV_CACHE_DIR" \
  > "$SERVER_LOG_B" 2>&1 &
SERVER_PID_B=$!
echo "Server PID: $SERVER_PID_B"

echo "Waiting for server to be ready ..."
wait_for_server "$BASE_URL"
echo "Server ready."
echo ""

# Give 1 extra second for warm-start log to flush
sleep 1

echo "Checking server log B for warm-start confirmation ..."
WARMSTART_LINE=$(grep -m1 "warm-started from disk" "$SERVER_LOG_B" 2>/dev/null || true)
if [[ -n "$WARMSTART_LINE" ]]; then
  echo "  CONFIRMED: $WARMSTART_LINE"
else
  echo "  WARNING: 'warm-started from disk' not found in log B — warm-start may not have triggered."
  echo "  Relevant log lines:"
  grep -i "kv\|prefix\|persist\|warm\|disk\|load" "$SERVER_LOG_B" | head -20 || echo "  (none)"
fi
echo ""

echo "Sending SAME large-prefix request (Run B — warm-after-restart) ..."
RUN_B_RESULT=$(timed_request "What is 2+2? Reply with just the number.")

B_ELAPSED=$(echo "$RUN_B_RESULT" | grep '^elapsed=' | cut -d= -f2)
B_PROMPT=$(echo "$RUN_B_RESULT" | grep '^prompt_tokens=' | cut -d= -f2)
B_COMP=$(echo "$RUN_B_RESULT" | grep '^completion_tokens=' | cut -d= -f2)
B_CONTENT=$(echo "$RUN_B_RESULT" | grep '^content=' | cut -d= -f2-)

echo "  Elapsed:       ${B_ELAPSED}s"
echo "  Prompt tokens: ${B_PROMPT}"
echo "  Completion:    ${B_COMP} tokens"
echo "  Answer:        ${B_CONTENT}"
echo ""

echo "Killing server (Run B, PID $SERVER_PID_B) ..."
kill "$SERVER_PID_B" 2>/dev/null || true
wait "$SERVER_PID_B" 2>/dev/null || true
echo "Server B stopped."
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
SPEEDUP=$(python3 -c "
cold = float('${A_ELAPSED}')
warm = float('${B_ELAPSED}')
if warm > 0:
    print(f'{cold/warm:.2f}x')
else:
    print('N/A')
")

echo "============================================================"
echo "RESULTS SUMMARY"
echo "============================================================"
echo ""
echo "  Run A (cold start):          ${A_ELAPSED}s  (${A_PROMPT} prompt tokens)"
echo "  Run B (warm-after-restart):  ${B_ELAPSED}s  (${B_PROMPT} prompt tokens)"
echo "  Speedup:                     ${SPEEDUP}"
echo ""
if [[ -n "$WARMSTART_LINE" ]]; then
  echo "  Warm-start confirmed: YES"
  echo "  Log line: $WARMSTART_LINE"
else
  echo "  Warm-start confirmed: NO (check logs)"
fi
echo ""
echo "  Run A answer: ${A_CONTENT}"
echo "  Run B answer: ${B_CONTENT}"
echo ""
echo "  Server A log: $SERVER_LOG_A"
echo "  Server B log: $SERVER_LOG_B"
