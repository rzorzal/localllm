#!/usr/bin/env bash
# measure_ttft.sh — Cold vs warm TTFT (Time To First Token) measurement
#
# Sends a large (~4-8k token) system prompt + short user message TWICE in
# the same server session. The first run is COLD (full prefill), the second
# is WARM (static prefix already in KV cache, only the tail is decoded).
# Uses max_tokens=10 so the total time is dominated by prefill cost.
#
# Usage: ./scripts/measure_ttft.sh
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"
MODEL="${MODEL:-qwen2.5-7b-instruct}"

echo "=== TTFT Measurement: Cold vs Warm KV-cache prefix reuse ==="
echo "Server: $BASE_URL"
echo ""

# Check server is up
if ! curl -sf "${BASE_URL}/health" > /dev/null 2>&1; then
  echo "ERROR: Server not responding at ${BASE_URL}. Start localllm first."
  exit 1
fi

# ---------------------------------------------------------------------------
# Build a large system prompt (~4k tokens) to simulate a Claude Code prompt.
# We repeat a realistic tool-schema block many times to reach the token count.
# ---------------------------------------------------------------------------
TOOL_BLOCK=$(cat <<'EOF'
{"name":"read_file","description":"Read the contents of a file at the given path. Use this when you need to examine source code, configuration files, or any text file. Returns the full file content as a string.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute or relative path to the file to read"},"start_line":{"type":"integer","description":"Optional: first line to read (1-indexed)"},"end_line":{"type":"integer","description":"Optional: last line to read (inclusive)"}},"required":["path"]}}
{"name":"write_file","description":"Write content to a file, creating it if it does not exist or overwriting if it does. Use for creating new files, updating configuration, writing generated code, or saving results. Always show the user what you are writing first.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute or relative path to write"},"content":{"type":"string","description":"Full content to write to the file"}},"required":["path","content"]}}
{"name":"bash","description":"Execute a shell command in the project directory. Use for running tests, builds, git operations, and inspecting the filesystem. Avoid long-running interactive commands. Always quote paths with spaces.","parameters":{"type":"object","properties":{"command":{"type":"string","description":"The shell command to execute"},"timeout_ms":{"type":"integer","description":"Optional timeout in milliseconds (default 120000)"}},"required":["command"]}}
{"name":"search_files","description":"Search for a pattern across files in the repository using grep or ripgrep. Returns matching lines with file paths and line numbers. Use to find symbol definitions, usages, configuration keys, or any textual pattern.","parameters":{"type":"object","properties":{"pattern":{"type":"string","description":"The regex or literal pattern to search for"},"path":{"type":"string","description":"Optional: directory or glob to restrict search"},"case_sensitive":{"type":"boolean","description":"Whether the search is case-sensitive (default true)"}},"required":["pattern"]}}
{"name":"list_directory","description":"List the files and subdirectories in a given directory. Use to explore project structure, find relevant files, and understand the layout before diving into specific files.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path to list"},"recursive":{"type":"boolean","description":"Whether to recurse into subdirectories (default false)"}},"required":["path"]}}
{"name":"get_diagnostics","description":"Retrieve compiler errors, linter warnings, and type-checking diagnostics for the current workspace. Use after making changes to verify correctness before committing.","parameters":{"type":"object","properties":{"severity":{"type":"string","enum":["error","warning","info","hint"],"description":"Minimum severity level to return"}},"required":[]}}
{"name":"edit_file","description":"Apply a targeted edit to an existing file, replacing a specific string with a new one. More efficient than write_file for small changes. The old_string must match exactly (including whitespace) and must be unique in the file.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to file to edit"},"old_string":{"type":"string","description":"Exact text to replace"},"new_string":{"type":"string","description":"Replacement text"}},"required":["path","old_string","new_string"]}}
{"name":"create_directory","description":"Create a new directory (and any missing parent directories). Use before writing files to paths that do not yet exist.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Directory path to create"}},"required":["path"]}}
{"name":"move_file","description":"Move or rename a file or directory. Use for reorganising the codebase, renaming symbols at the file level, or archiving files.","parameters":{"type":"object","properties":{"source":{"type":"string","description":"Current path of the file or directory"},"destination":{"type":"string","description":"Target path"}},"required":["source","destination"]}}
{"name":"delete_file","description":"Permanently delete a file or directory. Use with caution. Always confirm with the user before deleting important files.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Path to delete"},"recursive":{"type":"boolean","description":"If true, delete directories and contents recursively"}},"required":["path"]}}
EOF
)

# Repeat ~5 times to get ~2-3k tokens in the system prompt
# (keeping under 8192 context limit with room for the user message and output)
TOOLS_REPEATED=""
for i in $(seq 1 5); do
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

# Build the JSON payload using python3 to handle escaping reliably
PAYLOAD=$(python3 -c "
import json, sys

system = sys.argv[1]
payload = {
    'model': 'qwen2.5-7b-instruct',
    'messages': [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': 'What is 2+2? Reply in one word.'}
    ],
    'max_tokens': 10,
    'temperature': 0.0
}
print(json.dumps(payload))
" "$SYSTEM_PROMPT")

echo "System prompt length: $(echo "$SYSTEM_PROMPT" | wc -c | tr -d ' ') chars"
echo "max_tokens=10 (prefill-dominated measurement)"
echo ""

# ---------------------------------------------------------------------------
# Run cold request (first — empty KV cache)
# ---------------------------------------------------------------------------
echo "--- Run 1: COLD (empty KV cache) ---"
COLD_RESULT=$(python3 -c "
import urllib.request, json, time, sys

url = '${BASE_URL}/v1/chat/completions'
payload = sys.stdin.read().encode()
req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})

t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as resp:
    body = resp.read()
t1 = time.perf_counter()

data = json.loads(body)
elapsed = t1 - t0
completion_tokens = data.get('usage', {}).get('completion_tokens', 0)
prompt_tokens = data.get('usage', {}).get('prompt_tokens', 0)
content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
tps = completion_tokens / elapsed if elapsed > 0 else 0

print(f'elapsed={elapsed:.3f}')
print(f'prompt_tokens={prompt_tokens}')
print(f'completion_tokens={completion_tokens}')
print(f'tps={tps:.2f}')
print(f'content={content[:80]}')
" <<< "$PAYLOAD")

COLD_ELAPSED=$(echo "$COLD_RESULT" | grep '^elapsed=' | cut -d= -f2)
COLD_PROMPT=$(echo "$COLD_RESULT" | grep '^prompt_tokens=' | cut -d= -f2)
COLD_COMP=$(echo "$COLD_RESULT" | grep '^completion_tokens=' | cut -d= -f2)
COLD_TPS=$(echo "$COLD_RESULT" | grep '^tps=' | cut -d= -f2)
COLD_CONTENT=$(echo "$COLD_RESULT" | grep '^content=' | cut -d= -f2-)

echo "  Elapsed:           ${COLD_ELAPSED}s"
echo "  Prompt tokens:     ${COLD_PROMPT}"
echo "  Completion tokens: ${COLD_COMP}"
echo "  Tokens/sec:        ${COLD_TPS}"
echo "  Response:          ${COLD_CONTENT}"
echo ""

# ---------------------------------------------------------------------------
# Run warm request (second — prefix should be cached)
# Same prompt — the static prefix is identical, only user message differs.
# Use a slightly different user message to exercise the tail-only prefill path.
# ---------------------------------------------------------------------------
PAYLOAD_WARM=$(python3 -c "
import json, sys

system = sys.argv[1]
payload = {
    'model': 'qwen2.5-7b-instruct',
    'messages': [
        {'role': 'system', 'content': system},
        {'role': 'user', 'content': 'What is 3+3? Reply in one word.'}
    ],
    'max_tokens': 10,
    'temperature': 0.0
}
print(json.dumps(payload))
" "$SYSTEM_PROMPT")

echo "--- Run 2: WARM (same system prompt prefix in KV cache) ---"
WARM_RESULT=$(python3 -c "
import urllib.request, json, time, sys

url = '${BASE_URL}/v1/chat/completions'
payload = sys.stdin.read().encode()
req = urllib.request.Request(url, data=payload, headers={'Content-Type': 'application/json'})

t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=600) as resp:
    body = resp.read()
t1 = time.perf_counter()

data = json.loads(body)
elapsed = t1 - t0
completion_tokens = data.get('usage', {}).get('completion_tokens', 0)
prompt_tokens = data.get('usage', {}).get('prompt_tokens', 0)
content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
tps = completion_tokens / elapsed if elapsed > 0 else 0

print(f'elapsed={elapsed:.3f}')
print(f'prompt_tokens={prompt_tokens}')
print(f'completion_tokens={completion_tokens}')
print(f'tps={tps:.2f}')
print(f'content={content[:80]}')
" <<< "$PAYLOAD_WARM")

WARM_ELAPSED=$(echo "$WARM_RESULT" | grep '^elapsed=' | cut -d= -f2)
WARM_PROMPT=$(echo "$WARM_RESULT" | grep '^prompt_tokens=' | cut -d= -f2)
WARM_COMP=$(echo "$WARM_RESULT" | grep '^completion_tokens=' | cut -d= -f2)
WARM_TPS=$(echo "$WARM_RESULT" | grep '^tps=' | cut -d= -f2)
WARM_CONTENT=$(echo "$WARM_RESULT" | grep '^content=' | cut -d= -f2-)

echo "  Elapsed:           ${WARM_ELAPSED}s"
echo "  Prompt tokens:     ${WARM_PROMPT}"
echo "  Completion tokens: ${WARM_COMP}"
echo "  Tokens/sec:        ${WARM_TPS}"
echo "  Response:          ${WARM_CONTENT}"
echo ""

# ---------------------------------------------------------------------------
# Compute speedup
# ---------------------------------------------------------------------------
SPEEDUP=$(python3 -c "
cold = float('${COLD_ELAPSED}')
warm = float('${WARM_ELAPSED}')
if warm > 0:
    print(f'{cold/warm:.2f}x')
else:
    print('N/A')
")

echo "=== Summary ==="
echo "  COLD time:  ${COLD_ELAPSED}s  (full prefill of ${COLD_PROMPT} tokens)"
echo "  WARM time:  ${WARM_ELAPSED}s  (only tail re-prefilled; prefix cached)"
echo "  Speedup:    ${SPEEDUP}"
echo ""
echo "  Cold answer: ${COLD_CONTENT}"
echo "  Warm answer: ${WARM_CONTENT}"
echo ""
echo "Check server logs for 'prefix reuse: ...' lines to confirm KV reuse."
