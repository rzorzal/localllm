#!/usr/bin/env bash
# measure_prefix.sh — Measure Tscg tool-block size reduction
#
# Compares the prompt_tokens returned by the server when using a multi-tool
# request vs the estimated old verbose-JSON prompt size.
#
# Usage: bash scripts/measure_prefix.sh
# Requires: server running on localhost:8080 with RUST_LOG=debug (for char counts)
#
# The script sends a 3-tool request and captures prompt_tokens from the response.
# It also computes the character-level reduction offline for the tools block.

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8080}"

echo "=== Tscg Prefix-Size Measurement ==="
echo "Server: $BASE_URL"
echo ""

# ---------------------------------------------------------------------------
# Multi-tool request body (3 tools — representative multi-tool load)
# ---------------------------------------------------------------------------
REQUEST_BODY=$(cat <<'EOF'
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
            },
            "unit": {
              "type": "string",
              "enum": ["celsius", "fahrenheit"],
              "description": "Temperature unit"
            }
          },
          "required": ["location"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web for information.",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {
              "type": "string",
              "description": "The search query string"
            },
            "num_results": {
              "type": "integer",
              "description": "Number of results to return (default 10)"
            },
            "region": {
              "type": "string",
              "enum": ["us", "uk", "eu", "au"],
              "description": "Geographic region filter"
            }
          },
          "required": ["query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "read_file",
        "description": "Read the contents of a file from disk.",
        "parameters": {
          "type": "object",
          "properties": {
            "path": {
              "type": "string",
              "description": "Absolute or relative path to the file"
            },
            "encoding": {
              "type": "string",
              "enum": ["utf-8", "latin-1", "base64"],
              "description": "File encoding"
            },
            "max_bytes": {
              "type": "integer",
              "description": "Maximum bytes to read (0 = unlimited)"
            }
          },
          "required": ["path"]
        }
      }
    }
  ],
  "max_tokens": 64,
  "stream": false
}
EOF
)

# ---------------------------------------------------------------------------
# Compute offline char-level reduction for the tools block
# ---------------------------------------------------------------------------
# Verbose JSON block (what the old code did: one JSON object per tool, newline-separated)
VERBOSE_BLOCK='{"name":"get_weather","description":"Get the current weather for a given location.","parameters":{"type":"object","properties":{"location":{"type":"string","description":"The city or location to get weather for."},"unit":{"type":"string","enum":["celsius","fahrenheit"],"description":"Temperature unit"}},"required":["location"]}}
{"name":"search_web","description":"Search the web for information.","parameters":{"type":"object","properties":{"query":{"type":"string","description":"The search query string"},"num_results":{"type":"integer","description":"Number of results to return (default 10)"},"region":{"type":"string","enum":["us","uk","eu","au"],"description":"Geographic region filter"}},"required":["query"]}}
{"name":"read_file","description":"Read the contents of a file from disk.","parameters":{"type":"object","properties":{"path":{"type":"string","description":"Absolute or relative path to the file"},"encoding":{"type":"string","enum":["utf-8","latin-1","base64"],"description":"File encoding"},"max_bytes":{"type":"integer","description":"Maximum bytes to read (0 = unlimited)"}},"required":["path"]}}'

# Compact block (Tscg output)
COMPACT_BLOCK='get_weather(location:string! (The city or location to get weather for.), unit:string=enum[celsius,fahrenheit] (Temperature unit)) — Get the current weather for a given location.
search_web(query:string! (The search query string), num_results:integer (Number of results to return (default 10)), region:string=enum[us,uk,eu,au] (Geographic region filter)) — Search the web for information.
read_file(path:string! (Absolute or relative path to the file), encoding:string=enum[utf-8,latin-1,base64] (File encoding), max_bytes:integer (Maximum bytes to read (0 = unlimited))) — Read the contents of a file from disk.'

VERBOSE_LEN=${#VERBOSE_BLOCK}
COMPACT_LEN=${#COMPACT_BLOCK}

echo "--- Offline character-level tool-block analysis (3 tools) ---"
echo "Verbose JSON block: $VERBOSE_LEN chars"
echo "Compact Tscg block: $COMPACT_LEN chars"
RATIO=$(echo "scale=1; $COMPACT_LEN * 100 / $VERBOSE_LEN" | bc)
REDUCTION=$(echo "scale=1; 100 - $COMPACT_LEN * 100 / $VERBOSE_LEN" | bc)
echo "Ratio: ${RATIO}%  |  Reduction: ${REDUCTION}%"
echo ""

# ---------------------------------------------------------------------------
# Live server: get prompt_tokens from multi-tool request
# ---------------------------------------------------------------------------
echo "--- Live server prompt_tokens measurement ---"
echo "Sending 3-tool request to $BASE_URL/v1/chat/completions ..."

RESPONSE=$(curl -sf -X POST "${BASE_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "$REQUEST_BODY" \
  --max-time 300)

PROMPT_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.prompt_tokens // "N/A"')
COMPLETION_TOKENS=$(echo "$RESPONSE" | jq -r '.usage.completion_tokens // "N/A"')
FINISH_REASON=$(echo "$RESPONSE" | jq -r '.choices[0].finish_reason // "N/A"')

echo "prompt_tokens:      $PROMPT_TOKENS"
echo "completion_tokens:  $COMPLETION_TOKENS"
echo "finish_reason:      $FINISH_REASON"
echo ""
echo "=== Summary ==="
echo "Tool-block size: ${VERBOSE_LEN} chars (verbose JSON) → ${COMPACT_LEN} chars (Tscg) = ${REDUCTION}% reduction"
echo "Live prompt_tokens from server: $PROMPT_TOKENS"
