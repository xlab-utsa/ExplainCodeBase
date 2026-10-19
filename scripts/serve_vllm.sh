#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server for ExplainCodeBase.
#
#   scripts/serve_vllm.sh [model] [extra vllm serve args...]
#
# Examples:
#   scripts/serve_vllm.sh                                   # openai/gpt-oss-20b
#   scripts/serve_vllm.sh Qwen/Qwen2.5-Coder-32B-Instruct --tensor-parallel-size 2
#   PORT=8001 scripts/serve_vllm.sh meta-llama/Llama-3.1-8B-Instruct --max-model-len 32768
set -euo pipefail

MODEL="${1:-${VLLM_MODEL:-openai/gpt-oss-20b}}"
shift || true
PORT="${PORT:-8000}"

if ! command -v vllm >/dev/null 2>&1; then
  echo "vllm is not installed. Install it on a GPU machine with: pip install vllm" >&2
  exit 1
fi

echo "Serving $MODEL on http://localhost:$PORT/v1"
echo "Point ExplainCodeBase at it with: export VLLM_API_BASE=http://localhost:$PORT/v1"
exec vllm serve "$MODEL" --port "$PORT" "$@"
