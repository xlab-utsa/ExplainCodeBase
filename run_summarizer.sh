#!/usr/bin/env bash
# Backwards-compatible wrapper:  ./run_summarizer.sh [folder_path] [provider]
#
# Summarizes a codebase (or every folder in input/) with all five methods. The provider
# defaults to vllm; any extra arguments are passed to `explaincodebase summarize`.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

PROVIDERS=(vllm vllm_models openai anthropic gemini mock)
is_provider() { local p; for p in "${PROVIDERS[@]}"; do [[ "$1" == "$p" ]] && return 0; done; return 1; }

ARGS=()
PROVIDER="${ECB_PROVIDER:-vllm}"
for arg in "$@"; do
  if is_provider "$arg"; then PROVIDER="$arg"; else ARGS+=("$arg"); fi
done

PYTHON="${PYTHON:-python3}"
exec "$PYTHON" -m explaincodebase summarize --provider "$PROVIDER" "${ARGS[@]+"${ARGS[@]}"}"
