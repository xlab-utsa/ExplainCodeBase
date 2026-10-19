# ExplainCodeBase

Hierarchical, call-graph-aware summarization of every function in a codebase, written in
**any programming language**, using **open models served with [vLLM](https://docs.vllm.ai)**.

The tool parses a codebase with tree-sitter, builds a call graph, and asks an LLM to
summarize each function *after* the functions it calls, so every summary can build on
what its callees do. Four such hierarchical methods are compared against a baseline that
reads the whole codebase at once, using an LLM judge and multiple-choice comprehension
questions.

## Quick start

```bash
pip install -r requirements.txt            # or: pip install -e ".[dev]"

# 1. Serve an open model (on a GPU machine)
pip install vllm
scripts/serve_vllm.sh Qwen/Qwen2.5-Coder-32B-Instruct      # default: openai/gpt-oss-20b

# 2. Point the tool at it (only needed if it is not on localhost:8000)
export VLLM_API_BASE=http://gpu-box:8000/v1

# 3. Summarize, evaluate, quiz
python -m explaincodebase summarize path/to/project       # or put projects in input/
python -m explaincodebase evaluate                        # LLM-as-judge comparison
python -m explaincodebase mcq                             # questions in mcqs/<project>.json
```

vLLM is the default LLM for every command. The model is discovered from the server
(`/v1/models`) unless `VLLM_MODEL` or `--model` names one, and the server's context
length is used to size prompts so they fit. Any OpenAI-compatible server works the same
way (SGLang, Ollama, LM Studio, TGI, ...).

Want a different model for judging than for summarizing (recommended, so the judge does
not grade its own work)? Serve it on another port and pass `--base-url`/`--model`:

```bash
python -m explaincodebase evaluate --base-url http://gpu-box:8001/v1 --model meta-llama/Llama-3.3-70B-Instruct
```

## Languages

```bash
python -m explaincodebase languages
```

| Tier | Languages | What you get |
|---|---|---|
| Detailed | Python, JavaScript, TypeScript/TSX, Java, Go, Rust, C, C++, C#, Ruby, PHP, Kotlin, Scala, Swift, Lua | functions, methods, nested functions, classes and inheritance, constructors, `self`/`this`/`super` calls, receiver type inference |
| Generic | every other tree-sitter grammar that ships a `tags.scm` query (Elixir, Dart, R, OCaml, MATLAB, Solidity, F#, ...) | functions, classes and call references from the grammar's code-navigation query |
| File-level | any other source file (SQL, shell, Haskell, Perl, ...) | the whole file is summarized as one unit |

Data and markup files (Markdown, JSON, YAML, HTML, CSS, ...) are skipped, as are vendored
and build folders (`node_modules`, `venv`, `build`, `target`, ...). Grammars are
downloaded on first use by `tree-sitter-language-pack`.

## Methods

Every method writes one summary per *unit*. A unit is a function or method, the top-level
code of a file (`(global)`), or a whole file (`(file)`) when nothing finer can be parsed.
Unit ids are `<path relative to the project>::<qualified name>`, e.g.
`src/models/dog.py::Dog.bark`.

| Key | Method | Context given with each unit's code |
|---|---|---|
| `baseline` | Baseline | the whole codebase in one prompt; split into parts when it exceeds the model's context |
| `hucp` | Method 1, hierarchical | summaries of the units it calls directly |
| `iucp` | Method 2, inline | the same summaries, written as comments above the exact lines that make the calls |
| `llmucp` | Method 3 | Method 1, plus a line-by-line explanation generated first for leaf units (units that call nothing in the codebase) |
| `docucp` | Method 4 | Method 1, plus documentation: the enclosing class and its ancestors, and docs of library functions it calls (Python built-ins and standard library; third-party packages only with `--allow-imports`, because importing runs their code) |

Units are processed bottom-up over the call graph's strongly connected components, so
each unit is summarized exactly once, after everything it calls; mutually recursive
functions are summarized one after another within their cycle. Independent parts of the
graph are summarized in parallel (`--workers`, default 8; vLLM batches concurrent
requests efficiently).

## Commands

```
python -m explaincodebase [--output-dir output] [-v] <command> ...
```

| Command | Purpose |
|---|---|
| `summarize [paths...]` | Run the methods (default: all five) on each path, or on every folder in `input/`. Options: `--methods hucp,iucp`, `--label NAME`, `--languages python,go`, `--exclude 'tests/*'`, `--skip-existing`, `--allow-imports`, `--workers N`. |
| `graph [paths...]` | Only build the call graph (`output/graph/<project>.json`), handy for checking what was parsed. |
| `evaluate [projects...]` | Blind LLM-as-judge comparison of all runs of a project. Options: `--trials` (default 10), `--seed`, `--include-modules`. |
| `mcq [projects...]` | Answer `mcqs/<project>.json` using only each run's summaries, plus a no-documentation control. |
| `languages` | List what this installation can parse. |

Every command that calls a model accepts `--model`, `--base-url`, `--workers`,
`--no-cache` and a provider flag (`--provider` / `--judge` / `--answerer`) that defaults
to `vllm`.

The original entry point still works: `./run_summarizer.sh [folder] [provider]`.

## Configuration

Settings come from the environment or a `.env` file (see `.env.example`); real
environment variables win.

| Variable | Default | Meaning |
|---|---|---|
| `VLLM_API_BASE` | `http://localhost:8000/v1` | OpenAI-compatible endpoint |
| `VLLM_MODEL` | first model the server lists | model to use |
| `VLLM_API_KEY` | `EMPTY` | if the server was started with `--api-key` |
| `VLLM_MAX_TOKENS` | server default | cap on answer length |
| `VLLM_TEMPERATURE` | server default | sampling temperature |
| `VLLM_TIMEOUT` | `600` | seconds per request |
| `VLLM_MAX_MODEL_LEN` | reported by the server | context length used to size prompts |
| `ECB_PROVIDER` | `vllm` | default provider for every command |

Reasoning models that emit `<think>...</think>` (Qwen3, DeepSeek-R1, ...) are supported:
the reasoning is stripped before answers are stored or parsed.

Hosted APIs remain available as optional extras: `--provider openai` (`OPENAI_API_KEY`,
`OPENAI_MODEL`, `OPENAI_BASE_URL`), `--provider anthropic` (`ANTHROPIC_API_KEY`, or
`PROJECT_ID` + `VERTEX_LOCATION` for Vertex AI; `ANTHROPIC_MODEL`, default
`claude-sonnet-5-5`) and `--provider gemini` (`GEMINI_API_KEY`, `GEMINI_MODEL`). Install
`anthropic` / `google-genai` to use them. `--provider mock` runs everything offline with
canned answers, for testing.

## Outputs

Everything goes under `output/`:

- `graph/<project>.json`: units, classes, call edges, unresolved calls and statistics.
- `summaries/<project>_<label>_<method>.json`: one file per run. `label` defaults to
  the model name (e.g. `qwen2_5_coder_32b_instruct`), so runs with different open models
  sit side by side. Each file records the package, method, provider, model, token usage,
  run time, `summaries` (`{unit id: summary}`) and `errors` (units the model failed on,
  never silently left blank).
- `evaluation/<project>_judge_<label>.json`: wins and win rates per run, per method and
  per model, plus per-unit votes.
- `evaluation/<project>_mcq_<label>.json`: accuracy per run and for the control.
- `cache/`: model answers keyed by prompt, so an interrupted run resumes without paying
  twice. Disable with `--no-cache`.

### Evaluation protocol

The judge sees the unit's code and every run's summary under anonymous labels (A, B, C,
...). The order is reshuffled on every trial (seeded, so results are reproducible). A
warning is printed when the judge model also produced some of the candidates.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The tests run offline: a mock provider covers the pipeline, and a small fake
OpenAI-compatible server stands in for vLLM. The multi-language fixtures live in
`tests/fixtures/polyglot/`.

Code layout (`explaincodebase/`):

| Module | Role |
|---|---|
| `languages.py` | language registry and detection |
| `parsing.py` | tree-sitter extraction of units, classes and call sites |
| `graph.py` | call resolution and bottom-up ordering |
| `llm.py` | vLLM and optional providers, retries, caching, token accounting |
| `prompts.py` | every prompt template |
| `methods.py` | the five methods and the parallel bottom-up scheduler |
| `docs.py` | documentation context for `docucp` |
| `pipeline.py` | running methods over a project and writing outputs |
| `evaluate.py`, `mcq.py` | the two evaluations |
| `cli.py` | command line interface |

## Changes from version 1

Version 1 was a set of Python-only scripts built on `code2flow`. Version 2 rewrote it
into one package; the old `output/` files are not compatible. Besides multi-language
support and vLLM as the default backend, it fixes:

- Units reached along several call paths were summarized more than once, the second time
  with their own earlier summary in the prompt.
- Prompts contained the summaries of *all* transitive callees, which grew without bound
  near the top of the graph and appeared in random order. Now only direct callees are
  included, deterministically, within a size budget.
- Functions not reachable from module-level code got no call-graph context at all; now
  every unit is processed bottom-up.
- Method 2 discarded callee context unless the call was a bare statement, matched
  callees by name only, and inserted string literals instead of comments.
- The baseline parser dropped any summary containing a colon; answers are now JSON with
  the unit ids the tool supplies.
- The judge saw method and model names in a fixed order; it is now blind and shuffled.
- Token counts were reported as 0 for OpenAI and not tracked for Anthropic; vLLM errors
  silently produced empty summaries; every provider's credentials were needed even to
  use one.
- Source lookup by `endswith(name + ".py")` could pick the wrong file; ids now use full
  relative paths.
- `mcqs/` was unused; the `mcq` command now scores runs against it.
