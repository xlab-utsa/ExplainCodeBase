"""Run the summarization methods over a codebase and write their outputs."""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone

from .graph import CallGraph
from .llm import LLM, default_label, make_provider
from .methods import METHODS, Settings
from .parsing import parse_codebase

log = logging.getLogger(__name__)

ALL_METHODS = ("baseline", "hucp", "iucp", "llmucp", "docucp")


def package_name(path: str) -> str:
    return os.path.basename(os.path.normpath(os.path.abspath(path)))


def input_packages(input_dir: str = "input") -> list[str]:
    """Every package directory under ``input/``."""
    if not os.path.isdir(input_dir):
        return []
    return sorted(
        os.path.join(input_dir, name) for name in os.listdir(input_dir)
        if os.path.isdir(os.path.join(input_dir, name)) and not name.startswith(".")
    )


def build_graph(path: str, languages: set[str] | None = None, exclude: list[str] | None = None) -> CallGraph:
    files = parse_codebase(path, languages, exclude)
    if not files:
        raise ValueError(f"No source files found under {path}")
    return CallGraph(files)


def write_json(path: str, data) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def summary_path(output_dir: str, package: str, label: str, method: str) -> str:
    return os.path.join(output_dir, "summaries", f"{package}_{label}_{method}.json")


def summarize_package(
    path: str,
    provider: str | None = None,
    methods: list[str] = ALL_METHODS,
    *,
    output_dir: str = "output",
    model: str | None = None,
    base_url: str | None = None,
    label: str | None = None,
    settings: Settings | None = None,
    cache: bool = True,
    languages: set[str] | None = None,
    exclude: list[str] | None = None,
    skip_existing: bool = False,
) -> dict[str, str]:
    """Summarize one codebase with each method; returns ``{method: output file}``."""
    settings = settings or Settings()
    package = package_name(path)
    graph = build_graph(path, languages, exclude)
    graph_file = os.path.join(output_dir, "graph", f"{package}.json")
    write_json(graph_file, graph.to_json())
    stats = graph.to_json()["stats"]
    log.info("%s: %d files, %d units, %d call edges (%s)", package, stats["files"], stats["units"],
             stats["edges"], ", ".join(f"{k}: {v}" for k, v in sorted(stats["languages"].items())))

    backend = make_provider(provider, model, base_url)
    settings.fit_context(getattr(backend, "context_length", None))
    label = label or default_label(backend)
    cache_dir = os.path.join(output_dir, "cache") if cache else None
    written = {}
    for key in methods:
        out_file = summary_path(output_dir, package, label, key)
        if skip_existing and os.path.exists(out_file):
            log.info("%s: %s exists, skipping", package, out_file)
            written[key] = out_file
            continue
        method = METHODS[key]()
        llm = LLM(backend, cache_dir=cache_dir)
        log.info("%s: running %s with %s (%s)", package, method.title, backend.name, backend.model)
        started = time.time()
        last_report = [0.0]

        def progress(done: int, total: int, _key=key) -> None:
            now = time.time()
            if done == total or now - last_report[0] > 5:
                last_report[0] = now
                log.info("%s/%s: %d/%d units", package, _key, done, total)

        result = method.run(graph, llm, settings, progress)
        write_json(out_file, {
            "package": package,
            "source_path": os.path.abspath(path),
            "method": key,
            "label": label,
            "provider": backend.name,
            "model": backend.model,
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "seconds": round(time.time() - started, 1),
            "usage": llm.usage,
            "summaries": result.summaries,
            "errors": result.errors,
        })
        log.info("%s/%s: %d summaries, %d errors, tokens in/out %d/%d -> %s", package, key,
                 len(result.summaries), len(result.errors), llm.usage["input_tokens"],
                 llm.usage["output_tokens"], out_file)
        written[key] = out_file
    return written


def load_runs(output_dir: str, package: str | None = None) -> list[dict]:
    """Every summary file in ``output/summaries`` (optionally for one package)."""
    folder = os.path.join(output_dir, "summaries")
    runs = []
    if not os.path.isdir(folder):
        return runs
    for name in sorted(os.listdir(folder)):
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(folder, name), encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            log.warning("Ignoring unreadable file %s", name)
            continue
        if not isinstance(data, dict) or "summaries" not in data or "method" not in data:
            log.warning("Ignoring %s: not produced by this version of the tool", name)
            continue
        if package is None or data.get("package") == package:
            data["file"] = os.path.join(folder, name)
            runs.append(data)
    return runs
