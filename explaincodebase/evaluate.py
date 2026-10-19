"""LLM-as-judge comparison of the summaries produced by different methods and models.

For every unit, the judge sees the code and each run's summary under an anonymous label
(A, B, C, ...). The order is reshuffled on every trial so that neither the method name nor
the position can sway the verdict.
"""

from __future__ import annotations

import json
import logging
import random
import re
import string
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from . import prompts
from .llm import LLM, LLMError, default_label, make_provider
from .pipeline import build_graph, load_runs, write_json

log = logging.getLogger(__name__)


def run_key(run: dict) -> str:
    return f"{run['method']}@{run['label']}"


def parse_choice(text: str, labels: list[str]) -> str | None:
    """Read ``{"best": "B"}`` (or a bare letter) from a judge's answer."""
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try:
            value = str(json.loads(match.group(0)).get("best", "")).strip().strip(".")
            if value in labels:
                return value
        except (ValueError, AttributeError):
            pass
    match = re.search(r"best\W{0,12}([A-Z])\b", text, re.I)
    if match and match.group(1).upper() in labels:
        return match.group(1).upper()
    found = [label for label in re.findall(r"\b([A-Z])\b", text) if label in labels]
    return found[0] if len(set(found)) == 1 else None


def evaluate_package(package: str, *, output_dir: str = "output", judge: str | None = None,
                     model: str | None = None, base_url: str | None = None, trials: int = 10,
                     seed: int = 0, workers: int = 8, include_modules: bool = False,
                     source_path: str | None = None, cache: bool = True,
                     max_code_chars: int = 16_000) -> dict:
    runs = load_runs(output_dir, package)
    if len(runs) < 2:
        raise ValueError(f"Need at least two summary files for {package!r} in {output_dir}/summaries")
    path = source_path or runs[0].get("source_path")
    graph = build_graph(path)

    backend = make_provider(judge, model, base_url)
    llm = LLM(backend, cache_dir=f"{output_dir}/cache" if cache else None)
    context = getattr(backend, "context_length", None)
    if context:
        max_code_chars = min(max_code_chars, int(context * 0.4 * 3))
    judge_label = default_label(backend)
    warnings = []
    authored = sorted({run["label"] for run in runs if run["label"] == judge_label})
    if authored:
        warnings.append(f"The judge ({judge_label}) also wrote some candidates; it may favour its own "
                        "summaries. Consider judging with a different model.")

    tasks = []
    for uid, unit in graph.units.items():
        if unit.kind == "module" and not include_modules:
            continue
        candidates = [(run_key(run), run["summaries"][uid]) for run in runs if uid in run["summaries"]]
        if len(candidates) >= 2:
            tasks.append((uid, unit, candidates))
    if not tasks:
        raise ValueError(f"No unit of {package!r} has summaries from two or more runs to compare")

    def judge_unit(task):
        uid, unit, candidates = task
        rng = random.Random(f"{seed}:{uid}")
        votes: Counter = Counter()
        invalid = 0
        for trial in range(trials):
            order = candidates[:]
            rng.shuffle(order)
            labels = list(string.ascii_uppercase[: len(order)])
            prompt = prompts.judge_prompt(unit, unit.source, list(zip(labels, (s for _, s in order))),
                                          max_code_chars)
            try:
                answer = llm.complete(prompt, cache_tag=f"trial{trial}")
            except LLMError as exc:
                log.error("judge failed on %s: %s", uid, exc)
                invalid += 1
                continue
            choice = parse_choice(answer, labels)
            if choice is None:
                invalid += 1
            else:
                votes[order[labels.index(choice)][0]] += 1
        return uid, [key for key, _ in candidates], votes, invalid

    wins: Counter = Counter()
    appearances: Counter = Counter()
    per_unit = {}
    invalid_total = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for uid, keys, votes, invalid in pool.map(judge_unit, tasks):
            invalid_total += invalid
            valid = sum(votes.values())
            for key in keys:
                appearances[key] += valid
            wins.update(votes)
            per_unit[uid] = {"votes": dict(votes), "invalid": invalid}

    def grouped(index: int) -> dict:
        totals, seen = defaultdict(int), defaultdict(int)
        for key in appearances:
            group = key.split("@")[index]
            totals[group] += wins.get(key, 0)
            seen[group] += appearances[key]
        return {g: {"wins": totals[g], "win_rate": round(totals[g] / seen[g], 4) if seen[g] else 0.0}
                for g in sorted(totals, key=lambda g: -totals[g])}

    result = {
        "package": package,
        "judge": {"provider": backend.name, "model": backend.model, "label": judge_label},
        "trials_per_unit": trials,
        "units_evaluated": len(tasks),
        "invalid_answers": invalid_total,
        "by_run": grouped_runs(wins, appearances),
        "by_method": grouped(0),
        "by_model": grouped(1),
        "usage": llm.usage,
        "warnings": warnings,
        "per_unit": per_unit,
    }
    out_file = f"{output_dir}/evaluation/{package}_judge_{judge_label}.json"
    write_json(out_file, result)
    result["file"] = out_file
    return result


def grouped_runs(wins: Counter, appearances: Counter) -> dict:
    return {key: {"wins": wins.get(key, 0),
                  "win_rate": round(wins.get(key, 0) / appearances[key], 4) if appearances[key] else 0.0}
            for key in sorted(appearances, key=lambda k: -wins.get(k, 0))}
