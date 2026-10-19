"""Comprehension test: can a model answer questions about a codebase from its summaries alone?

Questions live in ``mcqs/<package>.json`` as a list of
``{"question", "options": {"A": ..., ...}, "correct_answer": "B"}``. Each summary run is used
as the only documentation an answering model gets; a "no documentation" control run shows how
much the model can guess without it.
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor

from . import prompts
from .evaluate import run_key
from .llm import LLM, LLMError, default_label, make_provider
from .pipeline import load_runs, write_json

log = logging.getLogger(__name__)
CONTROL = "no_documentation"


def load_questions(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    questions = data.get("questions", data) if isinstance(data, dict) else data
    return [q for q in questions if q.get("question") and q.get("options") and q.get("correct_answer")]


def documentation(run: dict, max_chars: int) -> str:
    parts, used = [], 0
    for uid in sorted(run["summaries"]):
        entry = f"### `{uid}`\n{run['summaries'][uid].strip()}"
        if max_chars and used + len(entry) > max_chars:
            parts.append(f"... ({len(run['summaries']) - len(parts)} more summaries omitted for length)")
            break
        parts.append(entry)
        used += len(entry) + 2
    return "\n\n".join(parts)


def parse_letter(text: str, options: list[str]) -> str | None:
    cleaned = text.strip().strip("*`()[]. ")
    if cleaned[:1].upper() in options and (len(cleaned) == 1 or not cleaned[1:2].isalpha()):
        return cleaned[0].upper()
    found = re.findall(r"\b([A-Z])\b", text)
    found = [letter for letter in found if letter in options]
    return found[-1] if found else None


def run_mcq(package: str, *, mcq_file: str | None = None, output_dir: str = "output",
            answerer: str | None = None, model: str | None = None, base_url: str | None = None,
            max_context_chars: int = 200_000, workers: int = 8, control: bool = True,
            cache: bool = True) -> dict:
    mcq_file = mcq_file or os.path.join("mcqs", f"{package}.json")
    questions = load_questions(mcq_file)
    if not questions:
        raise ValueError(f"No usable questions in {mcq_file}")
    runs = load_runs(output_dir, package)
    if not runs:
        raise ValueError(f"No summaries for {package!r} in {output_dir}/summaries")

    backend = make_provider(answerer, model, base_url)
    llm = LLM(backend, cache_dir=f"{output_dir}/cache" if cache else None)
    context = getattr(backend, "context_length", None)
    if context:
        max_context_chars = min(max_context_chars, int(context * 0.7 * 3))

    contexts = {run_key(run): documentation(run, max_context_chars) for run in runs}
    if control:
        contexts[CONTROL] = ""

    def answer(item):
        key, question = item
        options = sorted(question["options"])
        prompt = prompts.mcq_prompt(contexts[key], question["question"], question["options"])
        try:
            letter = parse_letter(llm.complete(prompt), options)
        except LLMError as exc:
            log.error("mcq: %s", exc)
            letter = None
        return key, question, letter

    items = [(key, q) for key in contexts for q in questions]
    scores = {key: {"correct": 0, "answered": 0, "total": len(questions)} for key in contexts}
    answers: dict[str, dict[str, str | None]] = {key: {} for key in contexts}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for key, question, letter in pool.map(answer, items):
            qid = question.get("question_number") or question["question"][:60]
            answers[key][qid] = letter
            if letter is not None:
                scores[key]["answered"] += 1
            if letter == str(question["correct_answer"]).strip().upper():
                scores[key]["correct"] += 1
    for score in scores.values():
        score["accuracy"] = round(score["correct"] / score["total"], 4)

    result = {
        "package": package,
        "questions_file": mcq_file,
        "answerer": {"provider": backend.name, "model": backend.model},
        "scores": dict(sorted(scores.items(), key=lambda kv: -kv[1]["accuracy"])),
        "answers": answers,
        "usage": llm.usage,
    }
    out_file = os.path.join(output_dir, "evaluation", f"{package}_mcq_{default_label(backend)}.json")
    write_json(out_file, result)
    result["file"] = out_file
    return result
