"""Every prompt the tool sends, in one place."""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    from .parsing import CodeUnit

SYSTEM = ("You are an expert software engineer who writes accurate, concise documentation "
          "for source code in any programming language.")

# Markers let the offline mock provider recognise what kind of answer is expected.
BASELINE_MARKER = "## Units to summarize"
JUDGE_MARKER = "## Candidate summaries"
MCQ_MARKER = "## Multiple-choice question"

SUMMARY_TASK = ("Summarize what `{name}` does in one sentence, then break that sentence into "
                "2-5 bullet points. Describe its purpose and behaviour rather than restating the "
                "syntax. Reply with the summary only.")

KIND_LABELS = {"function": "function", "method": "method", "module": "top-level code",
               "file": "file"}


def describe(unit: "CodeUnit") -> str:
    return f"{unit.language} {KIND_LABELS.get(unit.kind, unit.kind)} `{unit.qualname}`"


def truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    cut = text[:limit]
    dropped = text[limit:].count("\n") + 1
    return f"{cut}\n... [truncated: {dropped} more line(s)]"


def one_line(text: str, limit: int = 300) -> str:
    """Collapse a bullet-point summary into a single line (for inline code comments)."""
    parts = [line.strip().lstrip("-*•0123456789.) ").strip() for line in text.splitlines()]
    joined = "; ".join(part for part in parts if part)
    return joined if len(joined) <= limit else joined[: limit - 3].rstrip() + "..."


def code_section(unit: "CodeUnit", code: str, max_chars: int) -> str:
    return (f"# Code: {KIND_LABELS.get(unit.kind, unit.kind)} `{unit.qualname}` in "
            f"`{unit.file}` (lines {unit.start_line}-{unit.end_line}, {unit.language})\n"
            f"```{unit.language}\n{truncate(code, max_chars)}\n```")


def callee_section(unit: "CodeUnit", callees: list[tuple["CodeUnit", str]],
                   max_each: int, max_total: int) -> str | None:
    if not callees:
        return None
    lines, used = [], 0
    for callee, summary in callees:
        entry = f"- `{callee.qualname}` (`{callee.file}`):\n" + "\n".join(
            f"  {line}" for line in truncate(summary.strip(), max_each).splitlines())
        if max_total and used + len(entry) > max_total:
            lines.append(f"- ... {len(callees) - len(lines)} more callee(s) omitted for length")
            break
        lines.append(entry)
        used += len(entry)
    return f"# Context: what the code called by `{unit.qualname}` does\n" + "\n".join(lines)


def summary_prompt(unit: "CodeUnit", code: str, sections: Iterable[str | None], max_code_chars: int) -> str:
    parts = [section for section in sections if section]
    parts.append(code_section(unit, code, max_code_chars))
    parts.append("# Task\n" + SUMMARY_TASK.format(name=unit.qualname))
    return "\n\n".join(parts)


def line_explanation_prompt(unit: "CodeUnit", max_code_chars: int) -> str:
    return "\n\n".join([
        code_section(unit, unit.source, max_code_chars),
        f"# Task\nExplain every line of the {describe(unit)} above, in order. Be brief and "
        "concrete: one short sentence per line or per small group of closely related lines.",
    ])


def baseline_prompt(files: list[tuple[str, str, str]], unit_ids: list[str], part: str) -> str:
    """``files`` holds ``(path, language, text)``."""
    code = "\n\n".join(f"### File: `{path}`\n```{language}\n{text}\n```" for path, language, text in files)
    ids = "\n".join(f"- `{uid}`" for uid in unit_ids)
    return (
        f"# Codebase{part}\n{code}\n\n"
        f"{BASELINE_MARKER}\n"
        "Each id is `<file path>::<qualified name>`; `(global)` means the file's top-level code "
        "and `(file)` means the whole file.\n"
        f"{ids}\n\n"
        "# Task\nFor every id listed above, summarize what that code does in one sentence broken "
        "into 2-5 bullet points. Return only a JSON object that maps each id (exactly as written, "
        "without backticks) to its summary as a single string with the bullets separated by "
        "newlines."
    )


def judge_prompt(unit: "CodeUnit", code: str, candidates: list[tuple[str, str]], max_code_chars: int) -> str:
    listing = "\n\n".join(f"### Summary {label}\n{text.strip()}" for label, text in candidates)
    labels = ", ".join(label for label, _ in candidates)
    return (
        f"{code_section(unit, code, max_code_chars)}\n\n"
        f"{JUDGE_MARKER}\n{listing}\n\n"
        "# Task\nJudge which summary describes the code above most accurately, completely and "
        "concisely. Ignore formatting differences. Respond with JSON only, in the form "
        f'{{"best": "<label>"}}, where <label> is one of: {labels}.'
    )


def mcq_prompt(context: str, question: str, options: dict[str, str]) -> str:
    choices = "\n".join(f"{key}. {value}" for key, value in sorted(options.items()))
    context_block = context if context else "(no documentation provided)"
    return (
        f"# Documentation of a codebase\n{context_block}\n\n"
        f"{MCQ_MARKER}\n{question}\n\n{choices}\n\n"
        "# Task\nUsing the documentation above, answer with the letter of the correct option only."
    )
