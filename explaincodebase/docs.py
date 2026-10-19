"""Documentation context for the ``docucp`` method.

For each unit this gathers:

* the documentation of its enclosing class and of that class's ancestors in the codebase;
* for Python, the docstrings of built-ins and standard-library functions it calls, and
  (only with ``allow_imports``) of third-party base classes and functions. Importing a
  package executes its code, so that part is opt-in.
"""

from __future__ import annotations

import builtins
import importlib
import inspect
import re
import sys
from functools import lru_cache

from .graph import CallGraph
from .parsing import CodeUnit

# Standard-library modules with side effects on import.
UNSAFE_STDLIB = frozenset({"antigravity", "this", "idlelib", "turtle", "turtledemo", "tkinter"})
MAX_DOC_CHARS = 500


def _first_paragraph(doc: str | None, limit: int = MAX_DOC_CHARS) -> str | None:
    if not doc:
        return None
    paragraph = inspect.cleandoc(doc).split("\n\n", 1)[0].strip()
    return paragraph if len(paragraph) <= limit else paragraph[: limit - 3] + "..."


def class_context(unit: CodeUnit, graph: CallGraph) -> list[str]:
    if not unit.class_name:
        return []
    entries = []
    for name in graph.bases(unit.class_name):
        for info in graph.classes.get(name, []):
            doc = _first_paragraph(info.doc) or "(no documentation)"
            role = "defined in" if name == unit.class_name else "ancestor, defined in"
            entries.append(f"- `{info.header}` ({role} `{info.file}`): {doc}")
    return entries


def _python_imports(graph: CallGraph, unit: CodeUnit) -> dict[str, str]:
    """Local name -> imported dotted path, from the unit's file."""
    source = graph.files.get(unit.file)
    if source is None:
        return {}
    names = {}
    for line in source.imports:
        match = re.match(r"from\s+([\w.]+)\s+import\s+(.+)", line)
        if match:
            module, items = match.groups()
            for item in items.strip("() ").split(","):
                parts = item.strip().split(" as ")
                if parts[0] and parts[0] != "*":
                    names[parts[-1].strip()] = f"{module}.{parts[0].strip()}"
            continue
        match = re.match(r"import\s+(.+)", line)
        if match:
            for item in match.group(1).split(","):
                parts = item.strip().split(" as ")
                names[parts[-1].strip().split(".")[0]] = parts[0].strip() if len(parts) > 1 else parts[0].split(".")[0]
    return names


@lru_cache(maxsize=4096)
def _python_doc(dotted: str, allow_imports: bool) -> str | None:
    head = dotted.split(".")[0]
    is_stdlib = head in sys.stdlib_module_names and head not in UNSAFE_STDLIB
    if not (is_stdlib or allow_imports):
        return None
    parts = dotted.split(".")
    for split in range(len(parts), 0, -1):
        module_name = ".".join(parts[:split])
        try:
            obj = importlib.import_module(module_name)
        except Exception:
            continue
        try:
            for attr in parts[split:]:
                obj = getattr(obj, attr)
        except AttributeError:
            return None
        return _first_paragraph(inspect.getdoc(obj))
    return None


def external_call_docs(unit: CodeUnit, graph: CallGraph, allow_imports: bool = False) -> list[str]:
    if unit.language != "python":
        return []
    imports = _python_imports(graph, unit)
    entries, seen = [], set()
    for resolved in graph.calls.get(unit.id, []):
        if resolved.targets:
            continue
        site = resolved.site
        if site.receiver is None:
            if site.name in imports:
                dotted = imports[site.name]
            elif hasattr(builtins, site.name):
                dotted = f"builtins.{site.name}"
            else:
                continue
        else:
            root, _, rest = site.receiver.partition(".")
            if root not in imports or not re.fullmatch(r"[\w.]*", rest):
                continue
            dotted = ".".join(filter(None, [imports[root], rest, site.name]))
        if dotted in seen:
            continue
        seen.add(dotted)
        doc = _python_doc(dotted, allow_imports)
        if doc:
            label = dotted.removeprefix("builtins.")
            entries.append(f"- `{label}`: {doc}")
    return entries


def external_base_docs(unit: CodeUnit, graph: CallGraph, allow_imports: bool) -> list[str]:
    """Python base classes that live outside the codebase (e.g. ``nn.Module``)."""
    if unit.language != "python" or not unit.class_name:
        return []
    imports = _python_imports(graph, unit)
    entries = []
    for name in graph.bases(unit.class_name):
        for info in graph.classes.get(name, []):
            for base in info.bases:
                if base in graph.classes:
                    continue
                dotted = imports.get(base)
                if dotted is None:
                    header_match = re.search(rf"([\w.]+\.{re.escape(base)})\b", info.header)
                    if header_match:
                        root = header_match.group(1).split(".")[0]
                        if root in imports:
                            dotted = imports[root] + header_match.group(1)[len(root):]
                doc = _python_doc(dotted, allow_imports) if dotted else None
                if doc:
                    entries.append(f"- base class `{dotted}`: {doc}")
    return entries


def documentation_context(unit: CodeUnit, graph: CallGraph, allow_imports: bool = False) -> str | None:
    sections = []
    classes = class_context(unit, graph) + external_base_docs(unit, graph, allow_imports)
    if classes:
        sections.append("## Enclosing class and its ancestors\n" + "\n".join(classes))
    calls = external_call_docs(unit, graph, allow_imports)
    if calls:
        sections.append("## Library functions it calls\n" + "\n".join(calls))
    if not sections:
        return None
    return f"# Context: documentation relevant to `{unit.qualname}`\n" + "\n\n".join(sections)
