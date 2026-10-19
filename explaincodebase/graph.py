"""Resolve call sites to code units and order units bottom-up (callees before callers)."""

from __future__ import annotations

import os
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache

from .languages import detailed
from .parsing import CallSite, ClassInfo, CodeUnit, SourceFile

# Languages that can call each other's definitions directly.
FAMILIES = {"typescript": "js", "tsx": "js", "javascript": "js", "c": "c", "cpp": "c"}


def family(language: str) -> str:
    return FAMILIES.get(language, language)


@lru_cache(maxsize=8192)
def _pattern(template: str, variable: str) -> re.Pattern:
    return re.compile(template.replace("{v}", re.escape(variable)))


@dataclass
class ResolvedCall:
    site: CallSite
    targets: list[str]


class CallGraph:
    def __init__(self, files: list[SourceFile]):
        self.files = {f.path: f for f in files}
        self.units: dict[str, CodeUnit] = {}
        for source in files:
            for unit in source.units:
                self.units[unit.id] = unit
        self.classes: dict[str, list[ClassInfo]] = defaultdict(list)
        for source in files:
            for info in source.classes:
                self.classes[info.name].append(info)
        self._index()
        self.calls: dict[str, list[ResolvedCall]] = {}
        self.edges: dict[str, list[str]] = {}
        self._local_types: dict[tuple[str, str], str | None] = {}
        self._resolve_all()

    # ------------------------------------------------------------------ indexes

    def _index(self) -> None:
        self.top_level: dict[tuple[str, str], list[CodeUnit]] = defaultdict(list)
        self.functions_by_name: dict[str, list[CodeUnit]] = defaultdict(list)
        self.methods: dict[str, dict[str, list[CodeUnit]]] = defaultdict(lambda: defaultdict(list))
        self.methods_by_name: dict[str, list[CodeUnit]] = defaultdict(list)
        self.nested: dict[str, dict[str, CodeUnit]] = defaultdict(dict)
        self.files_by_stem: dict[str, list[str]] = defaultdict(list)
        self.files_by_dir: dict[str, list[str]] = defaultdict(list)
        for path in self.files:
            stem = os.path.splitext(os.path.basename(path))[0]
            self.files_by_stem[stem].append(path)
            self.files_by_dir[os.path.basename(os.path.dirname(path))].append(path)
        for unit in self.units.values():
            if unit.kind in ("module", "file"):
                continue
            if unit.parent:
                self.nested[unit.parent][unit.name] = unit
            elif unit.class_name:
                self.methods[unit.class_name][unit.name].append(unit)
                self.methods_by_name[unit.name].append(unit)
            else:
                self.top_level[(unit.file, unit.name)].append(unit)
                self.functions_by_name[unit.name].append(unit)

    def bases(self, class_name: str) -> list[str]:
        """Class plus its ancestors that are defined in the codebase, nearest first."""
        order, queue, seen = [], [class_name], set()
        while queue:
            name = queue.pop(0)
            if name in seen:
                continue
            seen.add(name)
            order.append(name)
            for info in self.classes.get(name, []):
                queue.extend(base for base in info.bases if base not in seen)
        return order

    def is_constructor(self, unit: CodeUnit) -> bool:
        spec = detailed(unit.language)
        if spec is None or not unit.class_name:
            return unit.name in ("__init__", "constructor", "initialize", "init", "new")
        return unit.name in spec.constructor_names or (
            spec.class_named_constructor and unit.name == unit.class_name)

    # ------------------------------------------------------------------ resolution

    def _resolve_all(self) -> None:
        for uid, unit in self.units.items():
            resolved, targets = [], []
            for site in unit.calls:
                found = [t for t in self.resolve(unit, site) if t in self.units]
                resolved.append(ResolvedCall(site, found))
                for target in found:
                    if target not in targets:
                        targets.append(target)
            self.calls[uid] = resolved
            self.edges[uid] = targets

    def _prefer(self, caller: CodeUnit, candidates: list[CodeUnit]) -> list[CodeUnit]:
        """Pick the most plausible of several same-named definitions, or none."""
        same_family = [c for c in candidates if family(c.language) == family(caller.language)]
        if len(same_family) <= 1:
            return same_family
        for key in (lambda c: c.file == caller.file,
                    lambda c: os.path.dirname(c.file) == os.path.dirname(caller.file)):
            narrowed = [c for c in same_family if key(c)]
            if len(narrowed) == 1:
                return narrowed
        imported = [c for c in same_family if self._imports_file(caller.file, c.file)]
        return imported if len(imported) == 1 else []

    def _imports_file(self, importer: str, target: str) -> bool:
        source = self.files.get(importer)
        if source is None:
            return False
        stem = os.path.splitext(os.path.basename(target))[0]
        pattern = re.compile(rf"\b{re.escape(stem)}\b")
        return any(pattern.search(line) for line in source.imports)

    def find_method(self, class_name: str, name: str, caller: CodeUnit,
                    include_self: bool = True) -> list[CodeUnit]:
        chain = self.bases(class_name)
        if not include_self:
            chain = chain[1:]
        for cls in chain:
            found = self._prefer(caller, self.methods.get(cls, {}).get(name, []))
            if found:
                return found
        return []

    def constructors(self, class_name: str, caller: CodeUnit) -> list[CodeUnit]:
        for cls in self.bases(class_name):
            ctors = [u for group in self.methods.get(cls, {}).values() for u in group
                     if self.is_constructor(u)]
            found = self._prefer(caller, ctors)
            if found:
                return found
            if ctors:
                return []
        return []

    def resolve(self, caller: CodeUnit, site: CallSite) -> list[str]:
        return [u.id for u in self._resolve(caller, site)]

    def _resolve(self, caller: CodeUnit, site: CallSite) -> list[CodeUnit]:
        spec = detailed(caller.language)
        self_names = (spec.self_names if spec else frozenset({"self", "this"}))
        super_names = (spec.super_names if spec else frozenset({"super"}))
        cls = caller.class_name or self._enclosing_class(caller)
        name = site.name

        if site.kind == "new":
            return self.constructors(name, caller)
        if site.kind == "super":
            if not cls:
                return []
            if self.is_constructor(caller):
                for base in self.bases(cls)[1:]:
                    found = self.constructors(base, caller)
                    if found:
                        return found
                return []
            return self.find_method(cls, caller.name, caller, include_self=False)

        receiver = site.receiver
        if receiver is None:
            scope = caller
            while scope is not None:
                hit = self.nested.get(scope.id, {}).get(name)
                if hit is not None:
                    return [hit]
                scope = self.units.get(scope.parent) if scope.parent else None
            if cls and (spec is None or spec.implicit_self):
                found = self.find_method(cls, name, caller)
                if found:
                    return found
            same_file = self.top_level.get((caller.file, name))
            if same_file:
                return same_file[:1]
            if name in self.classes:
                return self.constructors(name, caller)
            if "::" in name:  # C++ ns::func / Rust Type::func written as one token
                head, _, tail = name.rpartition("::")
                return self._resolve(caller, CallSite(tail, head, site.line, site.kind))
            return self._prefer(caller, self.functions_by_name.get(name, []))

        receiver = receiver.strip()
        root = re.split(r"[.\-:>(\[]", receiver, maxsplit=1)[0]
        if receiver.rstrip("()") in super_names or root in super_names:
            return self.find_method(cls, name, caller, include_self=False) if cls else []
        if receiver in self_names or (caller.receiver_var and receiver == caller.receiver_var):
            return self.find_method(cls, name, caller) if cls else []
        if receiver in self.classes:  # static call: Util.square(), Counter::new()
            found = self.find_method(receiver, name, caller)
            if found:
                return found
        if root in self_names or (caller.receiver_var and root == caller.receiver_var):
            attr = receiver[len(root):].lstrip(".->:")
            typ = self._attribute_type(cls, attr) if cls else None
            if typ:
                return self.find_method(typ, name, caller)
        typ = self._local_type(caller, receiver)
        if typ:
            found = self.find_method(typ, name, caller)
            if found:
                return found
        module_hits = self._module_function(caller, receiver, name)
        if module_hits:
            return module_hits
        if self._is_external(caller, root):
            return []
        # Last resort: a method name defined by exactly one class, and that class is
        # referenced in the caller's file (otherwise ``tensor.numel()`` would match any
        # repo class that happens to define ``numel``).
        candidates = self.methods_by_name.get(name, [])
        if len({c.class_name for c in candidates}) != 1:
            return []
        owner = candidates[0].class_name
        source = self.files.get(caller.file)
        if candidates[0].file != caller.file and (
                source is None or not re.search(rf"\b{re.escape(owner)}\b", source.text)):
            return []
        return self._prefer(caller, candidates)

    def _enclosing_class(self, unit: CodeUnit) -> str | None:
        while unit is not None and unit.parent:
            unit = self.units.get(unit.parent)
            if unit is not None and unit.class_name:
                return unit.class_name
        return None

    def _module_function(self, caller: CodeUnit, receiver: str, name: str) -> list[CodeUnit]:
        """``utils.helper()`` (module file) or ``pkg.Func()`` (Go package directory)."""
        last = re.split(r"[.:]+", receiver)[-1]
        paths = self.files_by_stem.get(last, []) + self.files_by_dir.get(last, [])
        hits = [u for path in paths for u in self.top_level.get((path, name), [])]
        return self._prefer(caller, hits)

    def _is_external(self, caller: CodeUnit, root: str) -> bool:
        """Receiver names an imported module that is not part of the codebase."""
        if not root or root in self.files_by_stem or root in self.files_by_dir or root in self.classes:
            return False
        source = self.files.get(caller.file)
        if source is None:
            return False
        pattern = re.compile(rf"\b{re.escape(root)}\b")
        return any(pattern.search(line) for line in source.imports)

    _ASSIGN = (
        r"\b{v}\s*(?::\s*[\w.<>\[\], ]+?\s*)?:?=\s*(?:new\s+|&)?([A-Z]\w*)\s*(?:[({]|::|\.new\b)",
        r"\b([A-Z]\w*)(?:<[^>]*>)?[*&\s]+{v}\s*[=;({]",
        r"\b{v}\s*:\s*\*?([A-Z]\w*)\b",
    )

    def _local_type(self, caller: CodeUnit, variable: str) -> str | None:
        key = (caller.id, variable)
        if key not in self._local_types:
            self._local_types[key] = None  # also guards against recursive lookups
            self._local_types[key] = self._infer_local_type(caller, variable)
        return self._local_types[key]

    def _infer_local_type(self, caller: CodeUnit, variable: str) -> str | None:
        if not self.classes or not re.fullmatch(r"[A-Za-z_$][\w$]*", variable):
            return None
        source = caller.source
        if variable not in source:
            return None
        for template in self._ASSIGN:
            for match in _pattern(template, variable).finditer(source):
                if match.group(1) in self.classes:
                    return match.group(1)
        # x = make_thing() where make_thing's signature names a known class
        match = _pattern(r"\b{v}\s*:?=\s*([A-Za-z_]\w*)\s*\(", variable).search(source)
        if match:
            for unit in self._resolve(caller, CallSite(match.group(1), None, 0)):
                header = unit.lines[0][1] if unit.lines else ""
                for word in re.findall(r"[A-Z]\w*", header.split(unit.name, 1)[-1]):
                    if word in self.classes:
                        return word
        return None

    def _attribute_type(self, cls: str, attribute: str) -> str | None:
        if not re.fullmatch(r"\w+", attribute or ""):
            return None
        pattern = re.compile(
            rf"(?:self|this)\s*(?:\.|->)\s*{re.escape(attribute)}\s*(?::\s*[\w.<>]+\s*)?=\s*(?:new\s+)?([A-Z]\w*)")
        for group in self.methods.get(cls, {}).values():
            for unit in group:
                match = pattern.search(unit.source)
                if match and match.group(1) in self.classes:
                    return match.group(1)
        return None

    # ------------------------------------------------------------------ ordering

    def callees(self, uid: str) -> list[str]:
        return self.edges.get(uid, [])

    def callers(self) -> dict[str, list[str]]:
        reverse: dict[str, list[str]] = defaultdict(list)
        for src, targets in self.edges.items():
            for dst in targets:
                reverse[dst].append(src)
        return reverse

    def components(self) -> list[list[str]]:
        """Strongly connected components, callees' components before callers' (Tarjan)."""
        index: dict[str, int] = {}
        low: dict[str, int] = {}
        on_stack: set[str] = set()
        stack: list[str] = []
        result: list[list[str]] = []
        counter = 0
        for start in self.units:
            if start in index:
                continue
            work = [(start, 0)]
            while work:
                node, child_i = work[-1]
                if child_i == 0:
                    index[node] = low[node] = counter
                    counter += 1
                    stack.append(node)
                    on_stack.add(node)
                children = self.edges.get(node, [])
                if child_i < len(children):
                    work[-1] = (node, child_i + 1)
                    child = children[child_i]
                    if child not in index:
                        work.append((child, 0))
                    elif child in on_stack:
                        low[node] = min(low[node], index[child])
                    continue
                work.pop()
                if work:
                    parent = work[-1][0]
                    low[parent] = min(low[parent], low[node])
                if low[node] == index[node]:
                    component = []
                    while True:
                        member = stack.pop()
                        on_stack.discard(member)
                        component.append(member)
                        if member == node:
                            break
                    result.append(sorted(component, key=self._order_key))
        return result

    def _order_key(self, uid: str):
        unit = self.units[uid]
        return (unit.file, unit.start_line, uid)

    def leaves(self) -> set[str]:
        return {uid for uid, targets in self.edges.items() if not targets}

    # ------------------------------------------------------------------ export

    def to_json(self) -> dict:
        unresolved = {}
        for uid, calls in self.calls.items():
            missing = sorted({(f"{c.site.receiver}." if c.site.receiver else "") + c.site.name
                              for c in calls if not c.targets})
            if missing:
                unresolved[uid] = missing
        languages = defaultdict(int)
        for source in self.files.values():
            languages[source.language.name] += 1
        return {
            "units": {
                uid: {
                    "name": u.name, "qualname": u.qualname, "kind": u.kind, "file": u.file,
                    "language": u.language, "start_line": u.start_line, "end_line": u.end_line,
                    "class": u.class_name, "parent": u.parent,
                } for uid, u in self.units.items()
            },
            "classes": {
                name: [{"file": c.file, "bases": c.bases, "start_line": c.start_line} for c in infos]
                for name, infos in self.classes.items()
            },
            "edges": [[src, dst] for src, targets in self.edges.items() for dst in targets],
            "unresolved_calls": unresolved,
            "stats": {
                "files": len(self.files), "languages": dict(languages), "units": len(self.units),
                "edges": sum(len(t) for t in self.edges.values()),
            },
        }
