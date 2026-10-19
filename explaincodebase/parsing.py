"""Turn source files into code units (functions, methods, module code) and call sites."""

from __future__ import annotations

import fnmatch
import logging
import os
from dataclasses import dataclass, field
from functools import lru_cache

from .languages import LanguageSpec, detect, detailed

log = logging.getLogger(__name__)

SKIP_DIRS = frozenset({
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "bower_components", "venv", ".venv",
    "env", ".env", ".tox", ".nox", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea",
    ".vscode", "build", "dist", "target", "out", "bin", "obj", ".gradle", ".next", ".nuxt",
    "vendor", "third_party", "site-packages", ".eggs", "coverage", ".terraform",
})
MAX_FILE_BYTES = 1_000_000
MIN_FILE_UNIT_CHARS = 40

GLOBAL = "(global)"
FILE = "(file)"


@dataclass
class CallSite:
    name: str
    receiver: str | None
    line: int
    # "call": ordinary call, "new": constructs ``name``, "super": the base-class version of
    # the calling method (or the base constructor when called from a constructor).
    kind: str = "call"


@dataclass
class ClassInfo:
    name: str
    qualname: str
    file: str
    language: str
    start_line: int
    end_line: int
    bases: list[str]
    doc: str | None
    header: str


@dataclass
class CodeUnit:
    id: str
    name: str
    qualname: str
    kind: str  # "function" | "method" | "module" | "file"
    file: str
    language: str
    start_line: int
    end_line: int
    # (original line number, text) pairs, so call sites can be located in module units
    # stitched together from several top-level fragments.
    lines: list[tuple[int, str]]
    class_name: str | None = None
    parent: str | None = None
    doc: str | None = None
    receiver_var: str | None = None
    calls: list[CallSite] = field(default_factory=list)

    @property
    def source(self) -> str:
        return "\n".join(text for _, text in self.lines)


@dataclass
class SourceFile:
    path: str  # relative, POSIX separators
    language: LanguageSpec
    text: str
    units: list[CodeUnit]
    classes: list[ClassInfo]
    imports: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- discovery


def discover_files(root: str, languages: set[str] | None = None,
                   exclude: list[str] | None = None) -> list[tuple[str, LanguageSpec]]:
    """Return ``(relative_path, spec)`` for every source file under ``root``."""
    exclude = exclude or []
    found = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.startswith("."))
        for filename in sorted(filenames):
            full = os.path.join(dirpath, filename)
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if any(fnmatch.fnmatch(rel, pattern) for pattern in exclude):
                continue
            spec = detect(filename)
            if spec is None or (languages and spec.name not in languages):
                continue
            try:
                if os.path.getsize(full) > MAX_FILE_BYTES:
                    log.warning("Skipping %s: larger than %d bytes", rel, MAX_FILE_BYTES)
                    continue
            except OSError:
                continue
            found.append((rel, spec))
    return found


def read_text(path: str) -> str | None:
    try:
        with open(path, "rb") as handle:
            data = handle.read()
    except OSError as exc:
        log.warning("Cannot read %s: %s", path, exc)
        return None
    if b"\0" in data[:8192]:
        return None  # binary
    return data.decode("utf-8", errors="replace").replace("\r\n", "\n")


# --------------------------------------------------------------------------- tree helpers

IDENT_TYPES = frozenset({
    "identifier", "type_identifier", "field_identifier", "property_identifier",
    "simple_identifier", "constant", "name", "namespace_identifier", "package_identifier",
    "shorthand_property_identifier", "private_property_identifier", "destructor_name",
    "operator_name", "variable_name", "word", "atom", "symbol",
})
MEMBER_ACCESS_TYPES = frozenset({
    "attribute", "member_expression", "field_expression", "navigation_expression",
    "selector_expression", "member_access_expression", "dot_index_expression",
})
MEMBER_FIELDS = ("property", "attribute", "field", "name", "method", "suffix", "rhs")
RECEIVER_FIELDS = ("object", "value", "operand", "argument", "expression", "table", "target",
                   "scope", "path", "receiver", "lhs")
# Ancestors through which an anonymous function can receive a name (``const f = () => …``).
NAMING_PARENTS = frozenset({
    "variable_declarator", "assignment", "assignment_expression", "assignment_statement",
    "pair", "public_field_definition", "field_definition", "property_declaration",
    "variable_declaration", "expression_list", "variable_list", "short_var_declaration",
    "let_declaration", "init_declarator", "lexical_declaration", "local_variable_declaration",
    "val_definition", "var_definition", "augmented_assignment",
})
NAMING_FIELDS = ("name", "left", "key", "property", "pattern")
WRAPPER_TYPES = frozenset({"decorated_definition", "export_statement", "template_declaration",
                           "type_declaration"})
BASE_CONTAINERS = frozenset({
    "superclass", "superclasses", "super_interfaces", "interfaces", "base_list",
    "base_class_clause", "class_heritage", "extends_clause", "implements_clause",
    "delegation_specifier", "inheritance_specifier", "base_clause", "extends_type_clause",
    "extends_interfaces", "class_interface_clause",
})
SKIP_BASE_TYPES = frozenset({"keyword_argument", "access_specifier", "arguments",
                             "value_arguments", "argument_list", "type_arguments"})


def text_of(node) -> str:
    return node.text.decode("utf-8", errors="replace") if node is not None else ""


def is_ident(node) -> bool:
    ntype = node.type
    return ntype in IDENT_TYPES or (
        ntype.endswith("identifier") and not ntype.startswith(("qualified", "scoped")))


def last_ident(node) -> str | None:
    """Text of the right-most identifier inside ``node`` (``a.b.c`` -> ``c``)."""
    if node is None:
        return None
    if is_ident(node):
        return text_of(node).lstrip("$")
    for child in reversed(node.named_children):
        found = last_ident(child)
        if found:
            return found
    return None


def first_ident_child(node):
    for child in node.named_children:
        if is_ident(child):
            return child
    return None


def field_node(node, *names):
    for name in names:
        child = node.child_by_field_name(name)
        if child is not None:
            return child
    return None


@lru_cache(maxsize=None)
def _parser(grammar: str):
    import tree_sitter_language_pack as tslp

    return tslp.get_parser(grammar)


@lru_cache(maxsize=None)
def _tags_query(grammar: str):
    import tree_sitter as ts
    import tree_sitter_language_pack as tslp

    return ts.Query(tslp.get_language(grammar), tslp.get_tags_query(grammar))


# --------------------------------------------------------------------------- raw items


@dataclass
class _RawDef:
    name: str
    start_byte: int
    end_byte: int
    node: object  # the span used for source text (includes decorators)
    scope: str | None = None  # class given explicitly, e.g. ``void Dog::run()`` or a Go receiver
    receiver_var: str | None = None
    doc: str | None = None


@dataclass
class _RawClass:
    name: str
    start_byte: int
    end_byte: int
    node: object
    bases: list[str]
    doc: str | None
    header_row: int | None = None  # line of the ``class`` keyword (after any decorators)


def _definition_name(node, spec: LanguageSpec) -> tuple[str | None, str | None]:
    """Return ``(name, explicit_scope)`` for a function node."""
    name_node = field_node(node, "name")
    if name_node is not None:
        if name_node.type in ("dot_index_expression", "method_index_expression"):  # Lua M.f / M:f
            table = field_node(name_node, "table")
            member = field_node(name_node, "field", "method")
            return last_ident(member), last_ident(table)
        if name_node.type == "scope_resolution":  # Ruby def Foo::bar
            return last_ident(field_node(name_node, "name")), last_ident(field_node(name_node, "scope"))
        return last_ident(name_node) or text_of(name_node), None
    declarator = field_node(node, "declarator")
    while declarator is not None:  # C/C++: function_declarator -> identifier/qualified_identifier
        if declarator.type == "qualified_identifier":
            return last_ident(field_node(declarator, "name")), last_ident(field_node(declarator, "scope"))
        if is_ident(declarator):
            return text_of(declarator), None
        nxt = field_node(declarator, "declarator")
        if nxt is None:
            return last_ident(declarator), None
        declarator = nxt
    special = {"init_declaration": "init", "deinit_declaration": "deinit",
               "secondary_constructor": "constructor", "destructor_declaration": "~destructor"}
    if node.type in special:
        return special[node.type], None
    if node.type in ("function_definition", "method_declaration", "function_declaration"):
        ident = first_ident_child(node)  # Kotlin: no ``name`` field
        if ident is not None:
            return text_of(ident), None
    return _name_from_parent(node), None


def _name_from_parent(node) -> str | None:
    parent = node.parent
    for _ in range(3):
        if parent is None or parent.type not in NAMING_PARENTS:
            return None
        target = field_node(parent, *NAMING_FIELDS)
        if target is not None and target.id != node.id:
            name = last_ident(target)
            if name:
                return name
        parent = parent.parent
    return None


def _go_receiver(node) -> tuple[str | None, str | None]:
    receiver = field_node(node, "receiver")
    if receiver is None:
        return None, None
    var, typ = None, None
    for param in receiver.named_children:
        var_node = field_node(param, "name")
        type_node = field_node(param, "type")
        var = text_of(var_node) if var_node is not None else None
        typ = last_ident(type_node)
    return typ, var


def _class_name(node) -> str | None:
    if node.type == "impl_item":
        return last_ident(field_node(node, "type"))
    name_node = field_node(node, "name")
    if name_node is not None:
        return last_ident(name_node) or text_of(name_node)
    ident = first_ident_child(node)
    if ident is not None:
        return text_of(ident)
    if node.type == "singleton_class":
        return "self"
    return None


def _class_bases(node) -> list[str]:
    bases: list[str] = []
    if node.type == "impl_item":
        trait = field_node(node, "trait")
        return [last_ident(trait)] if trait is not None else []

    def collect(container):
        stack = [container]
        while stack:
            current = stack.pop()
            if current.type in SKIP_BASE_TYPES and current is not container:
                continue
            if current.type == "subscript":  # Base[T]: only Base is a base class
                name = last_ident(field_node(current, "value"))
                if name and name not in bases:
                    bases.append(name)
                continue
            if is_ident(current):
                name = text_of(current)
                if name and name not in bases and name not in ("public", "private", "protected", "virtual"):
                    bases.append(name)
                continue
            if current.type in ("attribute", "member_expression", "scoped_identifier",
                                "qualified_identifier", "scope_resolution", "qualified_name",
                                "navigation_expression", "user_type", "generic_type",
                                "constructor_invocation"):
                name = _type_head(current)
                if name and name not in bases:
                    bases.append(name)
                continue
            stack.extend(reversed(current.named_children))

    for i, child in enumerate(node.children):
        field_name = node.field_name_for_child(i)
        if child.type in BASE_CONTAINERS or field_name in ("superclass", "superclasses", "bases", "interfaces"):
            collect(child)
    return bases


def _type_head(node) -> str | None:
    """Base type name of ``pkg.Base``, ``Base<T>``, ``Base(args)``."""
    if node.type in ("generic_type", "user_type", "constructor_invocation"):
        for child in node.named_children:
            if is_ident(child):
                return text_of(child)
            if child.type in ("user_type", "scoped_type_identifier", "generic_type"):
                return _type_head(child)
    return last_ident(field_node(node, "name", "attribute", "property", "field")) or last_ident(node)


def _leading_comments(node) -> str | None:
    target = node
    while target.parent is not None and target.parent.type in WRAPPER_TYPES:
        target = target.parent
    comments = []
    expected_end = target.start_point[0]
    prev = target.prev_sibling
    while prev is not None:
        if "comment" in prev.type:
            if prev.end_point[0] < expected_end - 1:
                break
            comments.append(text_of(prev))
            expected_end = prev.start_point[0]
        elif prev.type not in ("attribute_item", "decorator", "annotation", "marker_annotation"):
            break
        prev = prev.prev_sibling
    return "\n".join(reversed(comments)) or None


def _python_docstring(node) -> str | None:
    body = field_node(node, "body")
    if body is None or not body.named_children:
        return None
    first = body.named_children[0]
    if first.type == "expression_statement" and first.named_children:
        first = first.named_children[0]
    if first.type == "string":
        return text_of(first).strip("\"' \n") or None
    return None


def _doc(node, spec: LanguageSpec) -> str | None:
    if spec.docstrings:
        doc = _python_docstring(node)
        if doc:
            return doc
    return _leading_comments(node)


def _span_node(node):
    """Include decorators / ``export`` / ``template<>`` wrappers in a definition's source."""
    while node.parent is not None and node.parent.type in WRAPPER_TYPES:
        node = node.parent
    return node


# --------------------------------------------------------------------------- calls


def _call_site(node, spec: LanguageSpec) -> CallSite | None:
    line = node.start_point[0] + 1
    ntype = node.type
    if ntype in spec.new_types:
        target = field_node(node, "constructor", "type", "name") or next(
            (c for c in node.named_children if "argument" not in c.type), None)
        name = _type_head(target) if target is not None else None
        return CallSite(name, None, line, "new") if name else None
    if ntype == "explicit_constructor_invocation":  # Java this(...) / super(...)
        ctor = text_of(field_node(node, "constructor"))
        return CallSite("super" if ctor == "super" else "this", None, line, "super" if ctor == "super" else "call")
    if ntype == "constructor_initializer":  # C# : base(...) / : this(...)
        return CallSite("base", None, line, "super") if "base" in text_of(node).split("(")[0] else None

    name_node = field_node(node, "method") if ntype == "call" else None  # Ruby
    if name_node is None and field_node(node, "function") is None:
        name_node = field_node(node, "name")  # Java, PHP member/scoped calls, Lua
        if name_node is not None and name_node.type in ("dot_index_expression", "method_index_expression"):
            return _callee(name_node, line, spec)
    if name_node is not None:
        receiver = field_node(node, "object", "receiver", "scope")
        name = text_of(name_node)
        if name_node.type == "super" or (receiver is None and name in spec.super_names):
            return CallSite(name, None, line, "super")
        rtext = _receiver_text(receiver)
        if rtext is not None and spec.name == "ruby" and name == "new":
            return CallSite(last_ident(receiver) or rtext, None, line, "new")
        return CallSite(name, rtext, line)

    callee = field_node(node, "function")
    if callee is None:
        callee = next((c for c in node.named_children
                       if "argument" not in c.type and "suffix" not in c.type), None)
    if callee is None:
        return None
    return _callee(callee, line, spec)


def _callee(expr, line: int, spec: LanguageSpec) -> CallSite | None:
    if expr.type in ("super", "super_expression") or (is_ident(expr) and text_of(expr) in spec.super_names):
        return CallSite(text_of(expr), None, line, "super")
    if is_ident(expr):
        return CallSite(text_of(expr).lstrip("$"), None, line)
    if expr.type in ("template_function", "generic_name", "generic_function", "instantiation_expression"):
        inner = field_node(expr, "name", "function") or first_ident_child(expr)
        return CallSite(last_ident(inner), None, line) if inner is not None and last_ident(inner) else None
    member = field_node(expr, *MEMBER_FIELDS)
    receiver = field_node(expr, *RECEIVER_FIELDS)
    if member is None and len(expr.named_children) >= 2:
        receiver, member = expr.named_children[0], expr.named_children[-1]
    name = last_ident(member) if member is not None else last_ident(expr)
    if not name:
        return None
    return CallSite(name, _receiver_text(receiver), line)


def _receiver_text(node) -> str | None:
    if node is None:
        return None
    text = " ".join(text_of(node).split())
    return text[:200].lstrip("$")


# --------------------------------------------------------------------------- extraction


def _walk_detailed(root, spec: LanguageSpec):
    defs, classes, calls = [], [], []
    stack = [root]
    while stack:
        node = stack.pop()
        ntype = node.type
        if ntype in spec.function_types:
            name, scope = _definition_name(node, spec)
            if name:
                receiver_var = None
                if ntype == "method_declaration" and spec.name == "go":
                    scope, receiver_var = _go_receiver(node)
                span = _span_node(node)
                defs.append(_RawDef(name, span.start_byte, span.end_byte, span, scope,
                                    receiver_var, _doc(node, spec)))
        elif ntype in spec.class_types or (spec.name == "go" and ntype == "type_spec"):
            name = _class_name(node)
            if name:
                span = _span_node(node)
                classes.append(_RawClass(name, span.start_byte, span.end_byte, span,
                                         _class_bases(node), _doc(node, spec), node.start_point[0]))
        if ntype in spec.call_types or ntype in spec.new_types:
            try:
                site = _call_site(node, spec)
            except Exception:  # never let one odd construct abort the file
                log.debug("Could not read call at %s:%d", ntype, node.start_point[0] + 1, exc_info=True)
                site = None
            if site is not None and site.kind == "super" and node.parent is not None \
                    and node.parent.type in MEMBER_ACCESS_TYPES:
                site = None  # ``super()`` used as a receiver, as in ``super().__init__()``
            if site is not None and site.name:
                calls.append((node.start_byte, site))
        stack.extend(reversed(node.named_children))
    return defs, classes, calls


def _walk_generic(root, spec: LanguageSpec):
    import tree_sitter as ts

    defs, classes, calls = [], [], []
    # In some grammars (e.g. Elixir) a definition is itself a call node: ``def name(args)``.
    definition_spans: set[tuple[int, int]] = set()
    cursor = ts.QueryCursor(_tags_query(spec.grammar))
    for _, captures in cursor.matches(root):
        if any(key.startswith("definition.") for key in captures):
            for nodes in captures.values():
                definition_spans.update((n.start_byte, n.end_byte) for n in nodes)
    for _, captures in cursor.matches(root):
        names = captures.get("name") or []
        name_node = names[0] if names else None
        for key, nodes in captures.items():
            node = nodes[0]
            if key in ("definition.function", "definition.method") and name_node is not None:
                span = _span_node(node)
                defs.append(_RawDef(text_of(name_node), span.start_byte, span.end_byte, span,
                                    doc=_leading_comments(node)))
            elif key in ("definition.class", "definition.interface", "definition.module",
                         "definition.type") and name_node is not None:
                classes.append(_RawClass(text_of(name_node), node.start_byte, node.end_byte, node,
                                         _class_bases(node), _leading_comments(node)))
            elif key in ("reference.call", "reference.send") and name_node is not None:
                if (node.start_byte, node.end_byte) in definition_spans or \
                        (name_node.start_byte, name_node.end_byte) in definition_spans:
                    continue
                callee_text = text_of(node)[: name_node.start_byte - node.start_byte]
                receiver = callee_text.rstrip(".:->@ \n\t") or None
                calls.append((node.start_byte, CallSite(text_of(name_node), receiver,
                                                        node.start_point[0] + 1)))
    return defs, classes, calls


def _imports(root) -> list[str]:
    found = []
    for child in root.named_children:
        if "import" in child.type or child.type in ("use_declaration", "using_directive",
                                                     "preproc_include", "require_call"):
            found.append(" ".join(text_of(child).split())[:200])
    return found


def _node_lines(node, text_lines: list[str]) -> list[tuple[int, str]]:
    """Whole source lines covered by ``node``, keeping their original indentation."""
    start, end = node.start_point[0], min(node.end_point[0], len(text_lines) - 1)
    return [(i + 1, text_lines[i]) for i in range(start, end + 1)]


def parse_file(rel_path: str, text: str, spec: LanguageSpec) -> SourceFile:
    text_lines = text.split("\n")
    if spec.grammar is None:
        return _file_level(rel_path, text, text_lines, spec)
    try:
        tree = _parser(spec.grammar).parse(text.encode("utf-8"))
    except Exception as exc:
        log.warning("Cannot parse %s as %s (%s); summarizing it as one unit", rel_path, spec.name, exc)
        return _file_level(rel_path, text, text_lines, spec)
    root = tree.root_node
    walk = _walk_generic if spec.generic else _walk_detailed
    defs, raw_classes, calls = walk(root, spec)
    source = _assemble(rel_path, spec, root, text_lines, defs, raw_classes, calls)
    source.imports = _imports(root)
    if not source.units and len(text.strip()) >= MIN_FILE_UNIT_CHARS:
        fallback = _file_level(rel_path, text, text_lines, spec)
        source.units = fallback.units
        source.units[0].calls = [site for _, site in calls]
    return source


def _file_level(rel_path, text, text_lines, spec) -> SourceFile:
    units = []
    if len(text.strip()) >= MIN_FILE_UNIT_CHARS:
        units.append(CodeUnit(
            id=f"{rel_path}::{FILE}", name=FILE, qualname=FILE, kind="file", file=rel_path,
            language=spec.name, start_line=1, end_line=len(text_lines),
            lines=[(i + 1, line) for i, line in enumerate(text_lines)],
        ))
    return SourceFile(rel_path, spec, text, units, [])


def _assemble(rel_path, spec, root, text_lines, defs, raw_classes, calls) -> SourceFile:
    # A tags query can match one definition through several patterns.
    defs = list({(d.start_byte, d.end_byte, d.name): d for d in defs}.values())
    raw_classes = list({(c.start_byte, c.end_byte, c.name): c for c in raw_classes}.values())
    # Sorted so that enclosing items come before the items they contain.
    items = [("class", c.start_byte, c.end_byte, c) for c in raw_classes]
    items += [("def", d.start_byte, d.end_byte, d) for d in defs]
    items.sort(key=lambda item: (item[1], -item[2], 0 if item[0] == "class" else 1))
    calls.sort(key=lambda item: item[0])

    units: list[CodeUnit] = []
    parents: dict[int, CodeUnit] = {}
    classes: list[ClassInfo] = []
    global_calls: list[CallSite] = []
    stack: list[tuple[str, int, object, object]] = []  # (kind, end_byte, raw, built)
    pending = iter(calls)
    next_call = next(pending, None)

    def assign_calls(upto: float) -> None:
        nonlocal next_call
        while next_call is not None and next_call[0] < upto:
            position, site = next_call
            open_items = [entry for entry in stack if entry[1] > position]
            owner = next((entry[3] for entry in reversed(open_items) if entry[0] == "def"), None)
            (owner.calls if owner is not None else global_calls).append(site)
            next_call = next(pending, None)

    for kind, start, end, raw in items:
        assign_calls(start)
        while stack and stack[-1][1] <= start:
            stack.pop()
        scope_names = [entry[3].name for entry in stack]
        node = raw.node
        if kind == "class":
            row = raw.header_row if raw.header_row is not None else node.start_point[0]
            header = text_lines[row].strip() if row < len(text_lines) else ""
            info = ClassInfo(raw.name, ".".join(scope_names + [raw.name]), rel_path, spec.name,
                             node.start_point[0] + 1, node.end_point[0] + 1, raw.bases, raw.doc, header)
            classes.append(info)
            stack.append(("class", end, raw, info))
            continue
        direct_class = stack[-1][3] if stack and stack[-1][0] == "class" else None
        enclosing_def = next((entry[3] for entry in reversed(stack) if entry[0] == "def"), None)
        class_name = raw.scope or (direct_class.name if direct_class is not None else None)
        if raw.scope and (not scope_names or scope_names[-1] != raw.scope):
            scope_names.append(raw.scope)
        qualname = ".".join(scope_names + [raw.name])
        unit = CodeUnit(
            id=f"{rel_path}::{qualname}", name=raw.name, qualname=qualname,
            kind="method" if class_name else "function", file=rel_path, language=spec.name,
            start_line=node.start_point[0] + 1, end_line=node.end_point[0] + 1,
            lines=_node_lines(node, text_lines), class_name=class_name, doc=raw.doc,
            receiver_var=raw.receiver_var,
        )
        if enclosing_def is not None:
            parents[id(unit)] = enclosing_def
        units.append(unit)
        stack.append(("def", end, raw, unit))
    assign_calls(float("inf"))

    _dedupe_ids(units)
    for unit in units:
        if id(unit) in parents:
            unit.parent = parents[id(unit)].id
    if global_calls:
        units.insert(0, _module_unit(rel_path, spec, root, text_lines, global_calls, defs, raw_classes))
    return SourceFile(rel_path, spec, "\n".join(text_lines), units, classes)


def _dedupe_ids(units: list[CodeUnit]) -> None:
    """Overloads, property getters/setters etc. share a name: disambiguate by line."""
    by_id: dict[str, list[CodeUnit]] = {}
    for unit in units:
        by_id.setdefault(unit.id, []).append(unit)
    for group in by_id.values():
        if len(group) > 1:
            for unit in group:
                unit.id = f"{unit.id}@L{unit.start_line}"
            seen: dict[str, int] = {}
            for unit in group:  # several definitions on one line (e.g. minified code)
                count = seen.get(unit.id, 0)
                seen[unit.id] = count + 1
                if count:
                    unit.id = f"{unit.id}#{count + 1}"


def _module_unit(rel_path, spec, root, text_lines, global_calls, defs, raw_classes) -> CodeUnit:
    """Top-level code that is not part of any function or class definition."""
    covered = sorted((d.node.start_point[0], d.node.end_point[0]) for d in list(defs) + list(raw_classes))
    lines = []
    for child in root.named_children:
        start, end = child.start_point[0], child.end_point[0]
        if "comment" in child.type or "import" in child.type or child.type in (
                "package_clause", "package_declaration", "use_declaration", "using_directive",
                "preproc_include", "namespace_use_declaration", "php_tag"):
            continue
        if any(s <= start and end <= e for s, e in covered):
            continue
        lines.extend(_node_lines(child, text_lines))
    if not lines:
        lines = [(site.line, text_lines[site.line - 1]) for site in global_calls
                 if 0 < site.line <= len(text_lines)]
    return CodeUnit(
        id=f"{rel_path}::{GLOBAL}", name=GLOBAL, qualname=GLOBAL, kind="module", file=rel_path,
        language=spec.name, start_line=lines[0][0] if lines else 1,
        end_line=lines[-1][0] if lines else 1, lines=lines, calls=global_calls,
    )


def parse_codebase(root: str, languages: set[str] | None = None,
                   exclude: list[str] | None = None) -> list[SourceFile]:
    files = []
    for rel, spec in discover_files(root, languages, exclude):
        text = read_text(os.path.join(root, rel))
        if text is None:
            continue
        files.append(parse_file(rel, text, spec))
    return files


__all__ = [
    "CallSite", "ClassInfo", "CodeUnit", "SourceFile", "GLOBAL", "FILE",
    "discover_files", "parse_file", "parse_codebase", "detailed",
]
