"""Language registry: which source files are analysed and how their syntax trees are read.

Three tiers of support:

* **Detailed** languages have a hand-written :class:`LanguageSpec` listing the tree-sitter
  node types for functions, classes and calls, plus the language's ``self``/``super``/
  constructor conventions. These get the most accurate call graphs.
* **Generic** languages have a tree-sitter grammar that ships a ``tags.scm`` query
  (GitHub code-navigation style). Definitions and call references come from that query.
* **Fallback**: any other text source file (or a grammar without a tags query) is
  summarized as a single file-level unit, without a call graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

F = frozenset


@dataclass(frozen=True)
class LanguageSpec:
    name: str
    grammar: str | None
    extensions: tuple[str, ...] = ()
    line_comment: str = "//"
    function_types: frozenset[str] = F()
    class_types: frozenset[str] = F()
    call_types: frozenset[str] = F()
    # Expressions that construct an object, e.g. ``new Foo()``.
    new_types: frozenset[str] = F()
    self_names: frozenset[str] = F({"self", "this"})
    super_names: frozenset[str] = F({"super"})
    # Method names that act as constructors (besides a method named like its class).
    constructor_names: frozenset[str] = F()
    # Java/C#/C++/...: a constructor is a method with the same name as its class.
    class_named_constructor: bool = False
    # A bare ``foo()`` inside a method may call a method of the enclosing class.
    implicit_self: bool = False
    # Python-style docstring as the first statement of a body.
    docstrings: bool = False
    # Definitions/calls come from the grammar's tags query instead of the fields above.
    generic: bool = False

    @property
    def parsed(self) -> bool:
        return self.grammar is not None


DETAILED: tuple[LanguageSpec, ...] = (
    LanguageSpec(
        name="python",
        grammar="python",
        extensions=(".py", ".pyw", ".pyi"),
        line_comment="#",
        function_types=F({"function_definition"}),
        class_types=F({"class_definition"}),
        call_types=F({"call"}),
        self_names=F({"self", "cls"}),
        constructor_names=F({"__init__", "__new__"}),
        docstrings=True,
    ),
    LanguageSpec(
        name="javascript",
        grammar="javascript",
        extensions=(".js", ".mjs", ".cjs", ".jsx"),
        function_types=F({
            "function_declaration", "generator_function_declaration", "method_definition",
            "function_expression", "function", "arrow_function", "generator_function",
        }),
        class_types=F({"class_declaration", "class"}),
        call_types=F({"call_expression"}),
        new_types=F({"new_expression"}),
        constructor_names=F({"constructor"}),
    ),
    LanguageSpec(
        name="typescript",
        grammar="typescript",
        extensions=(".ts", ".mts", ".cts"),
        function_types=F({
            "function_declaration", "generator_function_declaration", "method_definition",
            "function_expression", "arrow_function", "generator_function",
        }),
        class_types=F({"class_declaration", "abstract_class_declaration", "class"}),
        call_types=F({"call_expression"}),
        new_types=F({"new_expression"}),
        constructor_names=F({"constructor"}),
    ),
    LanguageSpec(
        name="tsx",
        grammar="tsx",
        extensions=(".tsx",),
        function_types=F({
            "function_declaration", "generator_function_declaration", "method_definition",
            "function_expression", "arrow_function", "generator_function",
        }),
        class_types=F({"class_declaration", "abstract_class_declaration", "class"}),
        call_types=F({"call_expression"}),
        new_types=F({"new_expression"}),
        constructor_names=F({"constructor"}),
    ),
    LanguageSpec(
        name="java",
        grammar="java",
        extensions=(".java",),
        function_types=F({"method_declaration", "constructor_declaration", "compact_constructor_declaration"}),
        class_types=F({
            "class_declaration", "interface_declaration", "enum_declaration",
            "record_declaration", "annotation_type_declaration",
        }),
        call_types=F({"method_invocation", "explicit_constructor_invocation"}),
        new_types=F({"object_creation_expression"}),
        class_named_constructor=True,
        implicit_self=True,
    ),
    LanguageSpec(
        name="go",
        grammar="go",
        extensions=(".go",),
        function_types=F({"function_declaration", "method_declaration", "func_literal"}),
        class_types=F(),  # methods attach to types through their receiver
        call_types=F({"call_expression"}),
        self_names=F(),  # the receiver variable is named per method
        super_names=F(),
    ),
    LanguageSpec(
        name="rust",
        grammar="rust",
        extensions=(".rs",),
        function_types=F({"function_item", "closure_expression"}),
        class_types=F({"impl_item", "trait_item"}),
        call_types=F({"call_expression"}),
        self_names=F({"self", "Self"}),
        super_names=F(),
        constructor_names=F({"new"}),
    ),
    LanguageSpec(
        name="c",
        grammar="c",
        extensions=(".c", ".h"),
        function_types=F({"function_definition"}),
        call_types=F({"call_expression"}),
        self_names=F(),
        super_names=F(),
    ),
    LanguageSpec(
        name="cpp",
        grammar="cpp",
        extensions=(".cpp", ".cc", ".cxx", ".c++", ".hpp", ".hh", ".hxx", ".h++", ".ipp", ".tpp"),
        function_types=F({"function_definition", "lambda_expression"}),
        class_types=F({"class_specifier", "struct_specifier", "union_specifier"}),
        call_types=F({"call_expression"}),
        new_types=F({"new_expression"}),
        self_names=F({"this"}),
        super_names=F(),
        class_named_constructor=True,
        implicit_self=True,
    ),
    LanguageSpec(
        name="csharp",
        grammar="csharp",
        extensions=(".cs",),
        function_types=F({
            "method_declaration", "constructor_declaration", "local_function_statement",
            "operator_declaration", "destructor_declaration",
        }),
        class_types=F({
            "class_declaration", "struct_declaration", "interface_declaration",
            "record_declaration", "record_struct_declaration",
        }),
        call_types=F({"invocation_expression", "constructor_initializer"}),
        new_types=F({"object_creation_expression"}),
        self_names=F({"this"}),
        super_names=F({"base"}),
        class_named_constructor=True,
        implicit_self=True,
    ),
    LanguageSpec(
        name="ruby",
        grammar="ruby",
        extensions=(".rb", ".rake", ".gemspec"),
        line_comment="#",
        function_types=F({"method", "singleton_method"}),
        class_types=F({"class", "module", "singleton_class"}),
        call_types=F({"call"}),
        self_names=F({"self"}),
        constructor_names=F({"initialize"}),
        implicit_self=True,
    ),
    LanguageSpec(
        name="php",
        grammar="php",
        extensions=(".php", ".phtml"),
        function_types=F({
            "function_definition", "method_declaration", "anonymous_function", "arrow_function",
        }),
        class_types=F({"class_declaration", "trait_declaration", "interface_declaration", "enum_declaration"}),
        call_types=F({
            "function_call_expression", "member_call_expression", "scoped_call_expression",
            "nullsafe_member_call_expression",
        }),
        new_types=F({"object_creation_expression"}),
        self_names=F({"this", "$this", "self", "static"}),
        super_names=F({"parent"}),
        constructor_names=F({"__construct"}),
    ),
    LanguageSpec(
        name="kotlin",
        grammar="kotlin",
        extensions=(".kt", ".kts"),
        function_types=F({"function_declaration", "secondary_constructor", "anonymous_function", "lambda_literal"}),
        class_types=F({"class_declaration", "object_declaration", "companion_object"}),
        call_types=F({"call_expression"}),
        constructor_names=F({"init"}),
        implicit_self=True,
    ),
    LanguageSpec(
        name="scala",
        grammar="scala",
        extensions=(".scala", ".sc"),
        function_types=F({"function_definition"}),
        class_types=F({"class_definition", "object_definition", "trait_definition"}),
        call_types=F({"call_expression"}),
        new_types=F({"instance_expression"}),
        constructor_names=F({"this"}),
        implicit_self=True,
    ),
    LanguageSpec(
        name="swift",
        grammar="swift",
        extensions=(".swift",),
        function_types=F({"function_declaration", "init_declaration", "deinit_declaration"}),
        class_types=F({"class_declaration", "protocol_declaration"}),
        call_types=F({"call_expression"}),
        constructor_names=F({"init"}),
        implicit_self=True,
    ),
    LanguageSpec(
        name="lua",
        grammar="lua",
        extensions=(".lua",),
        line_comment="--",
        function_types=F({"function_declaration", "function_definition"}),
        call_types=F({"function_call"}),
        self_names=F({"self"}),
        super_names=F(),
    ),
)

# Grammars whose files are data, markup or prose rather than code with functions.
NON_CODE_GRAMMARS = F({
    "markdown", "markdown_inline", "json", "json5", "jsonc", "jsonnet", "yaml", "toml", "xml",
    "dtd", "csv", "tsv", "psv", "vimdoc", "ini", "properties", "gitattributes", "gitignore",
    "gitcommit", "git_config", "git_rebase", "diff", "html", "css", "scss", "less", "svg",
    "requirements", "comment", "doxygen", "regex", "jsdoc", "luadoc", "rst", "latex", "bibtex",
    "po", "pem", "hcl", "dockerfile", "make", "cmake", "ssh_config", "editorconfig", "mermaid",
})

# Line-comment syntax for grammars without a detailed spec (default ``//``).
LINE_COMMENTS = {
    **{g: "#" for g in (
        "bash", "fish", "zsh", "perl", "r", "julia", "elixir", "nim", "crystal", "powershell",
        "tcl", "gdscript", "starlark", "nix", "make", "cmake", "awk", "coffee", "hcl", "terraform",
    )},
    **{g: "--" for g in ("haskell", "sql", "elm", "purescript", "agda", "ada", "vhdl", "teal")},
    **{g: "%" for g in ("erlang", "matlab", "prolog", "latex")},
    **{g: ";" for g in ("clojure", "commonlisp", "scheme", "racket", "fennel", "elisp", "janet")},
    "fortran": "!",
    "vim": '"',
    "ocaml": "(*",
    "fsharp": "//",
}

_BY_NAME = {spec.name: spec for spec in DETAILED}
_BY_EXTENSION = {ext: spec for spec in DETAILED for ext in spec.extensions}

# Text files we never treat as source code.
SKIP_EXTENSIONS = F({
    ".md", ".markdown", ".rst", ".txt", ".json", ".yaml", ".yml", ".toml", ".ini", ".cfg",
    ".lock", ".csv", ".tsv", ".xml", ".svg", ".html", ".htm", ".css", ".map", ".log",
    ".ipynb", ".pyc", ".min.js",
})


def detailed(name: str) -> LanguageSpec | None:
    return _BY_NAME.get(name)


@lru_cache(maxsize=None)
def _generic_spec(grammar: str) -> LanguageSpec:
    return LanguageSpec(
        name=grammar,
        grammar=grammar if _has_tags_query(grammar) else None,
        line_comment=LINE_COMMENTS.get(grammar, "//"),
        generic=True,
    )


def _has_tags_query(grammar: str) -> bool:
    try:
        import tree_sitter_language_pack as tslp

        return bool(tslp.get_tags_query(grammar))
    except Exception:
        return False


def detect(path: str) -> LanguageSpec | None:
    """Return the spec for a source file, or ``None`` if it is not a code file."""
    lower = path.lower()
    if any(lower.endswith(ext) for ext in SKIP_EXTENSIONS):
        return None
    ext = os.path.splitext(lower)[1]
    if ext in _BY_EXTENSION:
        return _BY_EXTENSION[ext]
    try:
        import tree_sitter_language_pack as tslp

        grammar = tslp.detect_language_from_path(path)
    except Exception:
        grammar = None
    if not grammar or grammar in NON_CODE_GRAMMARS:
        return None
    return _BY_NAME.get(grammar) or _generic_spec(grammar)


def comment_syntax(language: str) -> tuple[str, str]:
    """``(opening, closing)`` strings for a one-line comment in ``language``."""
    spec = _BY_NAME.get(language)
    opener = spec.line_comment if spec else LINE_COMMENTS.get(language, "//")
    return (opener, " *)") if opener == "(*" else (opener, "")


def supported_languages() -> list[tuple[str, str]]:
    """``(name, tier)`` pairs describing what this installation can parse."""
    rows = [(spec.name, "detailed") for spec in DETAILED]
    try:
        import tree_sitter_language_pack as tslp

        names = tslp.manifest_languages()
    except Exception:
        names = []
    for grammar in sorted(set(names) - set(_BY_NAME) - NON_CODE_GRAMMARS):
        rows.append((grammar, "generic (tags query)" if _has_tags_query(grammar) else "file-level"))
    return rows
