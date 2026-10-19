import pytest

from explaincodebase.languages import detect, comment_syntax
from explaincodebase.parsing import parse_file

from .conftest import POLYGLOT


def parse(name):
    path = f"{POLYGLOT}/{name}"
    with open(path) as handle:
        return parse_file(name, handle.read(), detect(name))


@pytest.mark.parametrize("name, language, expected", [
    ("animals.py", "python", ["(global)", "Animal.__init__", "Animal.speak", "Dog.__init__",
                              "Dog.bark", "Dog.bark.loud", "describe"]),
    ("zoo.js", "javascript", ["Zoo.constructor", "Zoo.add", "count", "makeZoo"]),
    ("app.ts", "typescript", ["Service.constructor", "Service.fetch", "join", "main"]),
    ("Shapes.java", "java", ["Shape.Shape", "Shape.area", "Shape.describe", "Square.Square",
                             "Square.area", "Util.square", "Util.run"]),
    ("server.go", "go", ["NewServer", "Server.Start", "Server.listen", "main"]),
    ("lib.rs", "rust", ["Counter.new", "Counter.incr", "step", "run"]),
    ("geometry.cpp", "cpp", ["Point.Point", "Point.norm", "sq", "main"]),
    ("util.c", "c", ["twice", "compute"]),
    ("Bank.cs", "csharp", ["Account.Account", "Account.Deposit", "Account.Add", "Savings.Savings",
                           "Savings.Grow"]),
    ("greeter.rb", "ruby", ["Greeter.initialize", "Greeter.greet", "Greeter.format_name", "main"]),
    ("cart.php", "php", ["(global)", "Cart.add", "Cart.total", "count_items"]),
    ("Main.kt", "kotlin", ["Engine.start", "describe", "main"]),
    ("notes.ex", "elixir", ["Notes.add", "Notes.normalize"]),  # generic tier (tags query)
    ("query.sql", "sql", ["(file)"]),  # file-level fallback
])
def test_units_per_language(name, language, expected):
    source = parse(name)
    assert source.language.name == language
    assert [u.qualname for u in source.units] == expected
    for unit in source.units:
        assert unit.id == f"{name}::{unit.qualname}"
        assert unit.source.strip()


def test_python_details():
    source = parse("animals.py")
    units = {u.qualname: u for u in source.units}
    assert units["Dog.bark.loud"].parent == "animals.py::Dog.bark"
    assert units["Dog.bark.loud"].kind == "function"
    assert units["Dog.bark"].kind == "method"
    calls = [(c.receiver, c.name) for c in units["Dog.__init__"].calls]
    assert calls == [("super()", "__init__")]  # the bare super() is not a separate call
    classes = {c.name: c for c in source.classes}
    assert classes["Dog"].bases == ["Animal"]
    assert classes["Animal"].doc == "Base animal."
    module = units["(global)"]
    assert "dog.bark()" in module.source and "def describe" not in module.source


def test_qualified_calls_split_receiver():
    units = {u.qualname: u for u in parse("geometry.cpp").units}
    assert ("std", "sqrt") in [(c.receiver, c.name) for c in units["Point.norm"].calls]
    units = {u.qualname: u for u in parse("lib.rs").units}
    assert ("Counter", "new") in [(c.receiver, c.name) for c in units["run"].calls]


def test_go_receiver_and_docs():
    source = parse("server.go")
    units = {u.qualname: u for u in source.units}
    assert units["Server.Start"].class_name == "Server"
    assert units["Server.Start"].receiver_var == "s"
    assert "NewServer builds a server" in units["NewServer"].doc
    assert "Server serves things" in source.classes[0].doc


def test_constructor_and_super_calls():
    units = {u.qualname: u for u in parse("Shapes.java").units}
    assert [(c.kind, c.name) for c in units["Util.run"].calls][0] == ("new", "Square")
    assert units["Square.Square"].calls[0].kind == "super"
    units = {u.qualname: u for u in parse("greeter.rb").units}
    assert ("new", "Greeter") in [(c.kind, c.name) for c in units["main"].calls]


def test_duplicate_names_are_disambiguated():
    source = parse_file("p.py", "class A:\n    @property\n    def x(self):\n        return 1\n\n"
                                "    @x.setter\n    def x(self, v):\n        self._x = v\n", detect("p.py"))
    assert sorted(u.id for u in source.units) == ["p.py::A.x@L2", "p.py::A.x@L6"]


def test_anonymous_callbacks_belong_to_their_function():
    source = parse_file("cb.js", "function run(items) {\n  items.forEach((x) => helper(x));\n}\n"
                                 "function helper(x) { return x; }\n", detect("cb.js"))
    units = {u.qualname: u for u in source.units}
    assert set(units) == {"run", "helper"}
    assert "helper" in [c.name for c in units["run"].calls]


def test_detection_skips_data_files():
    assert detect("README.md") is None
    assert detect("data.json") is None
    assert detect("notes.txt") is None
    assert detect("x.py").name == "python"
    assert detect("x.hpp").name == "cpp"


def test_comment_syntax():
    assert comment_syntax("python") == ("#", "")
    assert comment_syntax("go") == ("//", "")
    assert comment_syntax("lua") == ("--", "")
    assert comment_syntax("ocaml") == ("(*", " *)")


def test_same_line_duplicates_get_unique_ids():
    source = parse_file("m.js", "function f(){} function f(){}\n", detect("m.js"))
    ids = [u.id for u in source.units]
    assert len(ids) == len(set(ids)) == 2
