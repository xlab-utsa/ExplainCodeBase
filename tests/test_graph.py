from .conftest import graph_from


def edges(graph):
    return {(src, dst) for src, targets in graph.edges.items() for dst in targets}


def test_inheritance_sample(inheritance_graph):
    assert edges(inheritance_graph) == {
        ("main.py::(global)", "main.py::main"),
        ("main.py::main", "derived_class.py::DerivedClass.__init__"),
        ("main.py::main", "derived_class.py::DerivedClass.display_info"),
        ("derived_class.py::DerivedClass.__init__", "base_class.py::BaseClass.__init__"),
        ("derived_class.py::DerivedClass.display_info", "base_class.py::BaseClass.display_name"),
    }


def test_polyglot_edges(polyglot_graph):
    expected = {
        ("animals.py::Dog.bark", "animals.py::Dog.bark.loud"),
        ("animals.py::Dog.bark", "animals.py::Animal.speak"),
        ("Shapes.java::Util.run", "Shapes.java::Square.Square"),
        ("Shapes.java::Util.run", "Shapes.java::Shape.describe"),
        ("Shapes.java::Square.Square", "Shapes.java::Shape.Shape"),
        ("Shapes.java::Square.area", "Shapes.java::Util.square"),
        ("server.go::main", "server.go::NewServer"),
        ("server.go::main", "server.go::Server.Start"),  # type inferred from NewServer's signature
        ("server.go::Server.Start", "server.go::Server.listen"),
        ("lib.rs::run", "lib.rs::Counter.new"),
        ("lib.rs::run", "lib.rs::Counter.incr"),
        ("geometry.cpp::main", "geometry.cpp::Point.norm"),
        ("geometry.cpp::Point.norm", "geometry.cpp::sq"),
        ("Bank.cs::Savings.Savings", "Bank.cs::Account.Account"),
        ("Bank.cs::Savings.Grow", "Bank.cs::Account.Deposit"),
        ("greeter.rb::main", "greeter.rb::Greeter.initialize"),
        ("greeter.rb::Greeter.greet", "greeter.rb::Greeter.format_name"),
        ("zoo.js::makeZoo", "zoo.js::Zoo.constructor"),
        ("app.ts::Service.fetch", "app.ts::join"),
        ("cart.php::Cart.add", "cart.php::Cart.total"),
        ("Main.kt::main", "Main.kt::Engine.start"),
        ("notes.ex::Notes.add", "notes.ex::Notes.normalize"),
        ("util.c::compute", "util.c::twice"),
    }
    assert expected <= edges(polyglot_graph)


def test_library_calls_stay_unresolved(polyglot_graph):
    unresolved = polyglot_graph.to_json()["unresolved_calls"]
    assert "os.path.join" in unresolved["animals.py::describe"]
    assert "fmt.Println" in unresolved["server.go::Server.Start"]


def test_ambiguous_method_names_do_not_create_edges(tmp_path):
    graph = graph_from(tmp_path, {
        "a.py": "class A:\n    def run(self):\n        pass\n",
        "b.py": "class B:\n    def run(self):\n        pass\n",
        "c.py": "def go(thing):\n    thing.run()\n",
    })
    assert graph.edges["c.py::go"] == []


def test_external_module_does_not_match_repo_function(tmp_path):
    graph = graph_from(tmp_path, {
        "paths.py": "import os\n\ndef join(a, b):\n    return a + b\n\ndef go():\n    return os.path.join('a', 'b')\n",
    })
    assert graph.edges["paths.py::go"] == []


def test_module_qualified_call(tmp_path):
    graph = graph_from(tmp_path, {
        "pkg/utils.py": "def helper():\n    return 1\n",
        "pkg/app.py": "from pkg import utils\n\ndef main():\n    return utils.helper()\n",
    })
    assert graph.edges["pkg/app.py::main"] == ["pkg/utils.py::helper"]


def test_attribute_type_inference(tmp_path):
    graph = graph_from(tmp_path, {
        "m.py": ("class Engine:\n    def start(self):\n        pass\n\n"
                 "class Other:\n    def start(self):\n        pass\n\n"
                 "class Car:\n    def __init__(self):\n        self.engine = Engine()\n\n"
                 "    def drive(self):\n        self.engine.start()\n"),
    })
    assert graph.edges["m.py::Car.drive"] == ["m.py::Engine.start"]


def test_components_are_bottom_up_and_group_cycles(tmp_path):
    graph = graph_from(tmp_path, {
        "r.py": ("def top():\n    a()\n\n"
                 "def a():\n    b()\n\n"
                 "def b():\n    a()\n    leaf()\n\n"
                 "def leaf():\n    leaf()\n"),
    })
    order = graph.components()
    position = {uid: i for i, members in enumerate(order) for uid in members}
    assert ["r.py::a", "r.py::b"] in order
    assert position["r.py::leaf"] < position["r.py::a"] < position["r.py::top"]
    assert sum(len(c) for c in order) == len(graph.units)


def test_deep_call_chain_does_not_recurse(tmp_path):
    n = 3000
    code = "".join(f"def f{i}():\n    f{i + 1}()\n\n" for i in range(n)) + f"def f{n}():\n    pass\n"
    graph = graph_from(tmp_path, {"deep.py": code})
    order = graph.components()
    assert order[0] == [f"deep.py::f{n}"] and order[-1] == ["deep.py::f0"]


def test_graph_json(inheritance_graph):
    data = inheritance_graph.to_json()
    assert data["stats"]["units"] == len(data["units"]) == 6
    assert data["stats"]["languages"] == {"python": 3}


def test_generic_type_arguments_are_not_bases(tmp_path):
    graph = graph_from(tmp_path, {
        "g.py": ("class Mixer:\n    def __init__(self):\n        pass\n\n"
                 "class Config(Generic[Mixer]):\n    pass\n\n"
                 "@register('x')\nclass MixerConfig(Config[Mixer]):\n    pass\n\n"
                 "def build():\n    return MixerConfig()\n"),
    })
    assert graph.classes["MixerConfig"][0].bases == ["Config"]
    assert graph.classes["MixerConfig"][0].header == "class MixerConfig(Config[Mixer]):"
    assert graph.edges["g.py::build"] == []


def test_unique_method_fallback_needs_a_reference_to_the_class(tmp_path):
    graph = graph_from(tmp_path, {
        "store.py": "class WeightStore:\n    def numel(self):\n        return 0\n",
        "use.py": "def count(p):\n    return p.numel()\n",
        "use2.py": "from store import WeightStore\n\ndef count(s):\n    return s.numel()\n",
    })
    assert graph.edges["use.py::count"] == []
    assert graph.edges["use2.py::count"] == ["store.py::WeightStore.numel"]
