import json
import re
from collections import Counter

from explaincodebase.llm import LLM
from explaincodebase.methods import METHODS, Settings, annotate_calls, parse_baseline

from .conftest import RecordingProvider, graph_from

DIAMOND = {
    "d.py": ("def a():\n    c()\n    b()\n\n"
             "def b():\n    return 1\n\n"
             "def c():\n    b()\n"),
}


def subject(prompt):
    match = re.search(r"^# Code: .*?`(.+?)`", prompt, re.M)
    return match.group(1) if match else None


def test_each_unit_summarized_once_in_dependency_order(tmp_path):
    # The original code summarized b twice here (and fed b its own earlier summary).
    graph = graph_from(tmp_path, DIAMOND)
    provider = RecordingProvider(lambda p: f"summary of {subject(p)}")
    result = METHODS["hucp"]().run(graph, LLM(provider), Settings(workers=4))
    seen = [subject(p) for p in provider.prompts]
    assert Counter(seen) == {"a": 1, "b": 1, "c": 1}
    assert seen.index("b") < seen.index("c") < seen.index("a")
    assert result.summaries["d.py::a"] == "summary of a"
    prompt_a = next(p for p in provider.prompts if subject(p) == "a")
    assert "summary of b" in prompt_a and "summary of c" in prompt_a


def test_context_is_direct_callees_only(tmp_path):
    graph = graph_from(tmp_path, {"chain.py": "def a():\n    b()\n\ndef b():\n    c()\n\ndef c():\n    pass\n"})
    provider = RecordingProvider(lambda p: f"summary of {subject(p)}")
    METHODS["hucp"]().run(graph, LLM(provider), Settings(workers=1))
    prompt_a = next(p for p in provider.prompts if subject(p) == "a")
    assert "summary of b" in prompt_a and "summary of c" not in prompt_a


def test_failures_are_recorded_not_fatal(tmp_path):
    graph = graph_from(tmp_path, DIAMOND)

    def answer(prompt):
        return "" if subject(prompt) == "b" else "ok"  # empty answer -> error, not a blank summary

    result = METHODS["hucp"]().run(graph, LLM(RecordingProvider(answer), max_retries=0), Settings())
    assert "d.py::b" in result.errors and "d.py::b" not in result.summaries
    assert result.summaries["d.py::a"] == "ok"


def test_iucp_inlines_summaries_at_call_sites(inheritance_graph):
    summaries = {uid: f"does {uid.split('::')[1]}\n- detail" for uid in inheritance_graph.units}
    code = annotate_calls(inheritance_graph.units["main.py::main"], inheritance_graph, summaries)
    lines = code.splitlines()
    i = next(n for n, line in enumerate(lines) if "person.display_info()" in line)
    assert lines[i - 1] == "    # calls DerivedClass.display_info: does DerivedClass.display_info; detail"
    j = next(n for n, line in enumerate(lines) if "DerivedClass(" in line and "calls" not in line)
    assert lines[j - 1].strip().startswith("# calls DerivedClass.__init__:")


def test_iucp_uses_language_comment_syntax(polyglot_graph):
    summaries = {uid: "x" for uid in polyglot_graph.units}
    code = annotate_calls(polyglot_graph.units["lib.rs::run"], polyglot_graph, summaries)
    assert "    // calls Counter.new: x" in code.splitlines()


def test_llmucp_explains_leaves_only(inheritance_graph):
    provider = RecordingProvider()
    METHODS["llmucp"]().run(inheritance_graph, LLM(provider), Settings(workers=1))
    explained = [p for p in provider.prompts if "Explain every line" in p]
    assert len(explained) == len(inheritance_graph.leaves()) == 2


def test_docucp_adds_class_and_library_docs(inheritance_graph):
    provider = RecordingProvider()
    METHODS["docucp"]().run(inheritance_graph, LLM(provider), Settings(workers=1))
    prompt = next(p for p in provider.prompts if subject(p) == "DerivedClass.display_info")
    assert "class BaseClass:" in prompt and "`print`:" in prompt


def test_baseline_returns_every_unit(inheritance_graph):
    def answer(prompt):
        ids = re.findall(r"^- `(.+)`$", prompt, re.M)
        return "```json\n" + json.dumps({uid: ["first", "second"] for uid in ids}) + "\n```"

    result = METHODS["baseline"]().run(inheritance_graph, LLM(RecordingProvider(answer)), Settings())
    assert set(result.summaries) == set(inheritance_graph.units)
    assert result.summaries["main.py::main"] == "- first\n- second"
    assert not result.errors


def test_baseline_splits_large_codebases(polyglot_graph):
    provider = RecordingProvider(lambda p: json.dumps({u: "s" for u in re.findall(r"^- `(.+)`$", p, re.M)}))
    result = METHODS["baseline"]().run(polyglot_graph, LLM(provider), Settings(baseline_max_chars=800))
    assert len(provider.prompts) > 3
    assert set(result.summaries) == set(polyglot_graph.units)


def test_parse_baseline_line_fallback():
    text = ("Here you go:\n- **main.py::main**: Runs the demo\n  - creates a person\n\n"
            "`base.py::Base.f`: does f: carefully\n")
    parsed = parse_baseline(text, ["main.py::main", "base.py::Base.f", "missing::x"])
    assert parsed == {"main.py::main": "Runs the demo\n- creates a person",
                      "base.py::Base.f": "does f: carefully"}


def test_settings_fit_small_context():
    settings = Settings().fit_context(8192)
    assert settings.max_code_chars + settings.max_context_chars < 8192 * 3
    assert Settings().fit_context(None).max_code_chars == Settings.max_code_chars
