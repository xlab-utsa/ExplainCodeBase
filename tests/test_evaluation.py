import json

from explaincodebase.cli import main
from explaincodebase.evaluate import evaluate_package, parse_choice
from explaincodebase.mcq import parse_letter, run_mcq
from explaincodebase.pipeline import load_runs, summarize_package

from .conftest import INHERITANCE, RecordingProvider


def summarize_all(tmp_path, label="mock"):
    return summarize_package(INHERITANCE, "mock", ["baseline", "hucp", "iucp", "llmucp", "docucp"],
                             output_dir=str(tmp_path), label=label)


def test_summary_files_are_self_describing(tmp_path):
    written = summarize_all(tmp_path)
    data = json.loads(open(written["hucp"]).read())
    assert data["package"] == "inheritance" and data["method"] == "hucp" and data["label"] == "mock"
    assert set(data["summaries"]) == {
        "main.py::(global)", "main.py::main", "derived_class.py::DerivedClass.__init__",
        "derived_class.py::DerivedClass.display_info", "base_class.py::BaseClass.__init__",
        "base_class.py::BaseClass.display_name"}
    assert data["errors"] == {}
    assert data["usage"]["calls"] == 6
    assert (tmp_path / "graph" / "inheritance.json").exists()


def test_judge_is_blind_and_shuffled(tmp_path, monkeypatch):
    summarize_all(tmp_path)
    recorder = RecordingProvider(lambda p: '{"best": "A"}')
    monkeypatch.setattr("explaincodebase.evaluate.make_provider", lambda *a, **k: recorder)
    result = evaluate_package("inheritance", output_dir=str(tmp_path), trials=4, cache=False)
    for prompt in recorder.prompts:
        for leaked in ("hucp", "iucp", "llmucp", "docucp", "baseline", "METHOD", "MODEL"):
            assert leaked not in prompt
    assert result["units_evaluated"] == 5  # module code excluded by default
    assert sum(r["wins"] for r in result["by_run"].values()) == 5 * 4
    # Always answering "A" must spread wins across runs because the order is reshuffled.
    assert sum(1 for r in result["by_run"].values() if r["wins"]) >= 3
    assert result["invalid_answers"] == 0


def test_parse_choice():
    assert parse_choice('{"best": "C"}', ["A", "B", "C"]) == "C"
    assert parse_choice("<think>A or B?</think> Best: B", ["A", "B"]) == "B"
    assert parse_choice("Summary B is best.", ["A", "B"]) == "B"
    assert parse_choice("nothing useful", ["A", "B"]) is None


def test_mcq_scores_runs_and_control(tmp_path):
    summarize_all(tmp_path)
    questions = [{"question_number": "Q1", "question": "What does main do?",
                  "options": {"A": "Builds a person", "B": "Nothing"}, "correct_answer": "A"},
                 {"question_number": "Q2", "question": "Base class?",
                  "options": {"A": "Derived", "B": "BaseClass"}, "correct_answer": "B"}]
    mcq_file = tmp_path / "q.json"
    mcq_file.write_text(json.dumps(questions))
    result = run_mcq("inheritance", mcq_file=str(mcq_file), output_dir=str(tmp_path),
                     answerer="mock", cache=False)
    assert set(result["scores"]) == {f"{m}@mock" for m in
                                     ("baseline", "hucp", "iucp", "llmucp", "docucp")} | {"no_documentation"}
    assert all(score["accuracy"] == 0.5 for score in result["scores"].values())  # mock always says A


def test_parse_letter():
    assert parse_letter("B", ["A", "B", "C", "D"]) == "B"
    assert parse_letter("**C.** because", ["A", "B", "C", "D"]) == "C"
    assert parse_letter("<think>maybe A</think>The answer is D", ["A", "B", "C", "D"]) == "D"
    assert parse_letter("Because", ["A", "B"]) is None


def test_bundled_mcq_files_load():
    import glob

    from explaincodebase.mcq import load_questions

    files = glob.glob("mcqs/*.json")
    assert files
    for path in files:
        assert load_questions(path)


def test_cli_full_flow(tmp_path, capsys):
    out = str(tmp_path)
    assert main(["--output-dir", out, "summarize", INHERITANCE, "--provider", "mock"]) == 0
    assert len(load_runs(out, "inheritance")) == 5
    assert main(["--output-dir", out, "evaluate", "--judge", "mock", "--trials", "2"]) == 0
    assert main(["--output-dir", out, "graph", INHERITANCE]) == 0
    printed = capsys.readouterr().out
    assert "units judged by mock" in printed
    assert main(["languages"]) == 0
