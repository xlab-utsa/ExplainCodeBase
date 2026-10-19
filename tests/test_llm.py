import pytest

from explaincodebase.cli import main
from explaincodebase.llm import (LLM, LLMError, VLLMProvider, default_label, make_provider,
                                 strip_reasoning)

from .conftest import RecordingProvider


def test_vllm_is_the_default_provider(monkeypatch):
    monkeypatch.delenv("ECB_PROVIDER", raising=False)
    assert isinstance(make_provider(), VLLMProvider)
    assert isinstance(make_provider("vllm_models"), VLLMProvider)  # old name still works


def test_vllm_discovers_model_and_context(fake_vllm):
    provider = make_provider()
    assert provider.model == "Qwen/Qwen3-8B"
    assert provider.context_length == 8192
    assert default_label(provider) == "qwen3_8b"


def test_vllm_strips_reasoning_and_counts_tokens(fake_vllm):
    llm = LLM(make_provider())
    assert llm.complete("Summarize add().") == "It adds two numbers."
    assert llm.usage["input_tokens"] == 12 and llm.usage["output_tokens"] == 7
    request = fake_vllm.requests[-1]
    assert request["model"] == "Qwen/Qwen3-8B"
    assert request["messages"][0]["role"] == "system"


def test_vllm_retries_rate_limits(fake_vllm, monkeypatch):
    monkeypatch.setattr("explaincodebase.llm.time.sleep", lambda s: None)
    fake_vllm.fail_next = 2
    assert LLM(make_provider()).complete("hi") == "It adds two numbers."
    assert len(fake_vllm.requests) == 3


def test_vllm_explicit_model_and_settings(fake_vllm, monkeypatch):
    monkeypatch.setenv("VLLM_MAX_TOKENS", "256")
    monkeypatch.setenv("VLLM_TEMPERATURE", "0")
    LLM(make_provider(model="Qwen/Qwen3-8B")).complete("hi")
    request = fake_vllm.requests[-1]
    assert request["max_tokens"] == 256 and request["temperature"] == 0


def test_vllm_unreachable_server_is_a_clear_error(monkeypatch):
    monkeypatch.delenv("VLLM_MODEL", raising=False)
    provider = VLLMProvider(base_url="http://127.0.0.1:9/v1")
    with pytest.raises(LLMError, match="is the server running"):
        LLM(provider, max_retries=0).complete("hi")


def test_strip_reasoning():
    assert strip_reasoning("<think>a\nb</think>\nB") == "B"
    assert strip_reasoning("half of a thought</think>answer") == "answer"
    assert strip_reasoning("plain") == "plain"


def test_empty_answer_is_an_error():
    with pytest.raises(LLMError, match="empty"):
        LLM(RecordingProvider(lambda p: "<think>only thinking</think>"), max_retries=0).complete("x")


def test_cache_reuses_answers(tmp_path):
    provider = RecordingProvider(lambda p: "cached answer")
    llm = LLM(provider, cache_dir=str(tmp_path))
    assert llm.complete("q") == llm.complete("q") == "cached answer"
    assert len(provider.prompts) == 1 and llm.usage["cached_calls"] == 1
    llm.complete("q", cache_tag="trial1")
    assert len(provider.prompts) == 2


def test_summarize_end_to_end_against_vllm(fake_vllm, tmp_path, capsys):
    from .conftest import INHERITANCE

    fake_vllm.reply = lambda body: "<think>…</think>- does a thing"
    code = main(["--output-dir", str(tmp_path), "summarize", INHERITANCE, "--methods", "hucp,iucp"])
    assert code == 0
    out = capsys.readouterr().out
    assert "inheritance_qwen3_8b_hucp.json" in out and "inheritance_qwen3_8b_iucp.json" in out
