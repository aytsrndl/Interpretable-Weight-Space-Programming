"""Step 6: the loop, tested end to end with a scripted FakeLLM (no network, no key)."""

import json

import pytest

import agent_loop
import config
from llm import Budget, FakeLLM
from tests.fixtures import REVERSE_OK, ZEROS


def reply(code, why="test"):
    return f"JUSTIFICATION: {why}\n```python\n{code}\n```"


SMALLER = REVERSE_OK.replace("W_U\"][SCRATCH + d, d] = 1.0",
                             "W_U\"][SCRATCH + d, d] = 1.0  # same circuit")  # same size: no gain


def test_parse_reply():
    code, why = agent_loop.parse_reply("JUSTIFICATION: copy head.\n\n```python\nx = 1\n```")
    assert code == "x = 1" and why == "copy head."
    assert agent_loop.parse_reply("no code here")[0] == ""


def test_prompts_never_read_hidden_split(monkeypatch):
    """Building prompts may only touch the visible split. (Checking for hidden strings in the
    prompt text is too weak: a visible answer like 80677 can coincidentally equal a hidden input.)"""
    real = agent_loop.dataset

    def guarded(name, split):
        assert split == "visible", "prompt code read the hidden split"
        return real(name, split)

    monkeypatch.setattr(agent_loop, "dataset", guarded)
    for name in agent_loop.TASKS:
        agent_loop.build_user_prompt(name, [], None, "solve")


def test_loop_recovers_from_errors_then_solves(tmp_path):
    llm = FakeLLM([
        "I forgot the code block.",
        reply("import os\ndef write_weights(model): pass"),  # rejected by sandbox
        reply(ZEROS, "zeros"),                                 # runs, ~0%
        reply(REVERSE_OK, "copy head"),                        # 100%
    ])
    s = agent_loop.run_task("reverse", llm, max_iters=10, compress=False, out_dir=tmp_path)

    assert s["stop_reason"] == "solved" and s["first_solved_iter"] == 4
    assert s["best_visible"] == 1.0 and s["hidden_accuracy"] == 1.0
    assert sorted(p.name for p in tmp_path.glob("iter_*.json")) == [f"iter_0{i}.json" for i in range(1, 5)]
    it2 = json.loads((tmp_path / "iter_02.json").read_text())
    assert "Rejected" in it2["error"]
    # feedback reached the agent: the 4th prompt mentions the 3rd attempt's failures
    assert "expected" in llm.calls[3][1]["content"]
    assert (tmp_path / "best.py").exists() and (tmp_path / "best.npz").exists()


def test_compression_phase_stops_when_stalled(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "COMPRESS_PATIENCE", 2)
    llm = FakeLLM([reply(REVERSE_OK), reply(SMALLER), reply(ZEROS), reply(REVERSE_OK)])
    s = agent_loop.run_task("reverse", llm, max_iters=10, compress=True, out_dir=tmp_path)
    assert s["stop_reason"] == "compression_stalled" and s["iterations"] == 3
    assert "compression phase" in llm.calls[1][1]["content"]


def test_budget_stops_the_loop(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "MAX_TOTAL_TOKENS", 1)
    budget = Budget(tmp_path / "budget.json")
    llm = FakeLLM([reply(ZEROS)] * 5, budget=budget)
    s = agent_loop.run_task("reverse", llm, max_iters=5, out_dir=tmp_path / "run")
    assert s["stop_reason"].startswith("budget") and s["iterations"] == 1


def test_unknown_price_refuses_to_spend(tmp_path):
    with pytest.raises(Exception, match="no price"):
        Budget(tmp_path / "b.json").add("anthropic", 10, 10)


def test_llm_streams_and_counts_tokens(tmp_path):
    """LLM.chat must stream (ARC caps non-streaming at 8k) and record usage from the last chunk."""
    from types import SimpleNamespace as NS

    import llm as llm_mod

    def chunk(content=None, usage=None, reasoning=None):
        delta = NS(content=content, reasoning_content=reasoning)
        return NS(choices=[NS(delta=delta)] if usage is None else [], usage=usage)

    seen = {}

    def create(**kw):
        seen.update(kw)
        return iter([chunk(reasoning="thinking..."), chunk("JUSTIFICATION: hi\n"), chunk("```python\nx=1\n```"),
                     chunk(usage=NS(prompt_tokens=100, completion_tokens=40))])

    m = llm_mod.LLM.__new__(llm_mod.LLM)
    m.provider, m.model = "arc", "fake"
    m.budget = Budget(tmp_path / "b.json")
    m.client = NS(chat=NS(completions=NS(create=create)))

    r = m.chat([{"role": "user", "content": "hello"}])
    assert seen["stream"] is True and seen["stream_options"] == {"include_usage": True}
    assert r.text == "JUSTIFICATION: hi\n```python\nx=1\n```"
    assert (r.input_tokens, r.output_tokens) == (100, 40) and m.budget.total_tokens == 140
