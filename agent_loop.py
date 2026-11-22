"""Step 6: the iWSP loop. Write weights -> sandbox -> score -> revise.

    python agent_loop.py --list-models          # which ARC models are live right now
    python agent_loop.py --task reverse         # one task
    python agent_loop.py --task all             # reverse, add, sort + summary table

Each iteration is stateless: the agent gets the task, its best code so far, its last
attempt's result, and a short history table, not the whole conversation. That keeps
prompts a fixed size instead of growing every iteration.

Phases:
  solve    until the visible set is at 100%
  compress (if config.COMPRESS_AFTER_SOLVED) shrink param_count while staying at 100%,
           stopping after config.COMPRESS_PATIENCE iterations with no smaller model.

The agent only ever sees VISIBLE-set results. The best model is scored on the HIDDEN
set once, after the loop ends, and that number is never sent back to the agent.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import config
import sandbox
from evaluator import score
from llm import LLM, BudgetExceeded
import numpy as np

from model import MAX_LEN, N_VOCAB, TOK, VOCAB
from tasks import TASKS, dataset, encode_prompt

RUNS = Path(__file__).resolve().parent / "runs"

# ---- prompts ---------------------------------------------------------------------
SYSTEM_PROMPT = f"""You write the weights of a tiny transformer directly in Python code. The model is \
NEVER trained.

RULES
- No training of any kind: no gradient descent, no optimizer steps, no fitting loops, no search \
over weights by trial and error against data. Every weight is set in closed form by you, for a \
reason you can state.
- Your code never sees the task data. It only fills in weights.
- All else being equal, more interpretable is better. A small accuracy gain that adds ugly \
complexity is not worth it. Parameter count is a proxy, but the real criterion is \
human-understandable structure in the weights.
- Each revision starts with a one-line comment justifying what you changed and why.

THE MODEL (numpy, fixed code you cannot change)
Vocabulary ({N_VOCAB} tokens, id = index): {VOCAB}. Digit tokens have id == digit value.
MAX_LEN = {MAX_LEN} positions.

    x = W_E[tokens] + W_pos[0..T-1]                    # residual stream, shape (T, d_model)
    for l in range(n_layers):
        out = 0
        for h in range(n_heads):                          # heads run in parallel on the same x
            q, k, v = x @ W_Q[h], x @ W_K[h], x @ W_V[h]   # layer l's matrices
            A = softmax(q @ k.T with causal mask)          # NO 1/sqrt(d_head) scaling
            out = out + (A @ v) @ W_O[h]
        x = x + out
        if d_mlp > 0:
            x = x + relu(x @ W_in + b_in) @ W_out + b_out
    logits = x @ W_U                                    # no LayerNorm anywhere

Weights live in model.w, a dict of numpy float arrays that start at ZERO:
    "W_E" (N_VOCAB, d_model)     "W_pos" (MAX_LEN, d_model)     "W_U" (d_model, N_VOCAB)
    f"L{{l}}.W_Q", f"L{{l}}.W_K", f"L{{l}}.W_V"  each (n_heads, d_model, d_head)
    f"L{{l}}.W_O"                              (n_heads, d_head, d_model)
    f"L{{l}}.W_in" (d_model, d_mlp)   f"L{{l}}.b_in" (d_mlp,)
    f"L{{l}}.W_out" (d_mlp, d_model)  f"L{{l}}.b_out" (d_model,)   (MLP keys only if d_mlp > 0)
Write entries in place, e.g. model.w["L0.W_Q"][0, 3, 5] = 2.0. Do not add or replace keys.

ARCHITECTURE: optionally set a module-level dict
    CONFIG = dict(d_model=..., n_layers=..., n_heads=..., d_head=..., d_mlp=...)
Limits: {config.ARCH_LIMITS} (d_mlp may be 0). Default: {config.DEFAULT_ARCH}.
model.param_count() counts every entry of every array, zero or not. Smaller CONFIG = fewer params.

DECODING: the input is BOS + input tokens + separator. The model then greedily generates the \
answer one token at a time; the prediction at the last position is the next token. All answer \
tokens must match exactly.

SANDBOX: your code runs isolated. Available without import: np, math, VOCAB, TOK (token -> id), \
N_VOCAB, MAX_LEN. Only `import numpy` / `import math` are allowed. Not allowed: file I/O, \
getattr/eval/exec, any name or attribute starting with "_", str.format (f-strings are fine). \
print() output is shown back to you. To debug, you may call \
`logits, attn = model.forward(tokens, return_attn=True)` inside write_weights (tokens: int array \
(T,) or (B, T)); attn["L0.H0"][i, j] is how much position i attends to position j. Timeout {config.SANDBOX_TIMEOUT_S} s.

REPLY FORMAT: your reply is ONE python code block and nothing else. Its first line is the \
justification comment:
```python
# JUSTIFICATION: <1-3 sentences: what changed and why>
CONFIG = dict(...)   # optional
def write_weights(model):
    ...
```
Do not stop after the justification: the code block must contain the full write_weights."""

NO_CODE_NUDGE = ("Your reply contained no ```python code block. Reply now with ONLY the code block "
                 "(first line # JUSTIFICATION: ...), containing the complete write_weights.")


def format_examples(task_name: str, k: int = 5) -> str:
    task = TASKS[task_name]
    lines = []
    for x, y in dataset(task_name, "visible")[:k]:
        toks = ["BOS", *x, task.sep, *y]
        pos = "  ".join(f"{i}:{t}" for i, t in enumerate(toks))
        lines.append(f"  {x} -> {y}    positions: {pos}")
    return "\n".join(lines)


def build_user_prompt(task_name: str, history: list[dict], best: dict | None, phase: str) -> str:
    task = TASKS[task_name]
    parts = [
        f"TASK: {task.name}\n{task.prompt}",
        f"Separator token: {task.sep!r}. The prompt is {task.prompt_len} tokens "
        f"(positions 0..{task.prompt_len - 1}); the answer is {task.output_len} tokens.",
        "PREDICTION ALIGNMENT (off-by-one errors are the most common bug): the logits at "
        "position p predict the token at position p+1. So:\n"
        + "\n".join(f"  logits at position {task.prompt_len - 1 + i} (holding "
                    f"{'the separator' if i == 0 else f'answer token {i}'}) must predict answer token {i + 1}"
                    for i in range(task.output_len))
        + "\nPredictions at positions before the separator are never used.",
        "Visible examples (position:token for the full sequence, answer included):\n"
        + format_examples(task_name),
    ]
    if history:
        rows = "\n".join(f"  {h['iter']:>3}  {h['accuracy']:6.1%}  {h['param_count']:>8}  "
                         f"{(h['error'][:40] + '...') if h['error'] else h['justification'][:70]}"
                         for h in history[-10:])
        parts.append(f"HISTORY (last 10; iter, visible accuracy, params, note):\n{rows}")
    if best:
        parts.append(f"BEST SO FAR (iter {best['iter']}, {best['accuracy']:.1%} visible, "
                     f"{best['param_count']} params):\n```python\n{best['code']}\n```")
    if history:
        last = history[-1]
        fb = [f"LAST ATTEMPT (iter {last['iter']}):"]
        if last["error"]:
            fb.append(f"Failed to run:\n{last['error']}")
        else:
            fb.append(f"Visible accuracy {last['accuracy']:.1%}, {last['param_count']} params.")
            if last["failures"]:
                fb.append("Failures (input, expected, got):\n"
                          + "\n".join(f"  {x} -> expected {e}, got {g}" for x, e, g in last["failures"]))
            if last.get("attention"):
                fb.append(last["attention"])
        if last["stdout"]:
            fb.append(f"Your print() output:\n{last['stdout'][-1500:]}")
        if best is None or last["iter"] != best["iter"]:
            fb.append(f"Its code:\n```python\n{last['code']}\n```")
        parts.append("\n".join(fb))

    if phase == "solve":
        parts.append("GOAL: reach 100% visible accuracy. Write the full write_weights code.")
    else:
        parts.append(f"GOAL (compression phase): the best model already scores 100%. Make it smaller: "
                     f"lower param_count below {best['param_count']} while keeping 100%, e.g. by "
                     f"shrinking CONFIG and packing subspaces. Keep it readable.")
    return "\n\n".join(parts)


def attention_report(model, task_name: str, x: str, y: str) -> str:
    """For one (visible) example, show where each answer-predicting position attends.

    Uses teacher forcing: the model sees the true sequence BOS x sep y, so each line shows
    what the model would predict at that step if all previous answer tokens were right.
    """
    task = TASKS[task_name]
    seq = encode_prompt(task, x) + [TOK[c] for c in y]
    names = ["BOS", *x, task.sep, *y]
    logits, attn = model.forward(np.array(seq[:-1]), return_attn=True)
    lines = [f"Attention on {x} -> {y} (teacher-forced; top-2 attended positions per head):"]
    for i in range(task.output_len):
        p = task.prompt_len - 1 + i
        pred = VOCAB[int(logits[p].argmax())]
        heads = []
        for name, A in attn.items():
            top = np.argsort(A[p])[::-1][:2]
            heads.append(f"{name}: " + ", ".join(f"pos{j}({names[j]})={A[p, j]:.2f}" for j in top))
        lines.append(f"  pos {p} -> predicts {pred!r}, should be {y[i]!r} | " + " | ".join(heads))
    return "\n".join(lines)


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)
JUST_RE = re.compile(r"JUSTIFICATION:\s*(.+?)(?:\n\s*\n|\n(?!#)|```|$)", re.DOTALL)


def parse_reply(text: str) -> tuple[str, str]:
    """Returns (code, justification). code is "" if no code block was found."""
    blocks = CODE_RE.findall(text)
    code = blocks[-1].strip() if blocks else ""
    m = JUST_RE.search(text)  # finds "# JUSTIFICATION:" inside the code, or a bare line outside it
    just = m.group(1) if m else text.split("```")[0]
    just = " ".join(line.strip().lstrip("#").strip() for line in just.strip().splitlines())
    return code, just[:500]


# ---- the loop ----------------------------------------------------------------------
def _is_better(a: dict, b: dict | None) -> bool:
    """Higher accuracy wins; at equal accuracy, fewer params wins."""
    return b is None or (a["accuracy"], -a["param_count"]) > (b["accuracy"], -b["param_count"])


def run_task(task_name: str, llm, max_iters: int = config.MAX_ITERS,
             compress: bool = config.COMPRESS_AFTER_SOLVED, out_dir: Path | None = None) -> dict:
    out_dir = out_dir or RUNS / task_name / datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    history, best, best_model, stall, stop_reason = [], None, None, 0, "max_iters"

    for it in range(1, max_iters + 1):
        phase = "compress" if best and best["accuracy"] == 1.0 else "solve"
        if phase == "compress" and not compress:
            stop_reason = "solved"
            break
        if phase == "compress" and stall >= config.COMPRESS_PATIENCE:
            stop_reason = "compression_stalled"
            break

        prompt = build_user_prompt(task_name, history, best, phase)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
        t0 = time.time()
        try:
            reply = llm.chat(messages)
            n_in, n_out, raw = reply.input_tokens, reply.output_tokens, reply.text
            code, justification = parse_reply(reply.text)
            if not code:  # some models stop after the justification line: ask once more, same iteration
                retry = llm.chat(messages + [{"role": "assistant", "content": reply.text},
                                             {"role": "user", "content": NO_CODE_NUDGE}])
                n_in, n_out = n_in + retry.input_tokens, n_out + retry.output_tokens
                raw = reply.text + "\n\n[RETRY]\n" + retry.text
                code, just2 = parse_reply(retry.text)
                justification = justification if justification and not code else (just2 or justification)
        except BudgetExceeded as e:
            stop_reason = f"budget: {e}"
            break

        rec = {"iter": it, "phase": phase, "code": code, "justification": justification,
               "accuracy": 0.0, "param_count": 0, "nonzero_params": 0, "failures": [],
               "error": "", "stdout": "", "input_tokens": n_in,
               "output_tokens": n_out, "raw_reply": raw, "retried": raw is not reply.text}
        if not code:
            rec["error"] = "No ```python code block found in your reply."
        else:
            res = sandbox.run(code)
            rec["stdout"] = res.stdout
            if not res.ok:
                rec["error"] = res.error[-2000:]
            else:
                s = score(res.model, task_name, "visible")
                rec.update(accuracy=s.accuracy, failures=s.failures,
                           param_count=res.model.param_count(),
                           nonzero_params=res.model.nonzero_param_count())
                if s.failures:
                    x, y, _ = s.failures[0]
                    with np.errstate(all="ignore"):
                        rec["attention"] = attention_report(res.model, task_name, x, y)
        rec["seconds"] = round(time.time() - t0, 1)
        history.append(rec)
        (out_dir / f"iter_{it:02d}.json").write_text(json.dumps(rec, indent=2))

        if not rec["error"] and _is_better(rec, best):
            if phase == "compress":
                stall = 0
            best, best_model = rec, res.model
        elif phase == "compress":
            stall += 1
        print(f"[{task_name}] iter {it:>2} {phase:8s} acc={rec['accuracy']:6.1%} "
              f"params={rec['param_count']:>7} {'ERR ' + rec['error'][:60] if rec['error'] else ''}")
    else:
        stop_reason = "max_iters"

    summary = {"task": task_name, "model": llm.model, "provider": llm.provider,
               "iterations": len(history), "stop_reason": stop_reason,
               "input_tokens": sum(h["input_tokens"] for h in history),
               "output_tokens": sum(h["output_tokens"] for h in history),
               "first_solved_iter": next((h["iter"] for h in history if h["accuracy"] == 1.0), None),
               "best_iter": best and best["iter"], "best_visible": best and best["accuracy"],
               "best_params": best and best["param_count"],
               "best_nonzero_params": best and best["nonzero_params"],
               "hidden_accuracy": None}
    if best_model is not None:
        summary["hidden_accuracy"] = score(best_model, task_name, "hidden").accuracy  # never shown to the agent
        (out_dir / "best.py").write_text(best["code"], encoding="utf-8")
        best_model.save(out_dir / "best.npz")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def print_table(summaries: list[dict]) -> None:
    print(f"\n{'task':8} {'visible':>8} {'hidden':>8} {'params':>8} {'nonzero':>8} "
          f"{'solved@':>8} {'iters':>6} {'tokens':>10}  stop")
    for s in summaries:
        fmt = lambda v: "-" if v is None else f"{v:.1%}"  # noqa: E731
        print(f"{s['task']:8} {fmt(s['best_visible']):>8} {fmt(s['hidden_accuracy']):>8} "
              f"{s['best_params'] or '-':>8} {s['best_nonzero_params'] or '-':>8} "
              f"{s['first_solved_iter'] or '-':>8} {s['iterations']:>6} "
              f"{s['input_tokens'] + s['output_tokens']:>10,}  {s['stop_reason']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="reverse", choices=[*TASKS, "all"])
    ap.add_argument("--iters", type=int, default=config.MAX_ITERS)
    ap.add_argument("--model", default=config.MODEL)
    ap.add_argument("--no-compress", action="store_true", help="stop as soon as 100%% visible is reached")
    ap.add_argument("--list-models", action="store_true")
    args = ap.parse_args()

    llm = LLM(model=args.model)
    if args.list_models:
        print("\n".join(llm.list_models()))
        return

    summaries = []
    for name in (list(TASKS) if args.task == "all" else [args.task]):
        s = run_task(name, llm, args.iters, compress=not args.no_compress)
        summaries.append(s)
        if s["stop_reason"].startswith("budget"):
            print("Budget cap reached; stopping all runs.")
            break
    print_table(summaries)
    print(f"Budget used so far: {llm.budget.total_tokens:,} tokens, ${llm.budget.state['usd']:.2f}")
    (RUNS / "latest_summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
