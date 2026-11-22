# mini-iWSP — Interpretable Weight-Space Programming at toy scale

A small reproduction of *Interpretable weight-space programming builds transparent transformers
and brain encoding models* (Mehta et al., arXiv:2611.18114) on algorithmic tasks.

**What iWSP is.** An LLM agent writes every weight of a small transformer directly in code, with no
training of any kind. A fixed evaluator scores the model, and the agent revises its weights using
that score. Each edit is meant to be readable, so the final network is an explanation as well as a predictor.

Status: work in progress. Results tables and plots will be added as steps finish.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add your ARC key (needed from Step 6 on)
```

## Running
```bash
python -m pytest -q              # all checks
python -m handwritten.reverse    # hand-written reverse model: 100% visible and hidden
```

## Steps
| Step | File | Status |
|---|---|---|
| 1 | `model.py` — numpy transformer | done |
| 2 | `tasks.py` — reverse / add / sort, 200 visible + 2,000 hidden each | done |
| 3 | `evaluator.py` — fixed scoring | done |
| 4 | `handwritten/reverse.py` ⏸ | done: 100% hidden, 4,510 params (82 nonzero) |
| 5 | `sandbox.py` | |
| 6 | `agent_loop.py` ⏸ | |
| 7 | `describe_rebuild.py` | |
| 8 | `ablation.py` ⏸ | |
| 9 | `golf.py` (stretch) | |
| 10 | circuit library + held-out tasks | |
