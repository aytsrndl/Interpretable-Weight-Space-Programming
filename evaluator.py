"""Fixed, deterministic scoring. Agent-written code must never import or modify this file.

score(model, task_name) runs greedy decoding on every example in a split and
reports exact-match accuracy plus a few failures (input, expected, got) for feedback.
"""

from dataclasses import dataclass, field

import numpy as np

import config
from model import VOCAB, Model
from tasks import TASKS, dataset, encode_prompt


@dataclass
class ScoreResult:
    task: str
    split: str
    accuracy: float
    n: int
    failures: list = field(default_factory=list)  # [(input, expected, got), ...]


def score(model: Model, task_name: str, split: str = "visible",
          n_failures: int = config.N_FAILURES_SHOWN) -> ScoreResult:
    task = TASKS[task_name]
    examples = dataset(task_name, split)
    prompts = np.array([encode_prompt(task, x) for x, _ in examples])
    with np.errstate(all="ignore"):  # broken weights may overflow; they just score badly
        out = model.generate(prompts, task.output_len)

    correct, failures = 0, []
    for (x, expected), toks in zip(examples, out):
        got = "".join(VOCAB[t] if len(VOCAB[t]) == 1 else f"<{VOCAB[t]}>" for t in toks)
        if got == expected:
            correct += 1
        elif len(failures) < n_failures:
            failures.append((x, expected, got))

    return ScoreResult(task_name, split, correct / len(examples), len(examples), failures)
