"""Task definitions and deterministic (input, target) generators.

Every example becomes one token sequence:

    BOS  <input tokens>  <sep>  <target tokens>

The model sees everything up to and including <sep>, then greedily generates
len(target) tokens. Exact match on all of them counts as correct.

    reverse:  BOS 3 1 4 1 5 |  5 1 4 1 3
    add:      BOS 0 7 + 5 8 =  0 6 5        (fixed width, zero-padded)
    sort:     BOS 3 1 4 1 5 |  1 1 3 4 5
"""

import zlib
from dataclasses import dataclass
from typing import Callable

import numpy as np

import config
from model import TOK


@dataclass(frozen=True)
class Task:
    name: str
    prompt: str        # plain-English description shown to the agent
    sep: str           # token between input and answer
    input_len: int
    output_len: int
    make: Callable[[np.random.Generator], tuple[str, str]]  # rng -> (input, target) as strings

    @property
    def prompt_len(self) -> int:
        return 1 + self.input_len + 1  # BOS + input + sep


def _digits(rng, n):
    return "".join(str(d) for d in rng.integers(0, 10, size=n))


def _make_reverse(rng):
    x = _digits(rng, 5)
    return x, x[::-1]


def _make_add(rng):
    a, b = rng.integers(0, 100, size=2)
    return f"{a:02d}+{b:02d}", f"{a + b:03d}"


def _make_sort(rng):
    x = _digits(rng, 5)
    return x, "".join(sorted(x))


TASKS = {
    "reverse": Task(
        name="reverse",
        prompt="Reverse a string of 5 digits. Example: input 31415 -> output 51413.",
        sep="|", input_len=5, output_len=5, make=_make_reverse,
    ),
    "add": Task(
        name="add",
        prompt=("Add two 2-digit numbers, each zero-padded to 2 digits. Output the sum as exactly "
                "3 digits, zero-padded. Example: input 07+58 -> output 065."),
        sep="=", input_len=5, output_len=3, make=_make_add,
    ),
    "sort": Task(
        name="sort",
        prompt="Sort 5 digits in ascending order (duplicates kept). Example: input 31415 -> output 11345.",
        sep="|", input_len=5, output_len=5, make=_make_sort,
    ),
}


def _unique_examples(task, seed, n, exclude=frozenset()):
    """n distinct examples whose inputs are not in `exclude`, deterministic given seed."""
    rng = np.random.default_rng([seed, zlib.crc32(task.name.encode())])
    seen, out = set(exclude), []
    while len(out) < n:
        x, y = task.make(rng)
        if x not in seen:
            seen.add(x)
            out.append((x, y))
    return out


def dataset(task_name: str, split: str) -> list[tuple[str, str]]:
    """split is 'visible' (shown to the agent) or 'hidden' (final scores only, disjoint from visible)."""
    task = TASKS[task_name]
    visible = _unique_examples(task, config.VISIBLE_SEED, config.VISIBLE_SIZE)
    if split == "visible":
        return visible
    if split == "hidden":
        return _unique_examples(task, config.HIDDEN_SEED, config.HIDDEN_SIZE,
                                exclude=frozenset(x for x, _ in visible))
    raise ValueError(f"unknown split {split!r}")


def encode_prompt(task: Task, x: str) -> list[int]:
    """'07+58' -> [BOS, 0, 7, +, 5, 8, =] as token ids. Every character is one token."""
    return [TOK["BOS"]] + [TOK[c] for c in x] + [TOK[task.sep]]
