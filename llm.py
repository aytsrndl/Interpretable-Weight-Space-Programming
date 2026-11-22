"""One small client for every provider, plus a global budget.

ARC (default) and Anthropic both expose OpenAI-compatible chat endpoints, so a single
`openai.OpenAI(base_url=...)` client covers both. Only the key for the selected provider
is read from .env.

The budget is stored in runs/budget.json so it holds across separate script runs.
Delete that file to reset it.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import config

BUDGET_FILE = Path(__file__).resolve().parent / "runs" / "budget.json"
KEY_ENV = {"arc": "ARC_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    def __init__(self, path: Path = BUDGET_FILE):
        self.path = path
        self.state = {"input_tokens": 0, "output_tokens": 0, "usd": 0.0}
        if path.exists():
            self.state.update(json.loads(path.read_text()))

    @property
    def total_tokens(self) -> int:
        return self.state["input_tokens"] + self.state["output_tokens"]

    def check(self) -> None:
        if self.total_tokens >= config.MAX_TOTAL_TOKENS:
            raise BudgetExceeded(f"token cap reached: {self.total_tokens:,} >= {config.MAX_TOTAL_TOKENS:,}")
        if self.state["usd"] >= config.MAX_USD:
            raise BudgetExceeded(f"spend cap reached: ${self.state['usd']:.2f} >= ${config.MAX_USD:.2f}")

    def add(self, provider: str, input_tokens: int, output_tokens: int) -> None:
        price_in, price_out = config.PRICE_PER_MTOK.get(provider, (None, None))
        if price_in is None:
            raise BudgetExceeded(f"no price for provider {provider!r} in config.PRICE_PER_MTOK; "
                                 "refusing to spend an unknown amount")
        self.state["input_tokens"] += input_tokens
        self.state["output_tokens"] += output_tokens
        self.state["usd"] += (input_tokens * price_in + output_tokens * price_out) / 1e6
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.state, indent=2))


@dataclass
class Reply:
    text: str
    input_tokens: int
    output_tokens: int


class LLM:
    """Usage: LLM().chat([{"role": "system", ...}, {"role": "user", ...}])"""

    def __init__(self, provider: str = config.PROVIDER, model: str = config.MODEL,
                 budget: Budget | None = None):
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parent / ".env")
        key = os.environ.get(KEY_ENV[provider], "").strip()
        if not key:
            raise RuntimeError(f"{KEY_ENV[provider]} is empty. Put it in .env (see .env.example).")
        self.provider, self.model = provider, model
        self.budget = budget or Budget()
        # max_retries + long timeout: ARC queues API calls behind web chat when busy.
        self.client = OpenAI(base_url=config.BASE_URLS[provider], api_key=key,
                             max_retries=6, timeout=900)

    def list_models(self) -> list[str]:
        return sorted(m.id for m in self.client.models.list())

    def chat(self, messages: list[dict], temperature: float = config.TEMPERATURE,
             seed: int | None = config.SEED) -> Reply:
        self.budget.check()
        kwargs = dict(model=self.model, messages=messages, temperature=temperature,
                      max_tokens=config.MAX_OUTPUT_TOKENS)
        if seed is not None and self.provider == "arc":
            kwargs["seed"] = seed  # vLLM honors this; not every provider accepts it
        resp = self.client.chat.completions.create(**kwargs)
        usage = resp.usage
        reply = Reply(resp.choices[0].message.content or "",
                      usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0)
        self.budget.add(self.provider, reply.input_tokens, reply.output_tokens)
        return reply


class FakeLLM:
    """Scripted replies for tests. Same interface as LLM, never touches the network."""

    def __init__(self, replies: list[str], budget: Budget | None = None):
        self.replies, self.calls = list(replies), []
        self.provider, self.model = "arc", "fake"
        self.budget = budget

    def chat(self, messages, temperature=None, seed=None) -> Reply:
        if self.budget:
            self.budget.check()
        self.calls.append(messages)
        text = self.replies.pop(0) if self.replies else "no more scripted replies"
        n_in = sum(len(m["content"]) for m in messages) // 4
        if self.budget:
            self.budget.add("arc", n_in, len(text) // 4)
        return Reply(text, n_in, len(text) // 4)
