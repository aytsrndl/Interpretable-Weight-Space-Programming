"""All knobs in one place. Nothing here is secret; API keys live in .env."""

# ---- Data / evaluation (Steps 2–3) ----------------------------------------
# The agent only ever sees the VISIBLE set. The HIDDEN set is used for final
# scores and must never appear in a prompt (mirrors the paper's visible/hidden split).
VISIBLE_SIZE = 200
HIDDEN_SIZE = 2000
VISIBLE_SEED = 1234
HIDDEN_SEED = 98765
N_FAILURES_SHOWN = 5  # failing examples returned for agent feedback

# ---- LLM provider (Step 6+) -------------------------------------------------
# "arc" = VT ARC's free OpenAI-compatible endpoint; "anthropic" = paid API.
PROVIDER = "arc"
BASE_URLS = {
    "arc": "https://llm-api.arc.vt.edu/api/v1",
    "anthropic": "https://api.anthropic.com/v1/",
}
MODEL = "gpt-oss-120b"  # ARC model names rotate: check with --list-models

# ---- Budget caps (Step 6+) ---------------------------------------------------
MAX_TOTAL_TOKENS = 5_000_000  # hard stop across all runs (main cap on ARC, where cost is $0)
MAX_USD = 5.00                # hard stop for paid providers
PRICE_PER_MTOK = {"arc": (0.0, 0.0)}  # (input, output) $/1M tokens; fill in for paid providers

# ---- Agent loop (Step 6+) ----------------------------------------------------
MAX_ITERS = 30
SANDBOX_TIMEOUT_S = 10
