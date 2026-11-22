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
COMPRESS_AFTER_SOLVED = True  # after 100% visible, keep iterating to shrink param_count
COMPRESS_PATIENCE = 5         # stop compressing after this many iterations with no smaller model
TEMPERATURE = 0.7
SEED = 0                      # sent to the API where supported (ARC serves via vLLM)
MAX_OUTPUT_TOKENS = 16_000    # reasoning models spend output tokens thinking

# ---- Sandbox (Step 5) --------------------------------------------------------
SANDBOX_TIMEOUT_S = 10
# Architecture the agent gets unless its code sets CONFIG = {...}.
DEFAULT_ARCH = dict(d_model=64, n_layers=2, n_heads=2, d_head=16, d_mlp=64)
# Upper bounds on any CONFIG the agent picks, to keep scoring fast.
ARCH_LIMITS = dict(d_model=256, n_layers=4, n_heads=8, d_head=64, d_mlp=512)
