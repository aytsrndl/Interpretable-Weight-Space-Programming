"""A minimal decoder-only transformer whose weights are plain numpy arrays.

There is no training code here and there never will be. Weights start at zero
and are set by hand (or by an agent) through `model.w[name][...] = value`.

Forward pass, for a sequence of T tokens:

    x = W_E[tokens] + W_pos[0..T-1]                    # residual stream, (T, d_model)
    for each layer l:
        # causal self-attention; every head reads the same x, outputs are summed
        x = x + sum over heads h of softmax(causal_mask(q_h @ k_h.T)) @ v_h @ W_O[h]
            where q_h, k_h, v_h = x @ W_Q[h], x @ W_K[h], x @ W_V[h]
        x = x + relu(x @ W_in + b_in) @ W_out + b_out  # MLP (skipped if d_mlp == 0)
    logits = x @ W_U

Deliberate simplifications, so that hand-set numbers mean exactly what they say:
  * No LayerNorm.
  * No 1/sqrt(d_head) scaling: attention scores are exactly q·k.
"""

from dataclasses import dataclass

import numpy as np

# ---- Vocabulary -------------------------------------------------------------
# Digits come first so token id == digit value for '0'..'9'.
VOCAB = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "+", "=", "|", "BOS", "PAD"]
TOK = {s: i for i, s in enumerate(VOCAB)}  # e.g. TOK["|"] == 12
N_VOCAB = len(VOCAB)  # 15
MAX_LEN = 16          # longest sequence any task uses, including the answer


@dataclass
class ModelConfig:
    d_model: int = 41
    n_layers: int = 1
    n_heads: int = 1
    d_head: int = 16
    d_mlp: int = 0  # 0 means "no MLP"


class Model:
    def __init__(self, cfg: ModelConfig):
        self.cfg = cfg
        d, H, dh, dm = cfg.d_model, cfg.n_heads, cfg.d_head, cfg.d_mlp
        shapes = {
            "W_E": (N_VOCAB, d),    # token embedding: row t is what token t writes into x
            "W_pos": (MAX_LEN, d),  # position embedding: row p is what position p writes
            "W_U": (d, N_VOCAB),    # unembedding: column t scores token t
        }
        for l in range(cfg.n_layers):
            shapes[f"L{l}.W_Q"] = (H, d, dh)
            shapes[f"L{l}.W_K"] = (H, d, dh)
            shapes[f"L{l}.W_V"] = (H, d, dh)
            shapes[f"L{l}.W_O"] = (H, dh, d)
            if dm > 0:
                shapes[f"L{l}.W_in"] = (d, dm)
                shapes[f"L{l}.b_in"] = (dm,)
                shapes[f"L{l}.W_out"] = (dm, d)
                shapes[f"L{l}.b_out"] = (d,)
        self.shapes = shapes
        self.w = {name: np.zeros(shape) for name, shape in shapes.items()}

    # ---- bookkeeping ----------------------------------------------------------
    def param_count(self) -> int:
        """Total number of weights, zero or not (the paper's size metric)."""
        return sum(a.size for a in self.w.values())

    def nonzero_param_count(self) -> int:
        """Weights that are actually used. A second, more forgiving size metric."""
        return int(sum(np.count_nonzero(a) for a in self.w.values()))

    # ---- forward pass ---------------------------------------------------------
    def forward(self, tokens: np.ndarray, return_attn: bool = False):
        """tokens: int array (B, T) or (T,).  Returns logits (B, T, N_VOCAB) or (T, N_VOCAB).

        With return_attn=True, returns (logits, attn) where attn[f"L{l}.H{h}"][..., i, j] is how
        much query position i attends to key position j. Useful for debugging hand-set weights.
        """
        tokens = np.asarray(tokens)
        single = tokens.ndim == 1
        if single:
            tokens = tokens[None, :]
        B, T = tokens.shape
        assert T <= MAX_LEN, f"sequence length {T} > MAX_LEN {MAX_LEN}"
        w = self.w

        x = w["W_E"][tokens] + w["W_pos"][:T][None, :, :]  # (B, T, d)
        causal = np.tril(np.ones((T, T), dtype=bool))       # key j visible to query i iff j <= i

        attn = {}  # f"L{l}.H{h}" -> attention weights (B, T, T), row i = where query i looks
        for l in range(self.cfg.n_layers):
            head_out = np.zeros_like(x)
            for h in range(self.cfg.n_heads):  # all heads read the SAME x (run in parallel)
                q = x @ w[f"L{l}.W_Q"][h]  # (B, T, dh)
                k = x @ w[f"L{l}.W_K"][h]
                v = x @ w[f"L{l}.W_V"][h]
                scores = q @ k.transpose(0, 2, 1)  # (B, T, T)
                scores = np.where(causal, scores, -np.inf)
                scores = scores - scores.max(axis=-1, keepdims=True)  # numerical stability
                A = np.exp(scores)
                A = A / A.sum(axis=-1, keepdims=True)
                attn[f"L{l}.H{h}"] = A
                head_out = head_out + (A @ v) @ w[f"L{l}.W_O"][h]
            x = x + head_out
            if self.cfg.d_mlp > 0:
                hidden = np.maximum(0.0, x @ w[f"L{l}.W_in"] + w[f"L{l}.b_in"])
                x = x + hidden @ w[f"L{l}.W_out"] + w[f"L{l}.b_out"]

        logits = x @ w["W_U"]
        if single:
            logits, attn = logits[0], {k: a[0] for k, a in attn.items()}
        return (logits, attn) if return_attn else logits

    def generate(self, prompts: np.ndarray, n_new: int) -> np.ndarray:
        """Greedy decoding. prompts: (B, T0) ints. Returns the n_new generated tokens, (B, n_new).

        Recomputes the whole sequence each step (no KV cache). Slow in theory, but
        sequences here are at most 16 tokens.
        """
        seq = np.asarray(prompts)
        for _ in range(n_new):
            next_tok = self.forward(seq)[:, -1, :].argmax(axis=-1)
            seq = np.concatenate([seq, next_tok[:, None]], axis=1)
        return seq[:, -n_new:]

    # ---- save / load (used by the sandbox later) -----------------------------
    def save(self, path: str) -> None:
        np.savez(path, **self.w)

    def load(self, path: str) -> None:
        """Load weights, refusing anything with the wrong names, shapes, or non-finite values."""
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != set(self.shapes):
                raise ValueError(f"weight names differ: {sorted(set(data.files) ^ set(self.shapes))}")
            for name, shape in self.shapes.items():
                arr = data[name].astype(np.float64)
                if arr.shape != shape:
                    raise ValueError(f"{name}: shape {arr.shape}, expected {shape}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{name}: contains NaN or inf")
                self.w[name] = arr
