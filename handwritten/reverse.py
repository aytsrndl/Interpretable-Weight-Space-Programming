"""Hand-written weights for 5-digit string reversal. No training anywhere.

The algorithm, in plain English
-------------------------------
The sequence is laid out by position like this:

    position:  0    1   2   3   4   5   6   7   8   9   10
    token:     BOS  d1  d2  d3  d4  d5  |   y1  y2  y3  y4

The model predicts the next token at every position. The answer digits are
predicted at positions 6..10:

    at position 6  predict y1 = d5, which sits at position 5
    at position 7  predict y2 = d4, which sits at position 4
    ...
    at position 10 predict y5 = d1, which sits at position 1

So at position p, the answer is "whatever digit is at position 11 - p".
That is one lookup, which one attention head can do:

  1. Query  = "I am at position p, so I am looking for position 11 - p".
  2. Key    = "I am at position j".
  3. The score is large only where j == 11 - p, so attention is effectively
     a hard pointer to that one position.
  4. Value  = "the digit stored at position j". The head copies it into a
     scratch area of the residual stream.
  5. Unembed: read the scratch area and turn it into logits.

No MLP is needed (d_mlp = 0).

Residual stream layout (d_model = 41)
-------------------------------------
    dims  0..14   TOKEN block     one-hot of the current token (15 tokens)
    dims 15..30   POSITION block  one-hot of the current position (16 positions)
    dims 31..40   SCRATCH block   one-hot of the digit copied by the head

Run:  python -m handwritten.reverse
"""

from evaluator import score
from model import MAX_LEN, N_VOCAB, Model, ModelConfig

# ---- Layout constants --------------------------------------------------------
TOKEN = 0                      # TOKEN block starts at dim 0
POS = TOKEN + N_VOCAB          # POSITION block starts at dim 15
SCRATCH = POS + MAX_LEN        # SCRATCH block starts at dim 31
D_MODEL = SCRATCH + 10         # 41: one scratch dim per digit 0..9

SEP_POS = 6                    # where '|' sits; first position that predicts an answer digit
LAST_INPUT_POS = 5             # where d5 sits
N_DIGITS = 5

# How sharp the attention pointer is. Score is SHARP at the target position and 0
# everywhere else. Softmax then gives the target weight ~ 1 - 10*exp(-SHARP),
# about 1 - 2e-8 for SHARP = 20, which is a hard pointer for all practical purposes.
SHARP = 20.0

CONFIG = ModelConfig(
    d_model=D_MODEL,
    n_layers=1,
    n_heads=1,
    d_head=MAX_LEN,  # 16: the head's query/key space is "which position", one dim per position
    d_mlp=0,
)


def write_weights(model: Model) -> None:
    w = model.w  # every array starts at zero; we only write the nonzero entries

    # ---- W_E: token embedding (N_VOCAB x d_model) --------------------------
    # Reads:  the token id.
    # Writes: TOKEN block. Token t sets dim TOKEN + t to 1 (a one-hot copy of the token).
    # Why:    the head needs to read "which digit is here" from a clean, separate subspace.
    for t in range(N_VOCAB):
        w["W_E"][t, TOKEN + t] = 1.0

    # ---- W_pos: position embedding (MAX_LEN x d_model) ---------------------
    # Reads:  the position index.
    # Writes: POSITION block. Position p sets dim POS + p to 1.
    # Why:    the query and key are both built purely from position.
    for p in range(MAX_LEN):
        w["W_pos"][p, POS + p] = 1.0

    # ---- W_Q: query (d_model x d_head), layer 0, head 0 --------------------
    # Reads:  POSITION block only.
    # Writes: head dim (11 - p), scaled by SHARP, for each answer-predicting position p.
    # Why:    the query at position p says "I want position 11 - p". Positions 6..10
    #         ask for 5..1, i.e. d5..d1. All other positions get a zero query, so they
    #         attend uniformly. Their predictions are never used, so that is harmless.
    for p in range(SEP_POS, SEP_POS + N_DIGITS):  # p = 6..10
        target = (SEP_POS + LAST_INPUT_POS) - p     # 11 - p = 5..1
        w["L0.W_Q"][0, POS + p, target] = SHARP

    # ---- W_K: key (d_model x d_head), layer 0, head 0 ----------------------
    # Reads:  POSITION block only.
    # Writes: head dim j for position j (identity map from position to head space).
    # Why:    the key at position j says "I am position j". Then q·k = SHARP exactly
    #         when j == 11 - p, and 0 otherwise.
    #         Causality is fine: target 11 - p is at most 5, always before p (6 or more).
    for j in range(MAX_LEN):
        w["L0.W_K"][0, POS + j, j] = 1.0

    # ---- W_V: value (d_model x d_head), layer 0, head 0 --------------------
    # Reads:  TOKEN block, digit dims only (tokens 0..9).
    # Writes: head dims 0..9, one per digit.
    # Why:    the value is "which digit is at this position". Reusing head dims 0..9
    #         is fine because the value space is separate from the query/key space.
    for d in range(10):
        w["L0.W_V"][0, TOKEN + d, d] = 1.0

    # ---- W_O: output (d_head x d_model), layer 0, head 0 -------------------
    # Reads:  head dims 0..9 (the copied digit).
    # Writes: SCRATCH block. Digit d goes to dim SCRATCH + d.
    # Why:    keeps the copied digit separate from the TOKEN block, which still holds
    #         the token at the current position ('|' or the previous answer digit).
    for d in range(10):
        w["L0.W_O"][0, d, SCRATCH + d] = 1.0

    # ---- W_U: unembedding (d_model x N_VOCAB) ------------------------------
    # Reads:  SCRATCH block only.
    # Writes: logit for digit d = value of SCRATCH + d (about 1 for the copied digit, 0 elsewhere).
    # Why:    the prediction is exactly the copied digit. Nothing else feeds the logits.
    for d in range(10):
        w["W_U"][SCRATCH + d, d] = 1.0


def build() -> Model:
    model = Model(CONFIG)
    write_weights(model)
    return model


if __name__ == "__main__":
    m = build()
    for split in ("visible", "hidden"):
        r = score(m, "reverse", split)
        print(f"reverse [{split:7s}] accuracy = {r.accuracy:.2%} on {r.n} examples; failures: {r.failures}")
    print(f"param_count = {m.param_count()}, nonzero = {m.nonzero_param_count()}")
