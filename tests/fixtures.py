"""Agent-style code strings used by the tests (what a well-behaved agent would send)."""

# Same algorithm as handwritten/reverse.py, written the way the agent must write it:
# self-contained, only numpy/math, using the injected constants TOK, N_VOCAB, MAX_LEN.
REVERSE_OK = '''
CONFIG = dict(d_model=41, n_layers=1, n_heads=1, d_head=16, d_mlp=0)

def write_weights(model):
    POS, SCRATCH = N_VOCAB, N_VOCAB + MAX_LEN
    w = model.w
    for t in range(N_VOCAB):
        w["W_E"][t, t] = 1.0
    for p in range(MAX_LEN):
        w["W_pos"][p, POS + p] = 1.0
    for p in range(6, 11):
        w["L0.W_Q"][0, POS + p, 11 - p] = 20.0
    for j in range(MAX_LEN):
        w["L0.W_K"][0, POS + j, j] = 1.0
    for d in range(10):
        w["L0.W_V"][0, d, d] = 1.0
        w["L0.W_O"][0, d, SCRATCH + d] = 1.0
        w["W_U"][SCRATCH + d, d] = 1.0
'''

# Correct but uses default (bigger) architecture: all-zero weights, scores ~0%.
ZEROS = '''
def write_weights(model):
    print("hello from the sandbox")
'''
