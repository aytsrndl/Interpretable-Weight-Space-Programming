"""Checks for Steps 1–4. Run: python -m pytest -q"""

import numpy as np
import pytest

from evaluator import score
from handwritten import reverse
from model import MAX_LEN, N_VOCAB, Model, ModelConfig
from tasks import TASKS, dataset


# ---- model ------------------------------------------------------------------
def test_shapes_and_param_count():
    m = Model(ModelConfig(d_model=8, n_layers=2, n_heads=2, d_head=4, d_mlp=6))
    logits = m.forward(np.zeros((3, 10), dtype=int))
    assert logits.shape == (3, 10, N_VOCAB)
    assert m.param_count() == sum(int(np.prod(s)) for s in m.shapes.values())


def test_attention_is_causal():
    """Changing a later token must not change logits at earlier positions."""
    rng = np.random.default_rng(0)
    m = Model(ModelConfig(d_model=8, n_layers=2, n_heads=2, d_head=4, d_mlp=6))
    for k in m.w:
        m.w[k] = rng.normal(size=m.w[k].shape)
    a = rng.integers(0, N_VOCAB, size=MAX_LEN)
    b = a.copy()
    b[-1] = (b[-1] + 1) % N_VOCAB
    assert np.allclose(m.forward(a)[:-1], m.forward(b)[:-1])


def test_save_load_roundtrip_and_validation(tmp_path):
    m = reverse.build()
    path = tmp_path / "w.npz"
    m.save(path)
    m2 = Model(reverse.CONFIG)
    m2.load(path)
    assert all(np.array_equal(m.w[k], m2.w[k]) for k in m.w)

    bad = dict(m.w)
    bad["W_U"] = np.full_like(bad["W_U"], np.nan)
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError):
        m2.load(tmp_path / "bad.npz")


# ---- tasks ------------------------------------------------------------------
@pytest.mark.parametrize("name", list(TASKS))
def test_datasets_deterministic_and_disjoint(name):
    vis, hid = dataset(name, "visible"), dataset(name, "hidden")
    assert vis == dataset(name, "visible")
    assert len(vis) == 200 and len(hid) == 2000
    assert not {x for x, _ in vis} & {x for x, _ in hid}


@pytest.mark.parametrize("name", list(TASKS))
def test_targets_are_correct(name):
    task = TASKS[name]
    for x, y in dataset(name, "visible"):
        assert len(x) == task.input_len and len(y) == task.output_len
        if name == "reverse":
            assert y == x[::-1]
        elif name == "sort":
            assert y == "".join(sorted(x))
        else:
            a, b = x.split("+")
            assert int(y) == int(a) + int(b)


# ---- evaluator --------------------------------------------------------------
def test_zero_model_scores_low_and_reports_failures():
    m = Model(reverse.CONFIG)  # all weights zero -> always predicts token 0
    r = score(m, "reverse")
    assert r.accuracy < 0.01
    assert len(r.failures) == 5 and all(len(f) == 3 for f in r.failures)


def test_evaluator_deterministic():
    m = reverse.build()
    assert score(m, "reverse", "hidden") == score(m, "reverse", "hidden")


# ---- step 4 -----------------------------------------------------------------
def test_handwritten_reverse_is_perfect():
    m = reverse.build()
    assert score(m, "reverse", "visible").accuracy == 1.0
    assert score(m, "reverse", "hidden").accuracy == 1.0


def test_handwritten_reverse_fails_other_tasks():
    """Sanity: the reverse circuit should not accidentally solve sort."""
    assert score(reverse.build(), "sort", "hidden").accuracy < 0.1
