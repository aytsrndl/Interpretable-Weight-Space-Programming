"""Step 5: the sandbox runs good code and refuses or contains bad code."""

import pytest

import sandbox
from evaluator import score
from tests.fixtures import REVERSE_OK, ZEROS


def test_good_code_runs_and_scores_100():
    r = sandbox.run(REVERSE_OK)
    assert r.ok, r.error
    assert score(r.model, "reverse", "hidden").accuracy == 1.0


def test_default_arch_and_stdout_captured():
    r = sandbox.run(ZEROS)
    assert r.ok, r.error
    assert r.model.cfg.d_model == 64 and "hello from the sandbox" in r.stdout


# Each of these must be rejected by the static check, before anything runs.
STATIC_ESCAPES = {
    "import os": "import os\ndef write_weights(model): pass",
    "import socket": "import socket\ndef write_weights(model): pass",
    "from numpy import load": "from numpy import load\ndef write_weights(model): pass",
    "numpy submodule": "import numpy.ctypeslib\ndef write_weights(model): pass",
    "open file": "def write_weights(model):\n    open('evaluator.py').read()",
    "__import__": "def write_weights(model):\n    __import__('os')",
    "getattr": "def write_weights(model):\n    getattr(model, 'w')",
    "dunder walk": "def write_weights(model):\n    ().__class__.__bases__[0].__subclasses__()",
    "format walk": "def write_weights(model):\n    '{0.__class__}'.format(model)",
    "np.load": "def write_weights(model):\n    np.load('runs/x.npz')",
    "np.fromfile": "def write_weights(model):\n    np.fromfile('tasks.py')",
    "model.save": "def write_weights(model):\n    model.save('/tmp/x')",
    "exec": "def write_weights(model):\n    exec('import os')",
    "eval": "def write_weights(model):\n    eval('1')",
    "builtins name": "def write_weights(model):\n    __builtins__",
    "no write_weights": "x = 1",
    "syntax error": "def write_weights(model)\n    pass",
}


@pytest.mark.parametrize("name", list(STATIC_ESCAPES))
def test_static_check_rejects(name):
    r = sandbox.run(STATIC_ESCAPES[name])
    assert not r.ok and r.error.startswith("Rejected before running"), (name, r.error)


def test_infinite_loop_times_out():
    r = sandbox.run("def write_weights(model):\n    while True:\n        pass", timeout=3)
    assert not r.ok and "Timed out" in r.error


def test_runtime_error_returns_traceback():
    r = sandbox.run("def write_weights(model):\n    model.w['W_E'][999, 0] = 1.0")
    assert not r.ok and "IndexError" in r.error


@pytest.mark.parametrize("bad", [
    "def write_weights(model):\n    model.w['W_E'] = np.ones((3, 3))",               # wrong shape
    "def write_weights(model):\n    model.w['W_E'][0, 0] = float('nan')",             # NaN
    "def write_weights(model):\n    model.w['extra'] = np.ones(3)",                  # unknown name
])
def test_bad_weights_rejected(bad):
    r = sandbox.run(bad)
    assert not r.ok, bad


def test_oversized_config_rejected():
    code = ("CONFIG = dict(d_model=10_000, n_layers=1, n_heads=1, d_head=8, d_mlp=0)\n"
            "def write_weights(model): pass")
    r = sandbox.run(code)
    assert not r.ok and "d_model" in r.error


def test_api_key_not_inherited(monkeypatch):
    """Even if code could read the environment, the key isn't there to read."""
    monkeypatch.setenv("ARC_API_KEY", "sk-secret")
    assert "ARC_API_KEY" not in sandbox._child_env()


def test_memory_bomb_contained():
    r = sandbox.run("def write_weights(model):\n    x = np.ones((100_000, 100_000))", timeout=10)
    assert not r.ok
