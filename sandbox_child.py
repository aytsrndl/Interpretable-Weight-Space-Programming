"""Runs inside the sandbox subprocess. Started only by sandbox.run(); never import this.

argv: repo_dir work_dir default_arch_json arch_limits_json

Executes work_dir/agent_code.py with restricted builtins, builds a Model from the
code's optional CONFIG, calls write_weights(model), and writes
work_dir/arch.json + work_dir/weights.npz. The parent re-validates both.
"""

import builtins
import json
import math
import sys

repo_dir, work_dir, default_arch, arch_limits = sys.argv[1:5]
default_arch, arch_limits = json.loads(default_arch), json.loads(arch_limits)


def _set_limits():
    try:
        import resource  # POSIX only; absent on Windows
    except ImportError:
        return
    gb = 1024 ** 3
    for lim, val in ((resource.RLIMIT_AS, 2 * gb), (resource.RLIMIT_FSIZE, 64 * 1024 ** 2),
                     (resource.RLIMIT_CPU, 15)):
        try:
            resource.setrlimit(lim, (val, val))
        except (ValueError, OSError):
            pass


_set_limits()

import numpy as np  # noqa: E402  (imported after limits so numpy is loaded under them)

sys.path.insert(0, repo_dir)
from model import MAX_LEN, N_VOCAB, TOK, VOCAB, Model, ModelConfig  # noqa: E402

# Only model.py is importable from the repo. evaluator.py and tasks.py are never loaded here.

_SAFE = ["abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "int",
         "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed",
         "round", "set", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError",
         "AssertionError", "IndexError", "KeyError", "True", "False", "None", "__build_class__"]


def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name not in ("numpy", "math") or level != 0:
        raise ImportError(f"import {name} not allowed")
    return {"numpy": np, "math": math}[name]


safe_builtins = {k: getattr(builtins, k) for k in _SAFE}
safe_builtins["__import__"] = _restricted_import

namespace = {
    "__builtins__": safe_builtins, "__name__": "agent_code",
    "np": np, "numpy": np, "math": math,
    "VOCAB": VOCAB, "TOK": TOK, "N_VOCAB": N_VOCAB, "MAX_LEN": MAX_LEN,
}

with open(f"{work_dir}/agent_code.py", encoding="utf-8") as f:
    code = f.read()
exec(compile(code, "agent_code.py", "exec"), namespace)

arch = namespace.get("CONFIG", default_arch)
if not isinstance(arch, dict) or set(arch) != set(arch_limits):
    raise ValueError(f"CONFIG must be a dict with exactly the keys {sorted(arch_limits)}")
for key, limit in arch_limits.items():
    v = arch[key]
    low = 0 if key == "d_mlp" else 1
    if not isinstance(v, int) or isinstance(v, bool) or not low <= v <= limit:
        raise ValueError(f"CONFIG[{key!r}] = {v!r}; must be an int in [{low}, {limit}]")

model = Model(ModelConfig(**arch))
namespace["write_weights"](model)

with open(f"{work_dir}/arch.json", "w", encoding="utf-8") as f:
    json.dump({k: int(v) for k, v in arch.items()}, f)
np.savez(f"{work_dir}/weights.npz", **{k: np.asarray(v, dtype=np.float64) for k, v in model.w.items()})
