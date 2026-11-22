"""Run untrusted, agent-written `write_weights(model)` code safely.

Defense in layers. No single layer is airtight; together they are adequate for this project.

  1. Static check (AST). Before anything runs, reject:
       - imports other than numpy and math
       - dangerous builtins: open, eval, exec, getattr, __import__, ...
       - any name or attribute starting with "_" (blocks __class__.__subclasses__ tricks)
       - numpy file/FFI attributes: np.load, np.save, np.fromfile, np.ctypeslib, ...
  2. Separate process. The code runs in a fresh Python subprocess with:
       - a 10 s timeout (killed after that)
       - an empty temp directory as its working directory
       - a stripped environment: no API keys get inherited
       - restricted builtins inside exec()
       - on Linux/macOS: memory, CPU and file-size limits; on Linux, no network (unshare -rn)
  3. Structural isolation (the real guarantee). The child never imports evaluator.py or
     tasks.py and never sees the test sets. It can only produce a weights file, which
     the parent re-validates (names, shapes, finite values, architecture limits) before
     scoring it with its own, untouched evaluator.

Windows note: the resource limits and network isolation in (2) are not available there,
so (1) and (3) carry the load.
"""

import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import config
from model import Model, ModelConfig

REPO = Path(__file__).resolve().parent
CHILD = REPO / "sandbox_child.py"

ALLOWED_IMPORTS = {"numpy", "math"}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "getattr", "setattr", "delattr",
    "globals", "locals", "vars", "dir", "input", "breakpoint", "exit", "quit", "help",
    "memoryview", "type", "object", "super", "classmethod", "staticmethod", "property",
}
BANNED_ATTRS = {
    # numpy file I/O and foreign-function access
    "load", "save", "savez", "savez_compressed", "fromfile", "tofile", "loadtxt", "savetxt",
    "genfromtxt", "fromregex", "memmap", "ctypeslib", "ctypes", "lib", "DataSource",
    "f2py", "distutils", "testing", "os", "sys",
    # str.format can walk attributes from inside a string: "{0.__class__}".format(x)
    "format", "format_map",
}
MAX_STDOUT = 4000


@dataclass
class SandboxResult:
    ok: bool
    model: Model | None = None
    error: str = ""   # rejection reason or traceback; shown to the agent
    stdout: str = ""  # whatever the code printed (truncated)


# ---- Layer 1: static check -----------------------------------------------------
def check_code(code: str) -> list[str]:
    """Return a list of problems. Empty list = allowed to run."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]

    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS or "." in a.name:
                    problems.append(f"line {node.lineno}: import {a.name} not allowed (only numpy, math)")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or "." in (node.module or ""):
                problems.append(f"line {node.lineno}: from {node.module} import not allowed")
            for a in node.names:
                if a.name in BANNED_ATTRS or a.name.startswith("_") or a.name == "*":
                    problems.append(f"line {node.lineno}: importing {a.name} not allowed")
        elif isinstance(node, ast.Name):
            if node.id in BANNED_NAMES or node.id.startswith("_"):
                problems.append(f"line {node.lineno}: name {node.id!r} not allowed")
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in BANNED_ATTRS:
                problems.append(f"line {node.lineno}: attribute .{node.attr} not allowed")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            problems.append(f"line {node.lineno}: global/nonlocal not allowed")

    defines = any(isinstance(n, ast.FunctionDef) and n.name == "write_weights" for n in tree.body)
    if not defines:
        problems.append("code must define a top-level function write_weights(model)")
    return problems


def validate_arch(arch: dict) -> ModelConfig:
    """Turn an agent-chosen CONFIG into a ModelConfig, enforcing ARCH_LIMITS."""
    if set(arch) != set(config.ARCH_LIMITS):
        raise ValueError(f"CONFIG must have exactly the keys {sorted(config.ARCH_LIMITS)}")
    for key, limit in config.ARCH_LIMITS.items():
        v = arch[key]
        low = 0 if key == "d_mlp" else 1
        if not isinstance(v, int) or isinstance(v, bool) or not low <= v <= limit:
            raise ValueError(f"CONFIG[{key!r}] = {v!r}; must be an int in [{low}, {limit}]")
    return ModelConfig(**arch)


# ---- Layer 2: subprocess ------------------------------------------------------------
@lru_cache(maxsize=1)
def _no_network_prefix() -> tuple[str, ...]:
    """On Linux, run the child in a new network namespace (no interfaces = no network)."""
    if sys.platform.startswith("linux") and shutil.which("unshare"):
        try:
            if subprocess.run(["unshare", "-rn", "true"], capture_output=True, timeout=5).returncode == 0:
                return ("unshare", "-rn")
        except (OSError, subprocess.TimeoutExpired):
            pass
    return ()


def _child_env() -> dict:
    """Only what Python needs to start. Notably, no ARC_API_KEY / ANTHROPIC_API_KEY."""
    keep = ("PATH", "SYSTEMROOT", "TEMP", "TMP", "LANG")
    env = {k: os.environ[k] for k in keep if k in os.environ}
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8")
    return env


def run(code: str, timeout: float = config.SANDBOX_TIMEOUT_S) -> SandboxResult:
    problems = check_code(code)
    if problems:
        return SandboxResult(False, error="Rejected before running:\n" + "\n".join(problems))

    with tempfile.TemporaryDirectory(prefix="iwsp_sbx_") as tmp:
        tmp = Path(tmp)
        (tmp / "agent_code.py").write_text(code, encoding="utf-8")
        cmd = [*_no_network_prefix(), sys.executable, "-I", str(CHILD), str(REPO), str(tmp),
               json.dumps(config.DEFAULT_ARCH), json.dumps(config.ARCH_LIMITS)]
        try:
            proc = subprocess.run(cmd, cwd=tmp, env=_child_env(), capture_output=True,
                                  text=True, encoding="utf-8", errors="replace", timeout=timeout)
        except subprocess.TimeoutExpired:
            return SandboxResult(False, error=f"Timed out after {timeout:.0f} s.")

        stdout = proc.stdout[-MAX_STDOUT:]
        if proc.returncode != 0:
            err = proc.stderr.strip()[-MAX_STDOUT:] or f"child exited with code {proc.returncode}"
            return SandboxResult(False, error=err, stdout=stdout)

        # ---- Layer 3: trust nothing the child wrote ----
        try:
            arch = json.loads((tmp / "arch.json").read_text(encoding="utf-8"))
            model = Model(validate_arch(arch))
            model.load(tmp / "weights.npz")
        except Exception as e:  # noqa: BLE001 — any failure here is the agent's code's fault
            return SandboxResult(False, error=f"Invalid output: {e}", stdout=stdout)
        return SandboxResult(True, model=model, stdout=stdout)
