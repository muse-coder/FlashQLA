# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Static checks for autograd ``Function`` signatures.

PyTorch validates that ``Function.backward`` returns exactly as many gradients
as ``Function.forward`` received non-``ctx`` inputs; mismatches raise at
``.backward()`` time.

These tests parse the source files with ``ast`` instead of importing the
modules so they run on CPU-only / non-Hopper machines.
"""

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CHUNK_DIR = REPO_ROOT / "flash_qla/ops/gated_delta_rule/chunk"
CHUNK_INIT = "flash_qla/ops/gated_delta_rule/chunk/__init__.py"
PPU_INIT = "flash_qla/ops/gated_delta_rule/chunk/ppu/__init__.py"
PPU_OPS = "flash_qla/ops/gated_delta_rule/chunk/ppu/ops.py"


def _parse(rel_path: str) -> ast.Module:
    return ast.parse((REPO_ROOT / rel_path).read_text(encoding="utf-8"))


def _get_class(module: ast.Module, name: str) -> ast.ClassDef:
    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == name:
            return node
    raise AssertionError(f"class {name!r} not found")


def _get_function(module: ast.Module, name: str) -> ast.FunctionDef:
    for node in module.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found")


def _get_method(cls: ast.ClassDef, name: str) -> ast.FunctionDef:
    for node in cls.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"method {name!r} not found on {cls.name}")


def test_chunk_gated_delta_rule_grad_count_matches_forward_inputs():
    """``backward`` must return one gradient per non-``ctx`` input of ``forward``."""
    module = _parse(CHUNK_INIT)
    cls = _get_class(module, "ChunkGatedDeltaRuleFunction")

    fwd = _get_method(cls, "forward")
    fwd_args = fwd.args.args
    assert fwd_args and fwd_args[0].arg == "ctx", (
        "forward must take `ctx` as its first argument"
    )
    n_inputs = len(fwd_args) - 1  # exclude ctx

    bwd = _get_method(cls, "backward")
    returns = [n for n in ast.walk(bwd) if isinstance(n, ast.Return)]
    assert len(returns) == 1, f"expected one Return in backward, got {len(returns)}"
    assert isinstance(returns[0].value, ast.Tuple), (
        "backward must return a tuple literal"
    )
    n_grads = len(returns[0].value.elts)

    assert n_inputs == n_grads, (
        f"backward returns {n_grads} gradients but forward takes {n_inputs} non-ctx "
        f"inputs; PyTorch will raise a count-mismatch error at .backward() time."
    )


def test_public_api_is_defined_only_by_shared_chunk_entry():
    """The shared chunk module is the sole owner of orchestration symbols."""
    expected = {
        "ChunkGatedDeltaRuleFunction": ast.ClassDef,
        "chunk_gated_delta_rule_fwd": ast.FunctionDef,
        "chunk_gated_delta_rule_bwd": ast.FunctionDef,
        "chunk_gated_delta_rule": ast.FunctionDef,
    }
    definitions = {name: [] for name in expected}
    for path in CHUNK_DIR.rglob("*.py"):
        module = ast.parse(path.read_text(encoding="utf-8"))
        for node in module.body:
            name = getattr(node, "name", None)
            if name in expected and isinstance(node, expected[name]):
                definitions[name].append(path.relative_to(REPO_ROOT).as_posix())

    for name, paths in definitions.items():
        assert paths == [CHUNK_INIT], (
            f"{name} must be defined only by {CHUNK_INIT}, found {paths}"
        )


def _qualified_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def test_ppu_ops_has_no_legacy_or_autograd_entry():
    module = _parse(PPU_OPS)
    legacy = [
        node.name
        for node in ast.walk(module)
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
        and node.name.startswith("official_chunk_")
    ]
    autograd_classes = [
        node.name
        for node in ast.walk(module)
        if isinstance(node, ast.ClassDef)
        and any(
            _qualified_name(base) == "Function"
            or _qualified_name(base).endswith(".autograd.Function")
            for base in node.bases
        )
    ]
    assert not legacy, f"legacy PPU orchestration remains: {legacy}"
    assert not autograd_classes, f"duplicate PPU autograd Functions remain: {autograd_classes}"


def test_ppu_init_has_no_legacy_orchestration_exports():
    module = _parse(PPU_INIT)
    public_names = {
        alias.asname or alias.name
        for node in module.body
        if isinstance(node, ast.ImportFrom)
        for alias in node.names
    }
    for node in module.body:
        if (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            )
        ):
            public_names.update(
                element.value
                for element in node.value.elts
                if isinstance(element, ast.Constant)
                and isinstance(element.value, str)
            )

    assert not any(name.startswith("official_") for name in public_names)


if __name__ == "__main__":
    test_chunk_gated_delta_rule_grad_count_matches_forward_inputs()
    print("OK")
