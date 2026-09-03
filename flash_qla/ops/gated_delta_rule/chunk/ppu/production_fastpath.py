"""Native inference fast paths for the traced Qwen GDN production shapes.

The kernels implement the same gated-delta recurrence as the official PPU
backend.  Dispatch is deliberately narrow: unsupported layouts, dtypes,
shapes, or autograd calls return ``None`` and use the general implementation.
"""

from __future__ import annotations

import os
import tempfile
import weakref
from pathlib import Path

import torch


_BT = 64
_NF = 6
_KD = 128
_VD = 128
_ALL_STAGES = 2047
_VB_SEL = 32
_VAR = 18
_WHV = 1
_MP = 6
_PGRP = 34
_PI = 13
_PRV = 256
_UV = 9
_SF = 0
_SO = 7
_GS = 2
_EXACT_LENGTHS = frozenset((764, 769, 1280, 15360))
_MIXED_LENGTH = 1024
_MIXED_SHORT_REQUESTS = 255

_CSRC = Path(__file__).with_name("csrc")
_GDR_SOURCE = _CSRC / "production_fastpath_gdr.cu"
_LEN1_SOURCE = _CSRC / "production_fastpath_len1.cu"
_EXTENSIONS: dict[int | str, object] = {}
_PLANS: dict[tuple, tuple[torch.Tensor, ...]] = {}
_VALIDATED_CU: list[tuple[torch.Tensor, int | None, tuple[int, ...]]] = []
_TAIL_CU: dict[torch.device, torch.Tensor] = {}
_HOT_INPUTS: tuple[weakref.ReferenceType[torch.Tensor], ...] | None = None
_HOT_OPTIONS: tuple[object, ...] | None = None
_HOT_RUNNER = None


def _build_root() -> Path:
    root = Path(
        os.environ.get(
            "FLASHQLA_PPU_FASTPATH_BUILD_DIR",
            Path(tempfile.gettempdir()) / "flashqla_ppu_fastpath",
        )
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _load_extension(length: int | str):
    extension = _EXTENSIONS.get(length)
    if extension is not None:
        return extension

    from torch.utils.cpp_extension import load

    if length == "len1":
        name = "flashqla_production_len1_ppu15_v1"
        source = _LEN1_SOURCE
        flags = [
            "-O3",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "--use_fast_math",
        ]
    else:
        name = f"flashqla_production_gdr_exact_{length}_ppu15_v1"
        source = _GDR_SOURCE
        flags = [
            "-O3",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "--use_fast_math",
            f"-DFLASHQLA_EXACT_T={length}",
        ]
    build_directory = _build_root() / name
    build_directory.mkdir(parents=True, exist_ok=True)
    extension = load(
        name=name,
        sources=[str(source)],
        extra_cuda_cflags=flags,
        build_directory=str(build_directory),
        verbose=False,
    )
    _EXTENSIONS[length] = extension
    return extension


def _plan(q: torch.Tensor, v: torch.Tensor, initial_state: torch.Tensor):
    length = int(q.shape[1])
    signature = (
        length,
        int(q.shape[2]),
        int(v.shape[2]),
        int(initial_state.shape[0]),
        q.device,
    )
    plan = _PLANS.get(signature)
    if plan is not None:
        return plan

    hq = int(q.shape[2])
    hv = int(v.shape[2])
    requests = int(initial_state.shape[0])
    chunks = (length + _BT - 1) // _BT
    scratch_chunks = chunks + requests
    tg = torch.empty(
        scratch_chunks * hv * _BT * _BT,
        dtype=torch.bfloat16,
        device=q.device,
    )
    rg = torch.empty_like(tg)
    fg = torch.empty(
        scratch_chunks * hv * _NF * _BT,
        dtype=torch.float32,
        device=q.device,
    )
    kn = torch.empty(
        length * hq * _KD, dtype=torch.bfloat16, device=q.device
    )
    qn = torch.empty_like(kn)
    pgrp = _PGRP | ((_PRV & 1023) << 8) | ((_PI & 15) << 26)
    var_word = _VAR | (_WHV << 8) | (_MP << 16)
    uv_word = _UV | (_SF << 8) | (_SO << 16) | (_GS << 24)
    plan = (tg, rg, fg, kn, qn, pgrp, var_word, uv_word)
    _PLANS[signature] = plan
    return plan


def _make_exact_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
):
    extension = _load_extension(int(q.shape[1]))
    tg, rg, fg, kn, qn, pgrp, var_word, uv_word = _plan(
        q, v, initial_state
    )

    def run(q, k, v, g, beta, initial_state, cu_seqlens):
        return extension.forward_split(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            cu_seqlens,
            tg,
            rg,
            fg,
            kn,
            qn,
            _VB_SEL,
            _ALL_STAGES,
            var_word,
            pgrp,
            uv_word,
        )

    return run


def _cu_matches(cu_seqlens: torch.Tensor, expected: tuple[int, ...]) -> bool:
    try:
        version = cu_seqlens._version
    except RuntimeError:
        # Tensors constructed inside ``torch.inference_mode`` intentionally do
        # not carry version counters. Keep the tensor itself alive in the
        # bounded cache so identity cannot be recycled beneath this entry.
        version = None
    for tensor, cached_version, cached_values in _VALIDATED_CU:
        if tensor is cu_seqlens and cached_version == version:
            return cached_values == expected
    values = tuple(int(value) for value in cu_seqlens.tolist())
    if len(_VALIDATED_CU) >= 8:
        _VALIDATED_CU.pop(0)
    _VALIDATED_CU.append((cu_seqlens, version, values))
    return values == expected


def _common_contract(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
) -> bool:
    if initial_state is None or cu_seqlens is None:
        return False
    if any(tensor.requires_grad for tensor in (q, k, v, g, beta, initial_state)):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        return False
    if v.dtype != torch.bfloat16 or beta.dtype != torch.bfloat16:
        return False
    if g.dtype not in (torch.bfloat16, torch.float32):
        return False
    if initial_state.dtype not in (torch.bfloat16, torch.float32):
        return False
    if cu_seqlens.dtype != torch.int32:
        return False
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return False
    length = int(q.shape[1])
    if tuple(q.shape) != (1, length, 4, _KD):
        return False
    if tuple(k.shape) != tuple(q.shape):
        return False
    if tuple(v.shape) != (1, length, 16, _VD):
        return False
    if tuple(g.shape) != (1, length, 16):
        return False
    if tuple(beta.shape) != tuple(g.shape):
        return False
    requests = int(cu_seqlens.numel()) - 1
    if tuple(initial_state.shape) != (requests, 16, _VD, _KD):
        return False
    tensors = (q, k, v, g, beta, initial_state, cu_seqlens)
    if any(not tensor.is_cuda or not tensor.is_contiguous() for tensor in tensors):
        return False
    if any(tensor.device != q.device for tensor in tensors[1:]):
        return False
    return torch.cuda.get_device_capability(q.device) == (8, 9)


def _make_mixed_runner(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
):
    len1_extension = _load_extension("len1")
    tail_cu = _TAIL_CU.get(q.device)
    if tail_cu is None:
        tail_cu = torch.tensor([0, 769], dtype=torch.int32, device=q.device)
        _TAIL_CU[q.device] = tail_cu
    q_tail = q[:, _MIXED_SHORT_REQUESTS :]
    v_tail = v[:, _MIXED_SHORT_REQUESTS :]
    state_tail = initial_state[_MIXED_SHORT_REQUESTS :]
    tail_extension = _load_extension(769)
    tg, rg, fg, kn, qn, pgrp, var_word, uv_word = _plan(
        q_tail, v_tail, state_tail
    )

    def run(q, k, v, g, beta, initial_state, _cu_seqlens):
        output = torch.empty_like(v)
        final_state = torch.empty_like(initial_state, dtype=torch.float32)
        len1_extension.forward_len1_into(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            output,
            final_state,
            _MIXED_SHORT_REQUESTS,
        )
        tail_output, tail_state = tail_extension.forward_split(
            q[:, _MIXED_SHORT_REQUESTS :],
            k[:, _MIXED_SHORT_REQUESTS :],
            v[:, _MIXED_SHORT_REQUESTS :],
            g[:, _MIXED_SHORT_REQUESTS :],
            beta[:, _MIXED_SHORT_REQUESTS :],
            initial_state[_MIXED_SHORT_REQUESTS :],
            tail_cu,
            tg,
            rg,
            fg,
            kn,
            qn,
            _VB_SEL,
            _ALL_STAGES,
            var_word,
            pgrp,
            uv_word,
        )
        output[:, _MIXED_SHORT_REQUESTS :].copy_(tail_output)
        final_state[_MIXED_SHORT_REQUESTS :].copy_(tail_state)
        return output, final_state

    return run


def try_production_fastpath(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None,
    initial_state: torch.Tensor | None,
    output_final_state: bool,
    use_qk_l2norm_in_kernel: bool,
    cu_seqlens: torch.Tensor | None,
    head_first: bool,
    state_v_first: bool,
):
    """Return the native result, or ``None`` when the call is not specialised."""
    global _HOT_INPUTS, _HOT_OPTIONS, _HOT_RUNNER

    inputs = (q, k, v, g, beta, initial_state, cu_seqlens)
    options = (
        scale,
        output_final_state,
        use_qk_l2norm_in_kernel,
        head_first,
        state_v_first,
    )
    if (
        head_first
        or not state_v_first
        or not output_final_state
        or not use_qk_l2norm_in_kernel
    ):
        return None
    if scale is not None and float(scale) != _KD**-0.5:
        return None
    if not _common_contract(q, k, v, g, beta, initial_state, cu_seqlens):
        return None
    assert initial_state is not None
    assert cu_seqlens is not None

    length = int(q.shape[1])
    requests = int(initial_state.shape[0])
    if length in _EXACT_LENGTHS and requests == 1:
        expected = (0, length)
        make_runner = lambda: _make_exact_runner(
            q, k, v, g, beta, initial_state, cu_seqlens
        )
    elif (
        length == _MIXED_LENGTH
        and requests == 256
        and g.dtype == torch.bfloat16
        and initial_state.dtype == torch.bfloat16
    ):
        expected = tuple(range(256)) + (_MIXED_LENGTH,)
        make_runner = lambda: _make_mixed_runner(
            q, k, v, g, beta, initial_state
        )
    else:
        return None

    if not _cu_matches(cu_seqlens, expected):
        return None
    if (
        _HOT_INPUTS is not None
        and _HOT_OPTIONS == options
        and all(reference() is actual for reference, actual in zip(_HOT_INPUTS, inputs))
    ):
        return _HOT_RUNNER(*inputs)

    runner = make_runner()
    _HOT_INPUTS = tuple(weakref.ref(tensor) for tensor in inputs)
    _HOT_OPTIONS = options
    _HOT_RUNNER = runner
    return runner(*inputs)


__all__ = ["try_production_fastpath"]
