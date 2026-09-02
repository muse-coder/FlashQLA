# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

"""Correctness checks for the PPU implementation of the official equations."""

import os

import pytest
import torch


os.environ.setdefault("FLASHQLA_BACKEND", "ppu")

from flash_qla import (  # noqa: E402
    chunk_gated_delta_rule,
    chunk_gated_delta_rule_bwd,
    chunk_gated_delta_rule_fwd,
)
import flash_qla.ops.gated_delta_rule.chunk as chunk_backend  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk.ppu import (  # noqa: E402
    official_chunk_backward,
    official_chunk_forward,
    official_chunk_gated_delta_rule,
)
from flash_qla.ops.gated_delta_rule.chunk.ppu import native  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk.ppu import production_fastpath  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk.ppu import torch_backend  # noqa: E402
from flash_qla.ops.gated_delta_rule.chunk.ppu.backward import (  # noqa: E402
    torch_cumsum,
    torch_chunk_dqkwg_bwd,
    torch_chunk_dv_bwd,
    torch_chunk_gdr_bwd,
    torch_chunk_gdr_fwd,
    torch_chunk_wy_bwd,
    torch_w_u_fwd,
)


def _relative_error(actual, expected):
    return ((actual.float() - expected.float()).norm() / expected.float().norm()).item()


def test_ppu_top_level_api_uses_shared_chunk_entry():
    expected_module = "flash_qla.ops.gated_delta_rule.chunk"
    assert chunk_gated_delta_rule.__module__ == expected_module
    assert chunk_gated_delta_rule_fwd.__module__ == expected_module
    assert chunk_gated_delta_rule_bwd.__module__ == expected_module


class _FakeFastPathTensor:
    """Weak-referenceable tensor stand-in for dispatch-only tests."""

    def __init__(self, shape):
        self.shape = shape


def test_ppu_production_fastpath_reuses_exact_shape_runner(monkeypatch):
    tensors = (
        _FakeFastPathTensor((1, 764, 4, 128)),
        _FakeFastPathTensor((1, 764, 4, 128)),
        _FakeFastPathTensor((1, 764, 16, 128)),
        _FakeFastPathTensor((1, 764, 16)),
        _FakeFastPathTensor((1, 764, 16)),
        _FakeFastPathTensor((1, 16, 128, 128)),
        _FakeFastPathTensor((2,)),
    )
    sentinel = object()
    made_runners = []
    runner_calls = []

    def make_runner(*args):
        made_runners.append(args)

        def run(*run_args):
            runner_calls.append(run_args)
            return sentinel

        return run

    monkeypatch.setattr(production_fastpath, "_HOT_INPUTS", None)
    monkeypatch.setattr(production_fastpath, "_HOT_OPTIONS", None)
    monkeypatch.setattr(production_fastpath, "_HOT_RUNNER", None)
    monkeypatch.setattr(production_fastpath, "_common_contract", lambda *args: True)
    monkeypatch.setattr(production_fastpath, "_cu_matches", lambda *args: True)
    monkeypatch.setattr(production_fastpath, "_make_exact_runner", make_runner)

    kwargs = {
        "scale": None,
        "initial_state": tensors[5],
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": tensors[6],
        "head_first": False,
        "state_v_first": True,
    }
    result = production_fastpath.try_production_fastpath(*tensors[:5], **kwargs)
    cached_result = production_fastpath.try_production_fastpath(
        *tensors[:5], **kwargs
    )

    assert result is sentinel
    assert cached_result is sentinel
    assert len(made_runners) == 1
    assert runner_calls == [tensors, tensors]


def test_ppu_production_fastpath_rejects_unsupported_options_before_probe(
    monkeypatch,
):
    def unexpected_probe(*args):
        raise AssertionError("unsupported calls must not probe the PPU runtime")

    monkeypatch.setattr(production_fastpath, "_HOT_INPUTS", None)
    monkeypatch.setattr(production_fastpath, "_common_contract", unexpected_probe)
    tensors = tuple(_FakeFastPathTensor(()) for _ in range(5))

    assert production_fastpath.try_production_fastpath(
        *tensors,
        scale=None,
        initial_state=None,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=None,
        head_first=True,
        state_v_first=True,
    ) is None


def test_ppu_public_api_falls_back_when_shape_is_not_specialized(monkeypatch):
    sentinel = object()
    fallback_kwargs = {}
    tensors = tuple(object() for _ in range(5))

    monkeypatch.setattr(
        chunk_backend,
        "_try_ppu_production_fastpath",
        lambda **kwargs: None,
    )

    def fallback(**kwargs):
        fallback_kwargs.update(kwargs)
        return sentinel

    monkeypatch.setattr(chunk_backend, "_ppu_chunk_gated_delta_rule", fallback)

    result = chunk_backend.chunk_gated_delta_rule(
        *tensors,
        scale=0.125,
        output_final_state=True,
        auto_cp=False,
        enable_fwd_cp_cache=False,
    )

    assert result is sentinel
    assert fallback_kwargs["q"] is tensors[0]
    assert fallback_kwargs["scale"] == 0.125
    assert fallback_kwargs["output_final_state"] is True
    assert fallback_kwargs["auto_cp"] is False
    assert fallback_kwargs["enable_fwd_cp_cache"] is False


@pytest.mark.parametrize(
    ("chunks", "heads", "expected"),
    [
        (28, 4, 28),
        (32, 4, 4),
        (64, 4, 4),
        (64, 8, 16),
        (80, 4, 16),
        (128, 4, 16),
        (256, 4, 16),
        (320, 4, 20),
    ],
)
def test_ppu_backward_autocp_uses_profiled_segment_size(
    monkeypatch, chunks, heads, expected
):
    monkeypatch.delenv("FLASHQLA_PPU_BWD_CP_LOCAL_CHUNKS", raising=False)
    monkeypatch.delenv("FLASHQLA_PPU_CP_LOCAL_CHUNKS", raising=False)
    assert torch_backend._auto_cp_backward_local_chunks(
        chunks, heads, None
    ) == expected


@pytest.mark.gpu
def test_ppu_fused_backward_input_cast_matches_torch_exactly():
    if not native.is_flash_qla_backward_cast_available():
        pytest.skip("PPU fused backward-input cast kernel is unavailable")
    torch.manual_seed(3)
    batch, tokens, q_heads, value_heads = 2, 128, 2, 4
    q = torch.randn(
        batch, tokens, q_heads, 128, device="cuda", dtype=torch.bfloat16
    )
    k = torch.randn_like(q)
    v = torch.randn(
        batch, tokens, value_heads, 128,
        device="cuda", dtype=torch.bfloat16,
    )
    A = torch.randn(
        batch, tokens, value_heads, 64,
        device="cuda", dtype=torch.bfloat16,
    )
    do = torch.randn_like(v)
    actual = native.prepare_backward_inputs_bf16_128(q, k, v, A, do)
    for converted, original in zip(actual, (q, k, v, A, do)):
        torch.testing.assert_close(converted, original.float(), rtol=0, atol=0)


@pytest.mark.gpu
def test_ppu_vectorized_uncumsum_matches_chunk_reference():
    torch.manual_seed(5)
    raw = torch.randn(2, 137, 4, device="cuda", dtype=torch.float32)
    cumulative = torch.cat(
        [
            raw[:, start : start + 64].cumsum(dim=1)
            for start in range(0, raw.shape[1], 64)
        ],
        dim=1,
    )
    expected = torch.cat(
        [
            torch.cat(
                (chunk[:, :1], chunk[:, 1:] - chunk[:, :-1]),
                dim=1,
            )
            for chunk in cumulative.split(64, dim=1)
        ],
        dim=1,
    )

    actual = torch_backend._uncumsum_gate(cumulative, None)

    assert torch.equal(actual, expected)


def _recurrent_reference(q, k, v, g, beta, scale, initial_state=None):
    num_v_heads = v.shape[2]
    q = q.float().repeat_interleave(num_v_heads // q.shape[2], dim=2)
    k = k.float().repeat_interleave(num_v_heads // k.shape[2], dim=2)
    if initial_state is None:
        state = torch.zeros(
            q.shape[0],
            num_v_heads,
            q.shape[-1],
            v.shape[-1],
            device=q.device,
            dtype=torch.float32,
        )
    else:
        state = initial_state.float().clone()
    outputs = []
    for token in range(q.shape[1]):
        decay = torch.exp(g[:, token].float())
        prediction = torch.einsum("bhk,bhkv->bhv", k[:, token], state)
        residual = v[:, token].float() - decay[..., None] * prediction
        state = decay[..., None, None] * state + torch.einsum(
            "bhk,bhv->bhkv",
            k[:, token],
            beta[:, token].float()[..., None] * residual,
        )
        outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, token] * scale, state))
    return torch.stack(outputs, dim=1), state


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ppu_official_forward_matches_recurrence(dtype):
    torch.manual_seed(7)
    batch, tokens, q_heads, v_heads, key_dim, value_dim = 1, 129, 2, 4, 32, 24
    q = torch.randn(batch, tokens, q_heads, key_dim, dtype=dtype) / key_dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, key_dim), dim=-1
    ).to(dtype)
    v = torch.randn(batch, tokens, v_heads, value_dim, dtype=dtype) / value_dim**0.5
    g = -0.01 * torch.rand(batch, tokens, v_heads)
    beta = torch.sigmoid(torch.randn(batch, tokens, v_heads))
    scale = key_dim**-0.5

    expected_o, expected_state = _recurrent_reference(q, k, v, g, beta, scale)
    actual_o, actual_state = chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        scale=scale,
        output_final_state=True,
        auto_cp=False,
    )

    assert _relative_error(actual_o, expected_o) < 5e-3
    assert _relative_error(actual_state, expected_state) < 5e-4


def test_ppu_low_level_kkt_and_backward_graph():
    torch.manual_seed(11)
    batch, tokens, heads, key_dim, value_dim = 1, 65, 2, 16, 12
    q = torch.randn(batch, tokens, heads, key_dim, dtype=torch.float16, requires_grad=True)
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, key_dim), dim=-1
    ).to(torch.float16).requires_grad_(True)
    v = torch.randn(
        batch, tokens, heads, value_dim, dtype=torch.float16, requires_grad=True
    )
    g = (-0.01 * torch.rand(batch, tokens, heads)).requires_grad_(True)
    beta = torch.sigmoid(torch.randn(batch, tokens, heads)).requires_grad_(True)

    g_cumsum, a, output, _, final_state, cp_cache = official_chunk_forward(
        q, k, v, g, beta, output_final_state=True, auto_cp=False
    )
    do = torch.randn_like(output)
    dht = torch.randn_like(final_state)
    expected_grads = torch.autograd.grad(
        (output, final_state),
        (q, k, v, g, beta),
        (do, dht),
    )
    backward_result = official_chunk_backward(
        q.detach(),
        k.detach(),
        v.detach(),
        g_cumsum.detach(),
        beta.detach(),
        a.detach(),
        do,
        dht=dht,
        auto_cp=False,
    )
    actual_grads = (
        backward_result[0],
        backward_result[1],
        backward_result[2],
        backward_result[4],
        backward_result[3],
    )

    assert g_cumsum.shape == g.shape
    assert a.shape == (batch, tokens, heads, 64)
    assert cp_cache is None
    assert all(torch.isfinite(grad).all() for grad in actual_grads)
    relative_errors = [
        _relative_error(actual, expected)
        for actual, expected in zip(actual_grads, expected_grads)
    ]
    # CPU FP16 matmul reduction order varies slightly across Torch/platform
    # builds; keep this reference-only tolerance tight but platform-stable.
    assert all(error < 3e-5 for error in relative_errors), relative_errors


def test_ppu_varlen_and_v_first_state_match_independent_sequences():
    torch.manual_seed(13)
    lengths = (65, 73)
    tokens = sum(lengths)
    heads, key_dim, value_dim = 2, 16, 12
    q = torch.randn(1, tokens, heads, key_dim, dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(1, tokens, heads, key_dim), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(1, tokens, heads, value_dim, dtype=torch.bfloat16)
    g = -0.01 * torch.rand(1, tokens, heads)
    beta = torch.sigmoid(torch.randn(1, tokens, heads))
    h0 = torch.randn(len(lengths), heads, key_dim, value_dim)
    cu_seqlens = torch.tensor((0, lengths[0], tokens), dtype=torch.int32)

    actual_o, actual_state = official_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=h0.transpose(-1, -2).contiguous(),
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        state_v_first=True,
        auto_cp=False,
    )
    expected_outputs = []
    expected_states = []
    for sequence, (left, right) in enumerate(
        zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist())
    ):
        output, state = _recurrent_reference(
            q[:, left:right],
            k[:, left:right],
            v[:, left:right],
            g[:, left:right],
            beta[:, left:right],
            key_dim**-0.5,
            h0[sequence : sequence + 1],
        )
        expected_outputs.append(output)
        expected_states.append(state)
    expected_o = torch.cat(expected_outputs, dim=1)
    expected_state = torch.cat(expected_states).transpose(-1, -2)

    assert _relative_error(actual_o, expected_o) < 5e-3
    assert _relative_error(actual_state, expected_state) < 5e-4


def test_ppu_gradients_match_independent_recurrence():
    torch.manual_seed(19)
    batch, tokens, heads, key_dim, value_dim = 1, 67, 2, 16, 12
    base_q = torch.randn(batch, tokens, heads, key_dim, dtype=torch.bfloat16)
    base_k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, key_dim), dim=-1
    ).to(torch.bfloat16)
    base_v = torch.randn(batch, tokens, heads, value_dim, dtype=torch.bfloat16)
    base_g = -0.01 * torch.rand(batch, tokens, heads)
    base_beta = torch.sigmoid(torch.randn(batch, tokens, heads))
    output_grad = torch.randn(batch, tokens, heads, value_dim)
    state_grad = torch.randn(batch, heads, key_dim, value_dim)

    actual_inputs = [
        tensor.clone().requires_grad_(True)
        for tensor in (base_q, base_k, base_v, base_g, base_beta)
    ]
    actual_o, actual_state = official_chunk_gated_delta_rule(
        *actual_inputs, output_final_state=True, auto_cp=False
    )
    actual_grads = torch.autograd.grad(
        (actual_o.float(), actual_state),
        actual_inputs,
        (output_grad, state_grad),
    )

    reference_inputs = [
        tensor.clone().requires_grad_(True)
        for tensor in (base_q, base_k, base_v, base_g, base_beta)
    ]
    reference_o, reference_state = _recurrent_reference(
        *reference_inputs, key_dim**-0.5
    )
    reference_grads = torch.autograd.grad(
        (reference_o, reference_state),
        reference_inputs,
        (output_grad, state_grad),
    )

    assert all(
        _relative_error(actual, expected) < 1e-2
        for actual, expected in zip(actual_grads, reference_grads)
    )


@pytest.mark.gpu
def test_ppu_auto_cp_matches_sequential_chunks():
    torch.manual_seed(17)
    batch, tokens, q_heads, v_heads, key_dim, value_dim = 1, 1024, 2, 4, 32, 24
    q = torch.randn(
        batch, tokens, q_heads, key_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, key_dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(
        batch, tokens, v_heads, value_dim, device="cuda", dtype=torch.bfloat16
    )
    g = -0.01 * torch.rand(batch, tokens, v_heads, device="cuda")
    beta = torch.sigmoid(torch.randn(batch, tokens, v_heads, device="cuda"))

    expected_o, expected_state = official_chunk_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True, auto_cp=False
    )
    actual_o, actual_state = official_chunk_gated_delta_rule(
        q, k, v, g, beta, output_final_state=True, auto_cp=True
    )

    assert _relative_error(actual_o, expected_o) < 1e-3
    assert _relative_error(actual_state, expected_state) < 1e-5


@pytest.mark.gpu
def test_ppu_gate_warmup_matches_full_fallback(monkeypatch):
    torch.manual_seed(23)
    monkeypatch.setenv("FLASHQLA_PPU_CP_LOCAL_CHUNKS", "8")
    batch, tokens, heads, key_dim, value_dim = 1, 1024, 4, 32, 24
    q = torch.randn(
        batch, tokens, heads, key_dim, device="cuda", dtype=torch.bfloat16
    )
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, key_dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(
        batch, tokens, heads, value_dim, device="cuda", dtype=torch.bfloat16
    )
    g = torch.full(
        (batch, tokens, heads), -0.1, device="cuda", dtype=torch.float32
    )
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    initial_state = torch.randn(
        batch, heads, key_dim, value_dim, device="cuda"
    )

    expected_o, expected_state = official_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        auto_cp=False,
    )
    actual_o, actual_state = official_chunk_gated_delta_rule(
        q,
        k,
        v,
        g,
        beta,
        initial_state=initial_state,
        output_final_state=True,
        auto_cp=True,
    )

    assert _relative_error(actual_o, expected_o) < 1e-3
    assert _relative_error(actual_state, expected_state) < 1e-5


@pytest.mark.gpu
@pytest.mark.parametrize("gate_value", [-0.001, -0.1])
def test_ppu_native_affine_autocp_matches_sequential(
    monkeypatch, gate_value
):
    if not native.is_flash_qla_affine_available():
        pytest.skip("PPU native AutoCP affine kernel is unavailable")
    monkeypatch.setenv("FLASHQLA_PPU_CP_LOCAL_CHUNKS", "4")
    torch.manual_seed(71)
    batch, tokens, q_heads, value_heads, dim = 1, 1024, 2, 4, 128
    q = torch.randn(
        batch, tokens, q_heads, dim, device="cuda", dtype=torch.bfloat16
    ) / dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(
        batch, tokens, value_heads, dim,
        device="cuda", dtype=torch.bfloat16,
    ) / dim**0.5
    g = torch.full(
        (batch, tokens, value_heads), gate_value,
        device="cuda", dtype=torch.float32,
    )
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    )
    initial_state = torch.randn(
        batch, value_heads, dim, dim, device="cuda"
    ) / dim**0.5

    expected_o, expected_state = official_chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=initial_state,
        output_final_state=True,
        auto_cp=False,
    )
    actual_o, actual_state = official_chunk_gated_delta_rule(
        q, k, v, g, beta,
        initial_state=initial_state,
        output_final_state=True,
        auto_cp=True,
    )
    assert _relative_error(actual_o, expected_o) < 1e-2
    assert _relative_error(actual_state, expected_state) < 1e-2


@pytest.mark.gpu
def test_ppu_native_cp_dh_matches_official_equations(monkeypatch):
    if not native.is_flash_qla_cp_dh_backward_available():
        pytest.skip("PPU native AutoCP dH kernel is unavailable")
    torch.manual_seed(83)
    batch, tokens, q_heads, value_heads, dim = 2, 128, 2, 4, 128
    q = (
        torch.randn(batch, tokens, q_heads, dim, device="cuda")
        / dim**0.5
    ).contiguous()
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    ).contiguous()
    A = (
        torch.randn(batch, tokens, value_heads, 64, device="cuda")
        / 64**0.5
    ).contiguous()
    raw_g = -0.01 * torch.rand(
        batch, tokens, value_heads, device="cuda"
    )
    g = raw_g.reshape(batch, 2, 64, value_heads).cumsum(dim=2)
    g = g.reshape_as(raw_g).contiguous()
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    ).contiguous()
    do = (
        torch.randn(batch, tokens, value_heads, dim, device="cuda")
        / dim**0.5
    ).contiguous()

    monkeypatch.setattr(
        native, "is_flash_qla_cp_dh_backward_available", lambda: False
    )
    expected = torch_backend._official_cp_local_dh(
        q, k, A, g, beta, do, dim**-0.5
    )
    monkeypatch.undo()
    actual = torch_backend._official_cp_local_dh(
        q, k, A, g, beta, do, dim**-0.5
    )
    assert _relative_error(actual, expected) < 8e-3


@pytest.mark.gpu
def test_ppu_precomputed_cp_dh_matches_native_kernel(monkeypatch):
    if not native.is_flash_qla_cp_dh_backward_available():
        pytest.skip("PPU native AutoCP dH kernel is unavailable")
    torch.manual_seed(87)
    batch, tokens, q_heads, value_heads, dim = 8, 512, 2, 4, 128
    q = (
        torch.randn(batch, tokens, q_heads, dim, device="cuda")
        / dim**0.5
    ).contiguous()
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    ).contiguous()
    A = (
        torch.randn(batch, tokens, value_heads, 64, device="cuda")
        / 64**0.5
    ).contiguous()
    raw_g = -0.01 * torch.rand(
        batch, tokens, value_heads, device="cuda"
    )
    g = raw_g.reshape(batch, tokens // 64, 64, value_heads)
    g = g.cumsum(dim=2).reshape_as(raw_g).contiguous()
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    ).contiguous()
    do = (
        torch.randn(batch, tokens, value_heads, dim, device="cuda")
        / dim**0.5
    ).contiguous()

    monkeypatch.setenv("FLASHQLA_PPU_PRECOMPUTE_CP_DH", "0")
    expected = torch_backend._official_cp_local_dh(
        q, k, A, g, beta, do, dim**-0.5
    )
    monkeypatch.setenv("FLASHQLA_PPU_PRECOMPUTE_CP_DH", "1")
    actual = torch_backend._official_cp_local_dh(
        q, k, A, g, beta, do, dim**-0.5
    )
    assert _relative_error(actual, expected) < 8e-3


@pytest.mark.gpu
def test_ppu_varlen_autocp_cache_handles_partial_tail(monkeypatch):
    if not native.is_flash_qla_cp_dh_backward_available():
        pytest.skip("PPU native AutoCP dH kernel is unavailable")
    monkeypatch.setenv("FLASHQLA_PPU_BWD_CP_LOCAL_CHUNKS", "4")
    torch.manual_seed(89)
    batch, tokens, q_heads, value_heads, dim = 1, 1857, 2, 4, 128
    q = (
        torch.randn(
            batch, tokens, q_heads, dim,
            device="cuda", dtype=torch.bfloat16,
        )
        / dim**0.5
    )
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = (
        torch.randn(
            batch, tokens, value_heads, dim,
            device="cuda", dtype=torch.bfloat16,
        )
        / dim**0.5
    )
    g = -0.01 * torch.rand(batch, tokens, value_heads, device="cuda")
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    )
    initial_state = (
        torch.randn(1, value_heads, dim, dim, device="cuda")
        / dim**0.5
    )
    cu_seqlens = torch.tensor((0, tokens), device="cuda", dtype=torch.int32)

    expected = official_chunk_forward(
        q, k, v, g, beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        auto_cp=False,
    )
    actual = official_chunk_forward(
        q, k, v, g, beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        output_final_state=True,
        auto_cp=True,
        enable_fwd_cp_cache=True,
    )
    assert actual[5][0] == "ppu_auto_cp_varlen_v1"
    assert actual[5][2][0][0] == "ppu_auto_cp_v1"
    assert _relative_error(actual[2], expected[2]) < 1e-2
    assert _relative_error(actual[4], expected[4]) < 1e-2

    do = torch.randn_like(actual[2])
    dht = torch.randn_like(actual[4]) / dim**0.5
    expected_grads = official_chunk_backward(
        q, k, v, expected[0], beta, expected[1], do,
        dht=dht,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        auto_cp=False,
    )
    actual_grads = official_chunk_backward(
        q, k, v, actual[0], beta, actual[1], do,
        dht=dht,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        auto_cp=True,
        cp_cache=actual[5],
    )
    relative_errors = [
        _relative_error(actual_grad, expected_grad)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads)
    ]
    assert all(error < 1e-2 for error in relative_errors), relative_errors


@pytest.mark.gpu
def test_ppu_native_kkt_inverse_matches_triangular_solve():
    if not native.is_available():
        pytest.skip("PPU native library is unavailable")
    torch.manual_seed(41)
    keys = torch.nn.functional.normalize(
        torch.randn(16, 64, 128, device="cuda"), dim=-1
    )
    beta = torch.sigmoid(torch.randn(16, 64, device="cuda"))
    gram = keys @ keys.transpose(-1, -2)
    identity = torch.eye(64, device="cuda").expand(16, 64, 64)
    lower = identity + torch.tril(beta[..., :, None] * gram, diagonal=-1)
    expected = torch.linalg.solve_triangular(
        lower, identity, upper=False, unitriangular=True
    )
    actual = native.kkt_inverse(gram.contiguous(), beta.contiguous())
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.gpu
def test_ppu_aiu_kkt_inverse_matches_official_equations():
    if not native.is_aiu_available() or not hasattr(
        native, "aiu_kkt_inverse_bf16_64x128"
    ):
        pytest.skip("PPU AIU KKT kernel is unavailable")
    torch.manual_seed(47)
    keys = torch.nn.functional.normalize(
        torch.randn(7, 64, 128, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    beta = torch.sigmoid(torch.randn(7, 64, device="cuda"))
    gram = torch.bmm(
        keys, keys.transpose(-1, -2), out_dtype=torch.float32
    )
    identity = torch.eye(64, device="cuda").expand(7, 64, 64)
    lower = identity + torch.tril(beta[..., :, None] * gram, diagonal=-1)
    expected = torch.linalg.solve_triangular(
        lower, identity, upper=False, unitriangular=True
    )
    actual = native.aiu_kkt_inverse_bf16_64x128(
        keys.contiguous(), beta.contiguous()
    )
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.gpu
def test_ppu_native_gate_beta_preparation_matches_pytorch():
    if not native.is_available() or not hasattr(native, "prepare_gate_beta"):
        pytest.skip("PPU gate/beta preparation kernel is unavailable")
    torch.manual_seed(71)
    batch, tokens, heads = 2, 128, 4
    g = (-0.03 * torch.rand(batch, tokens, heads, device="cuda")).contiguous()
    beta = torch.sigmoid(
        torch.randn(batch, tokens, heads, device="cuda")
    ).contiguous()
    expected_g = g.transpose(1, 2).reshape(batch, heads, tokens // 64, 64)
    expected_g = expected_g.cumsum(dim=-1)
    expected_beta = beta.transpose(1, 2).reshape(
        batch, heads, tokens // 64, 64
    )
    actual_g, actual_beta = native.prepare_gate_beta(g, beta)
    torch.testing.assert_close(actual_g, expected_g, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(actual_beta, expected_beta, rtol=0, atol=0)


@pytest.mark.gpu
def test_ppu_strided_aiu_kkt_reads_original_gva_layout():
    if not native.is_aiu_available() or not hasattr(
        native, "aiu_kkt_solve_bf16_128_strided"
    ):
        pytest.skip("PPU strided AIU KKT kernel is unavailable")
    torch.manual_seed(73)
    batch, tokens, q_heads, value_heads = 2, 128, 2, 4
    keys = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, 128, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    beta = torch.sigmoid(
        torch.randn(batch, value_heads, tokens // 64, 64, device="cuda")
    ).contiguous()
    expanded = keys.repeat_interleave(value_heads // q_heads, dim=2)
    expanded = expanded.transpose(1, 2).reshape(-1, 64, 128).contiguous()
    gram = torch.bmm(
        expanded, expanded.transpose(-1, -2), out_dtype=torch.float32
    )
    flat_beta = beta.reshape(-1, 64)
    identity = torch.eye(64, device="cuda").expand(gram.shape[0], 64, 64)
    lower = identity + torch.tril(flat_beta[..., :, None] * gram, diagonal=-1)
    expected = torch.linalg.solve_triangular(
        lower, identity, upper=False, unitriangular=True
    ).reshape(batch, value_heads, tokens // 64, 64, 64).to(torch.bfloat16)
    actual = native.aiu_kkt_solve_bf16_128_strided(keys, beta)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
def test_ppu_gated_strided_aiu_kkt_matches_official_a():
    if not native.is_gated_strided_kkt_available():
        pytest.skip("PPU gated strided AIU KKT kernel is unavailable")
    torch.manual_seed(79)
    batch, tokens, q_heads, value_heads = 1, 128, 2, 4
    keys = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, 128, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    beta = torch.sigmoid(
        torch.randn(batch, value_heads, tokens // 64, 64, device="cuda")
    ).contiguous()
    gate = (-0.03 * torch.rand_like(beta)).cumsum(dim=-1).contiguous()
    expanded = keys.repeat_interleave(value_heads // q_heads, dim=2)
    expanded = expanded.transpose(1, 2).reshape(-1, 64, 128).contiguous()
    gram = torch.bmm(
        expanded, expanded.transpose(-1, -2), out_dtype=torch.float32
    )
    flat_beta = beta.reshape(-1, 64)
    identity = torch.eye(64, device="cuda").expand(gram.shape[0], 64, 64)
    lower = identity + torch.tril(flat_beta[..., :, None] * gram, diagonal=-1)
    inverse = torch.linalg.solve_triangular(
        lower, identity, upper=False, unitriangular=True
    ).reshape(batch, value_heads, tokens // 64, 64, 64)
    decay = torch.exp(gate[..., :, None] - gate[..., None, :])
    expected = (torch.tril(decay) * inverse).to(torch.bfloat16)
    actual = native.aiu_kkt_solve_bf16_128_strided_gated(keys, beta, gate)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.gpu
def test_ppu_fused_aiu_forward_matches_official_equations(monkeypatch):
    if not native.is_aiu_available():
        pytest.skip("DeepGEMM/Actlize headers were unavailable at build time")
    torch.manual_seed(43)
    batch, tokens, heads, dim = 1, 128, 2, 128
    q = torch.randn(batch, tokens, heads, dim, device="cuda", dtype=torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, heads, dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(
        batch, tokens, heads, dim, device="cuda", dtype=torch.bfloat16
    )
    g = -0.01 * torch.rand(batch, tokens, heads, device="cuda")
    beta = torch.sigmoid(torch.randn(batch, tokens, heads, device="cuda"))
    initial_state = torch.randn(batch, heads, dim, dim, device="cuda") / dim**0.5
    scale = dim**-0.5

    monkeypatch.setenv("FLASHQLA_PPU_AIU_FUSED", "0")
    g_cumsum, a, expected_o, _, expected_state, _ = official_chunk_forward(
        q, k, v, g, beta, scale=scale, initial_state=initial_state,
        output_final_state=True, auto_cp=False,
    )
    chunks = tokens // 64
    actual_o, actual_state = native.flash_qla_fused_forward_bf16_128(
        q,
        k,
        v,
        a.permute(0, 2, 1, 3).reshape(batch, heads, chunks, 64, 64).contiguous(),
        g_cumsum.permute(0, 2, 1).reshape(batch, heads, chunks, 64).contiguous(),
        beta.permute(0, 2, 1).reshape(batch, heads, chunks, 64).contiguous(),
        initial_state.contiguous(),
        scale,
    )
    assert _relative_error(actual_o, expected_o) < 8e-3
    assert _relative_error(actual_state, expected_state) < 8e-3


@pytest.mark.gpu
def test_ppu_fused_aiu_forward_gva_zero_state_and_strong_decay(monkeypatch):
    if not native.is_aiu_available():
        pytest.skip("DeepGEMM/Actlize headers were unavailable at build time")
    torch.manual_seed(59)
    batch, tokens, q_heads, value_heads, dim = 1, 192, 2, 4, 128
    q = torch.randn(
        batch, tokens, q_heads, dim, device="cuda", dtype=torch.bfloat16
    ) / dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn(
        batch, tokens, value_heads, dim,
        device="cuda", dtype=torch.bfloat16,
    ) / dim**0.5
    g = -0.2 * torch.rand(batch, tokens, value_heads, device="cuda")
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    )

    monkeypatch.setenv("FLASHQLA_PPU_AIU_FUSED", "0")
    expected = official_chunk_forward(
        q, k, v, g, beta, output_final_state=True, auto_cp=False
    )
    monkeypatch.setenv("FLASHQLA_PPU_AIU_FUSED", "1")
    actual = official_chunk_forward(
        q, k, v, g, beta, output_final_state=True, auto_cp=False
    )
    assert _relative_error(actual[2], expected[2]) < 8e-3
    assert _relative_error(actual[4], expected[4]) < 8e-3


@pytest.mark.gpu
def test_ppu_native_backward_state_preparation_matches_official_stage():
    if not native.is_flash_qla_chunk_state_available():
        pytest.skip("PPU native backward state preparation is unavailable")
    torch.manual_seed(97)
    batch, tokens, q_heads, value_heads, dim = 1, 128, 2, 4, 128
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    w = torch.randn(batch, tokens, value_heads, dim, device="cuda") / dim**0.5
    u = torch.randn_like(w) / dim**0.5
    raw_g = -0.03 * torch.rand(batch, tokens, value_heads, device="cuda")
    g = raw_g.view(batch, 2, 64, value_heads).cumsum(dim=2).reshape_as(raw_g)
    initial_state = torch.randn(
        batch, value_heads, dim, dim, device="cuda"
    ) / dim**0.5

    expected_h, expected_vn, expected_state = torch_chunk_gdr_fwd(
        k, w, u, g, initial_state, chunk_size=64
    )
    g_chunks = g.view(batch, 2, 64, value_heads).permute(0, 3, 1, 2)
    actual_h, actual_vn, actual_state = (
        native.flash_qla_chunk_state_forward_bf16_128(
            k.contiguous(), w.contiguous(), u.contiguous(),
            g_chunks.contiguous(), initial_state.contiguous(),
        )
    )
    assert _relative_error(actual_h, expected_h) < 8e-3
    assert _relative_error(actual_vn, expected_vn) < 8e-3
    assert _relative_error(actual_state, expected_state) < 8e-3


@pytest.mark.gpu
def test_ppu_fused_prepare_h_matches_official_stages(monkeypatch):
    if not native.is_flash_qla_fused_prepare_h_available():
        pytest.skip("PPU fused prepare-H kernel is unavailable")
    torch.manual_seed(98)
    batch, tokens, q_heads, value_heads, dim = 1, 128, 2, 4, 128
    chunks = tokens // 64
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    v = torch.randn(batch, tokens, value_heads, dim, device="cuda") / dim**0.5
    raw_g = -0.03 * torch.rand(batch, tokens, value_heads, device="cuda")
    g = raw_g.reshape(batch, chunks, 64, value_heads).cumsum(dim=2)
    g = g.reshape_as(raw_g)
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    )
    A = torch.randn(
        batch, tokens, value_heads, 64, device="cuda"
    ) / 64**0.5
    initial_state = torch.randn(
        batch, value_heads, dim, dim, device="cuda"
    ) / dim**0.5

    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_BWD_WU", "0")
    monkeypatch.setenv("FLASHQLA_PPU_HEAD_MAJOR_WU", "0")
    expected_w, expected_u = torch_w_u_fwd(
        k=k, v=v, g=g, beta=beta, A=A
    )
    expected_h, expected_vn, expected_state = torch_chunk_gdr_fwd(
        k, expected_w, expected_u, g, initial_state, chunk_size=64
    )
    g_chunks = g.reshape(batch, chunks, 64, value_heads)
    g_chunks = g_chunks.permute(0, 3, 1, 2).contiguous()
    actual_w, actual_h, actual_vn, actual_state = (
        native.flash_qla_fused_prepare_h_bf16_128(
            k.contiguous(), v.contiguous(), A.contiguous(), g_chunks,
            beta.contiguous(), initial_state.contiguous(),
        )
    )
    for actual, expected in (
        (actual_w, expected_w),
        (actual_h, expected_h),
        (actual_vn, expected_vn),
        (actual_state, expected_state),
    ):
        assert _relative_error(actual, expected) < 8e-3


@pytest.mark.gpu
def test_ppu_head_major_state_backward_matches_official_stage(monkeypatch):
    torch.manual_seed(99)
    batch, tokens, q_heads, value_heads, dim = 2, 128, 2, 4, 128
    q = torch.randn(batch, tokens, q_heads, dim, device="cuda") / dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    w = torch.randn(batch, tokens, value_heads, dim, device="cuda") / dim**0.5
    do = torch.randn_like(w) / dim**0.5
    dv = torch.randn_like(w) / dim**0.5
    raw_g = -0.01 * torch.rand(
        batch, tokens, value_heads, device="cuda"
    )
    g = raw_g.reshape(batch, 2, 64, value_heads).cumsum(dim=2)
    g = g.reshape_as(raw_g)
    h0 = torch.randn(batch, value_heads, dim, dim, device="cuda") / dim**0.5
    dht = torch.randn_like(h0) / dim**0.5

    monkeypatch.setenv("FLASHQLA_PPU_HEAD_MAJOR_STATE_BWD", "0")
    expected = torch_chunk_gdr_bwd(
        q, k, w, g, do, dv.clone(), h0, dht, scale=dim**-0.5
    )
    monkeypatch.setenv("FLASHQLA_PPU_HEAD_MAJOR_STATE_BWD", "1")
    actual = torch_chunk_gdr_bwd(
        q, k, w, g, do, dv.clone(), h0, dht, scale=dim**-0.5
    )
    relative_errors = [
        _relative_error(actual_tensor, expected_tensor)
        for actual_tensor, expected_tensor in zip(actual, expected)
    ]
    assert all(error < 2e-5 for error in relative_errors), relative_errors

@pytest.mark.gpu
def test_ppu_grouped_gqa_state_backward_matches_official_stage(monkeypatch):
    torch.manual_seed(1004)
    batch, tokens, q_heads, value_heads, dim = 2, 128, 2, 4, 128
    q = torch.randn(batch, tokens, q_heads, dim, device="cuda") / dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    w = torch.randn(batch, tokens, value_heads, dim, device="cuda") / dim**0.5
    do = torch.randn_like(w) / dim**0.5
    dv = torch.randn_like(w) / dim**0.5
    raw_g = -0.01 * torch.rand(batch, tokens, value_heads, device="cuda")
    g = raw_g.reshape(batch, 2, 64, value_heads).cumsum(dim=2)
    g = g.reshape_as(raw_g)
    h0 = torch.randn(batch, value_heads, dim, dim, device="cuda") / dim**0.5
    dht = torch.randn_like(h0) / dim**0.5

    monkeypatch.setenv("FLASHQLA_PPU_GROUPED_STATE_BWD", "0")
    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_GROUPED_DV_UPDATE", "0")
    expected = torch_chunk_gdr_bwd(
        q, k, w, g, do, dv.clone(), h0, dht, scale=dim**-0.5
    )
    monkeypatch.setenv("FLASHQLA_PPU_GROUPED_STATE_BWD", "1")
    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_GROUPED_DV_UPDATE", "1")
    actual = torch_chunk_gdr_bwd(
        q, k, w, g, do, dv.clone(), h0, dht, scale=dim**-0.5
    )
    relative_errors = [
        _relative_error(actual_tensor, expected_tensor)
        for actual_tensor, expected_tensor in zip(actual, expected)
    ]
    assert all(error < 2e-5 for error in relative_errors), relative_errors


@pytest.mark.gpu
def test_ppu_native_reverse_chunk_step_matches_official_stage():
    if not native.is_flash_qla_chunk_state_backward_step_available():
        pytest.skip("PPU native reverse chunk-step kernel is unavailable")
    torch.manual_seed(1005)
    batch, tokens, q_heads, value_heads, dim = 2, 128, 2, 4, 128
    chunks = tokens // 64
    q = torch.randn(batch, tokens, q_heads, dim, device="cuda") / dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    w = torch.randn(batch, tokens, value_heads, dim, device="cuda") / dim**0.5
    do = torch.randn_like(w) / dim**0.5
    dv = torch.randn_like(w) / dim**0.5
    raw_g = -0.01 * torch.rand(batch, tokens, value_heads, device="cuda")
    g = raw_g.reshape(batch, chunks, 64, value_heads).cumsum(dim=2)
    g = g.reshape_as(raw_g)
    h0 = torch.randn(batch, value_heads, dim, dim, device="cuda") / dim**0.5
    dht = torch.randn_like(h0) / dim**0.5
    scale = dim**-0.5

    expected_dh, expected_dh0, expected_dv = torch_chunk_gdr_bwd(
        q, k, w, g, do, dv.clone(), h0, dht, scale=scale
    )
    g_chunks = g.reshape(batch, chunks, 64, value_heads)
    g_chunks = g_chunks.permute(0, 3, 1, 2).contiguous()
    actual_dv = dv.clone().contiguous()
    actual_dstate = dht.clone().contiguous()
    actual_dh = torch.empty_like(expected_dh)
    for chunk in reversed(range(chunks)):
        native.flash_qla_chunk_state_backward_step_bf16_128(
            q.contiguous(), k.contiguous(), w.contiguous(), g_chunks,
            do.contiguous(), actual_dv, actual_dstate, actual_dh,
            chunk, scale,
        )

    relative_errors = [
        _relative_error(actual, expected)
        for actual, expected in (
            (actual_dh, expected_dh),
            (actual_dstate, expected_dh0),
            (actual_dv, expected_dv),
        )
    ]
    assert all(error < 8e-3 for error in relative_errors), relative_errors


@pytest.mark.gpu
def test_ppu_fused_reverse_state_update_matches_official_epilogue():
    if not native.is_available() or not hasattr(native, "reverse_state_update"):
        pytest.skip("PPU fused reverse state-update kernel is unavailable")
    torch.manual_seed(100)
    batch, chunks, heads, dim = 2, 3, 4, 128
    state = torch.randn(
        batch, heads, dim, dim, device="cuda", dtype=torch.float32
    ).contiguous()
    inter = torch.randn(
        batch, chunks, heads, dim, dim,
        device="cuda", dtype=torch.float32,
    ).contiguous()
    product = torch.randn_like(state).contiguous()
    decay = torch.rand(
        batch, chunks, heads, device="cuda", dtype=torch.float32
    ).contiguous()
    chunk = 1
    expected = (
        decay[:, chunk, :, None, None] * state
        + inter[:, chunk]
        - product
    )
    actual = native.reverse_state_update(
        state, inter, product, decay, chunk
    )
    assert _relative_error(actual, expected) < 1e-6


@pytest.mark.gpu
def test_ppu_fused_reverse_state_store_matches_official_epilogue():
    if not native.is_available() or not hasattr(
        native, "reverse_state_update_store_inplace"
    ):
        pytest.skip("PPU fused reverse state/history kernel is unavailable")
    torch.manual_seed(1002)
    batch, chunks, heads, dim = 2, 3, 4, 128
    state = torch.randn(
        batch, heads, dim, dim, device="cuda", dtype=torch.float32
    ).contiguous()
    original = state.clone()
    inter = torch.randn(
        batch, chunks, heads, dim, dim,
        device="cuda", dtype=torch.float32,
    ).contiguous()
    product = torch.randn_like(state).contiguous()
    decay = torch.rand(
        batch, chunks, heads, device="cuda", dtype=torch.float32
    ).contiguous()
    history = torch.full_like(inter, torch.nan)
    chunk = 1
    expected = (
        decay[:, chunk, :, None, None] * original
        + inter[:, chunk]
        - product
    )
    actual = native.reverse_state_update_store_inplace(
        state, inter, product, decay, history, chunk
    )
    assert actual.data_ptr() == state.data_ptr()
    assert _relative_error(actual, expected) < 1e-6
    assert _relative_error(history[:, chunk], original) < 1e-6
    assert torch.isnan(history[:, 0]).all()
    assert torch.isnan(history[:, 2]).all()


@pytest.mark.gpu
def test_ppu_native_reverse_chunk_cumsum_matches_torch(monkeypatch):
    if not native.is_available() or not hasattr(native, "reverse_chunk_cumsum"):
        pytest.skip("PPU reverse chunk-cumsum kernel is unavailable")
    torch.manual_seed(1001)
    x = torch.randn(2, 192, 4, device="cuda", dtype=torch.float32)
    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_REVERSE_CUMSUM", "0")
    expected = torch.flip(
        torch.flip(x.reshape(2, 3, 64, 4), dims=(2,)).cumsum(dim=2),
        dims=(2,),
    ).reshape_as(x)
    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_REVERSE_CUMSUM", "1")
    actual = torch_cumsum(x, chunk_size=64, reverse=True)
    assert _relative_error(actual, expected) < 2e-6


@pytest.mark.gpu
def test_ppu_native_wu_matches_official_equations(monkeypatch):
    if not native.is_flash_qla_wu_available():
        pytest.skip("PPU native W/U kernel is unavailable")
    torch.manual_seed(1003)
    batch, tokens, q_heads, value_heads, dim = 2, 128, 2, 4, 128
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    v = torch.randn(batch, tokens, value_heads, dim, device="cuda")
    g = -0.01 * torch.rand(batch, tokens, value_heads, device="cuda")
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    )
    A = torch.randn(
        batch, tokens, value_heads, 64, device="cuda"
    ) / 64**0.5
    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_BWD_WU", "0")
    monkeypatch.setenv("FLASHQLA_PPU_HEAD_MAJOR_WU", "0")
    expected = torch_w_u_fwd(k, v, g, beta, A)
    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_BWD_WU", "1")
    actual = torch_w_u_fwd(k, v, g, beta, A)
    relative_errors = [
        _relative_error(actual_tensor, expected_tensor)
        for actual_tensor, expected_tensor in zip(actual, expected)
    ]
    assert all(error < 8e-3 for error in relative_errors), relative_errors


@pytest.mark.gpu
def test_ppu_native_dqkwg_layout_matches_head_major_views():
    if not native.is_flash_qla_dqkwg_layout_available():
        pytest.skip("PPU native dQKWg layout kernel is unavailable")
    torch.manual_seed(1006)
    batch, tokens, q_heads, value_heads, dim = 1, 128, 2, 4, 128
    chunks = tokens // 64
    q = torch.randn(batch, tokens, q_heads, dim, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(batch, tokens, value_heads, dim, device="cuda")
    do = torch.randn_like(v)
    dv = torch.randn_like(v)
    g = torch.randn(batch, tokens, value_heads, device="cuda")

    actual = native.prepare_dqkwg_head_major(q, k, v, do, dv, g)

    def qk_head(x):
        return x.reshape(batch, chunks, 64, q_heads, dim).permute(
            0, 1, 3, 2, 4
        ).repeat_interleave(value_heads // q_heads, dim=2).contiguous()

    def value_head(x):
        return x.reshape(batch, chunks, 64, value_heads, dim).permute(
            0, 1, 3, 2, 4
        ).contiguous()

    expected = (
        qk_head(q),
        qk_head(k),
        value_head(v),
        value_head(do),
        value_head(dv),
        g.reshape(batch, chunks, 64, value_heads).permute(
            0, 1, 3, 2
        ).contiguous(),
    )
    assert all(torch.equal(a, e) for a, e in zip(actual, expected))


@pytest.mark.gpu
def test_ppu_native_chunk_dv_backward_matches_official_stage():
    if not native.is_flash_qla_chunk_dv_backward_available():
        pytest.skip("PPU native chunk-dV backward is unavailable")
    torch.manual_seed(101)
    batch, tokens, q_heads, value_heads, dim = 1, 128, 2, 4, 128
    q = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    do = torch.randn(batch, tokens, value_heads, dim, device="cuda")
    raw_g = -0.03 * torch.rand(batch, tokens, value_heads, device="cuda")
    g = raw_g.view(batch, 2, 64, value_heads).cumsum(dim=2).reshape_as(raw_g)
    scale = dim**-0.5
    expected = torch_chunk_dv_bwd(q, k, g, do, scale=scale, chunk_size=64)
    g_chunks = g.view(batch, 2, 64, value_heads).permute(0, 3, 1, 2)
    actual = native.flash_qla_chunk_dv_backward_bf16_128(
        q.contiguous(), k.contiguous(), g_chunks.contiguous(),
        do.contiguous(), scale,
    )
    assert _relative_error(actual, expected) < 8e-3


@pytest.mark.gpu
def test_ppu_native_chunk_dqkw_backward_matches_official_stage():
    if not native.is_flash_qla_chunk_dqkw_backward_available():
        pytest.skip("PPU native chunk-dQKW backward is unavailable")
    torch.manual_seed(103)
    batch, tokens, heads, dim = 1, 128, 4, 128
    chunks = tokens // 64
    do = torch.randn(batch, tokens, heads, dim, device="cuda")
    vn = torch.randn_like(do)
    dv = torch.randn_like(do)
    h = torch.randn(batch, chunks, heads, dim, dim, device="cuda") / dim**0.5
    dh = torch.randn_like(h) / dim**0.5
    raw_g = -0.03 * torch.rand(batch, tokens, heads, device="cuda")
    g = raw_g.view(batch, chunks, 64, heads).cumsum(dim=2)
    scale = dim**-0.5
    do_chunks = do.view(batch, chunks, 64, heads, dim)
    vn_chunks = vn.view(batch, chunks, 64, heads, dim)
    dv_chunks = dv.view(batch, chunks, 64, heads, dim)
    expected_dq = torch.einsum("bnchv,bnhkv->bnchk", do_chunks, h)
    expected_dq *= g.exp().unsqueeze(-1) * scale
    expected_dk = torch.einsum("bnchv,bnhkv->bnchk", vn_chunks, dh)
    expected_dk *= (g[:, :, -1:, :] - g).exp().unsqueeze(-1)
    expected_dw = -torch.einsum("bnchv,bnhkv->bnchk", dv_chunks, h)
    expected = tuple(
        x.reshape(batch, tokens, heads, dim)
        for x in (expected_dq, expected_dk, expected_dw)
    )
    g_chunks = g.permute(0, 3, 1, 2).contiguous()
    actual = native.flash_qla_chunk_dqkw_backward_bf16_128(
        do.contiguous(), vn.contiguous(), dv.contiguous(),
        h.contiguous(), dh.contiguous(), g_chunks, scale,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert _relative_error(actual_tensor, expected_tensor) < 8e-3


@pytest.mark.gpu
def test_ppu_native_chunk_dqkwg_backward_matches_official_stage(monkeypatch):
    if not native.is_flash_qla_chunk_dqkwg_backward_available():
        pytest.skip("PPU native chunk-dQKWg backward is unavailable")
    torch.manual_seed(107)
    batch, tokens, q_heads, value_heads, dim = 1, 128, 2, 4, 128
    chunks = tokens // 64
    q = torch.randn(batch, tokens, q_heads, dim, device="cuda") / dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    do = torch.randn(batch, tokens, value_heads, dim, device="cuda")
    vn = torch.randn_like(do) / dim**0.5
    dv = torch.randn_like(do) / dim**0.5
    w = torch.randn_like(do) / dim**0.5
    h = torch.randn(
        batch, chunks, value_heads, dim, dim, device="cuda"
    ) / dim**0.5
    dh = torch.randn_like(h) / dim**0.5
    raw_g = -0.03 * torch.rand(batch, tokens, value_heads, device="cuda")
    g = raw_g.view(batch, chunks, 64, value_heads).cumsum(dim=2)
    g_flat = g.reshape(batch, tokens, value_heads)
    scale = dim**-0.5
    monkeypatch.setenv("FLASHQLA_PPU_HEAD_MAJOR_DQKWG", "0")
    expected = torch_chunk_dqkwg_bwd(
        q=q, k=k, v=vn, w=w, g=g_flat, h=h, dv=dv, do=do, dh=dh,
        scale=scale, chunk_size=64,
    )
    monkeypatch.setenv("FLASHQLA_PPU_HEAD_MAJOR_DQKWG", "1")
    monkeypatch.setenv("FLASHQLA_PPU_GROUPED_DQKWG", "0")
    head_major = torch_chunk_dqkwg_bwd(
        q=q, k=k, v=vn, w=w, g=g_flat, h=h, dv=dv, do=do, dh=dh,
        scale=scale, chunk_size=64,
    )
    for actual_tensor, expected_tensor in zip(head_major, expected):
        assert _relative_error(actual_tensor, expected_tensor) < 2e-5
    monkeypatch.setenv("FLASHQLA_PPU_GROUPED_DQKWG", "1")
    grouped = torch_chunk_dqkwg_bwd(
        q=q, k=k, v=vn, w=w, g=g_flat, h=h, dv=dv, do=do, dh=dh,
        scale=scale, chunk_size=64,
    )
    for actual_tensor, expected_tensor in zip(grouped, expected):
        assert _relative_error(actual_tensor, expected_tensor) < 2e-5
    actual = native.flash_qla_chunk_dqkwg_backward_bf16_128(
        q.contiguous(), k.contiguous(), do.contiguous(), vn.contiguous(),
        dv.contiguous(), h.contiguous(), dh.contiguous(),
        g.permute(0, 3, 1, 2).contiguous(), scale,
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        assert _relative_error(actual_tensor, expected_tensor) < 1.2e-2


@pytest.mark.gpu
def test_ppu_fused_ska_backward_matches_official_stages(monkeypatch):
    if not native.is_flash_qla_fused_state_dqkwg_backward_available():
        pytest.skip("PPU fused S/K/A backward kernel is unavailable")
    torch.manual_seed(108)
    batch, tokens, q_heads, value_heads, dim = 1, 128, 2, 4, 128
    chunks = tokens // 64
    q = torch.randn(batch, tokens, q_heads, dim, device="cuda") / dim**0.5
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    w = torch.randn(batch, tokens, value_heads, dim, device="cuda") / dim**0.5
    do = torch.randn_like(w) / dim**0.5
    vn = torch.randn_like(w) / dim**0.5
    h = torch.randn(
        batch, chunks, value_heads, dim, dim, device="cuda"
    ) / dim**0.5
    raw_g = -0.01 * torch.rand(batch, tokens, value_heads, device="cuda")
    g = raw_g.reshape(batch, chunks, 64, value_heads).cumsum(dim=2)
    g_flat = g.reshape(batch, tokens, value_heads)
    h0 = torch.randn(batch, value_heads, dim, dim, device="cuda") / dim**0.5
    dht = torch.randn_like(h0) / dim**0.5
    scale = dim**-0.5

    monkeypatch.setenv("FLASHQLA_PPU_FUSED_REVERSE_UPDATE", "0")
    monkeypatch.setenv("FLASHQLA_PPU_GROUPED_STATE_BWD", "0")
    dv_seed = torch_chunk_dv_bwd(
        q=q, k=k, g=g_flat, do=do, scale=scale, chunk_size=64
    )
    expected_dh, expected_dh0, expected_dv = torch_chunk_gdr_bwd(
        q, k, w, g_flat, do, dv_seed.clone(), h0, dht, scale=scale
    )
    expected_dqkwg = torch_chunk_dqkwg_bwd(
        q=q, k=k, v=vn, w=w, g=g_flat, h=h,
        dv=expected_dv, do=do, dh=expected_dh, scale=scale,
    )
    actual = native.flash_qla_fused_state_dqkwg_backward_bf16_128(
        q.contiguous(), k.contiguous(), w.contiguous(),
        g.permute(0, 3, 1, 2).contiguous(), do.contiguous(),
        dht.contiguous(), vn.contiguous(), h.contiguous(), scale,
    )
    for actual_tensor, expected_tensor in zip(actual[:4], expected_dqkwg):
        assert _relative_error(actual_tensor, expected_tensor) < 1.5e-2
    assert _relative_error(actual[4], expected_dv) < 8e-3
    assert _relative_error(actual[5], expected_dh0) < 8e-3


@pytest.mark.gpu
def test_ppu_native_chunk_wy_backward_matches_official_stage(monkeypatch):
    if not native.is_flash_qla_wy_backward_available():
        pytest.skip("PPU native chunk-WY backward is unavailable")
    torch.manual_seed(109)
    batch, tokens, q_heads, value_heads, dim = 2, 128, 2, 4, 128
    k = torch.nn.functional.normalize(
        torch.randn(batch, tokens, q_heads, dim, device="cuda"), dim=-1
    )
    v = torch.randn(batch, tokens, value_heads, dim, device="cuda") / dim**0.5
    beta = torch.sigmoid(
        torch.randn(batch, tokens, value_heads, device="cuda")
    )
    A = (
        torch.randn(batch, tokens, value_heads, 64, device="cuda")
        / 64**0.5
    )
    raw_g = -0.01 * torch.rand(
        batch, tokens, value_heads, device="cuda"
    )
    g = raw_g.reshape(batch, 2, 64, value_heads).cumsum(dim=2)
    g = g.reshape_as(raw_g)
    vectors = tuple(
        torch.randn(batch, tokens, value_heads, dim, device="cuda")
        / dim**0.5
        for _ in range(3)
    )
    dw, du, dk1 = vectors
    dg1 = torch.randn(batch, tokens, value_heads, device="cuda") / dim**0.5

    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_BWD_WY", "0")
    expected = torch_chunk_wy_bwd(
        k, v, beta, A, g, dw, du, dk1, dg1
    )
    monkeypatch.setenv("FLASHQLA_PPU_NATIVE_BWD_WY", "1")
    actual = torch_chunk_wy_bwd(
        k, v, beta, A, g, dw, du, dk1, dg1
    )
    relative_errors = [
        _relative_error(actual_tensor, expected_tensor)
        for actual_tensor, expected_tensor in zip(actual, expected)
    ]
    assert all(error < 2e-5 for error in relative_errors), relative_errors

    if native.is_flash_qla_fused_wy_backward_available():
        monkeypatch.setenv("FLASHQLA_PPU_FUSED_WY_BWD", "1")
        fused = torch_chunk_wy_bwd(
            k, v, beta, A, g, dw, du, dk1, dg1
        )
        fused_errors = [
            _relative_error(actual_tensor, expected_tensor)
            for actual_tensor, expected_tensor in zip(fused, expected)
        ]
        assert all(error < 2e-2 for error in fused_errors), fused_errors
