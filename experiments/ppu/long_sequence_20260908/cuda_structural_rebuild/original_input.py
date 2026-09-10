import torch
import torch.nn.functional as F

def _make_inputs(tokens=1024, hq=4, hv=16, seqlens=None, state_dtype='bfloat16', g_dtype='bfloat16', nonzero_state=True, gate_mode='default', device='cuda'):
    seqlens = seqlens or [tokens]
    assert sum(seqlens) == tokens and min(seqlens) > 0 and hv % hq == 0
    q = torch.randn(1, tokens, hq, 128, dtype=torch.bfloat16, device=device)
    k = torch.randn_like(q)
    v = torch.randn(1, tokens, hv, 128, dtype=torch.bfloat16, device=device)
    g = (-F.softplus(torch.randn(1, tokens, hv, device=device)) * 0.05).to({'bfloat16': torch.bfloat16, 'float32': torch.float32}[g_dtype])
    if gate_mode == 'zero':
        g.zero_()
    elif gate_mode == 'weak':
        g.mul_(0.01)
    elif gate_mode == 'strong':
        g.mul_(20.0)
    elif gate_mode == 'mixed':
        g[..., ::2] = 0
    elif gate_mode != 'default':
        raise ValueError(gate_mode)
    beta = torch.rand(1, tokens, hv, dtype=torch.bfloat16, device=device)
    state = torch.zeros(len(seqlens), hv, 128, 128, dtype={'bfloat16': torch.bfloat16, 'float32': torch.float32}[state_dtype], device=device)
    if nonzero_state:
        state.copy_(torch.randn(state.shape, device=device) * 0.05)
    offsets = [0]
    for length in seqlens:
        offsets.append(offsets[-1] + length)
    return dict(q=q, k=k, v=v, g=g, beta=beta, initial_state=state,
                cu_seqlens=torch.tensor(offsets, dtype=torch.int32, device=device))
