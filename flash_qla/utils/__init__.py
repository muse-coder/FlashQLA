# Copyright (c) 2026 The Qwen team, Alibaba Group.
# Licensed under The MIT License [see LICENSE for details]

from .pack import pad_and_reshape, pack, unpack, fill_last_chunk_of_g
from .math import l2norm, l2norm_fwd, l2norm_bwd
from .index import prepare_chunk_indices, prepare_chunk_offsets, tensor_cache
from .contiguous import input_guard


def profile(func, inputs, wait: int = 50, warmup: int = 50, rep: int = 100):
    """Load the optional TileLang profiler only when profiling is requested."""
    from .profiler import profile as _profile

    return _profile(func, inputs, wait=wait, warmup=warmup, rep=rep)


__all__ = [
    "profile",
    "pad_and_reshape",
    "pack",
    "unpack",
    "fill_last_chunk_of_g",
    "l2norm",
    "l2norm_fwd",
    "l2norm_bwd",
    "prepare_chunk_indices",
    "prepare_chunk_offsets",
    "tensor_cache",
    "input_guard",
]
