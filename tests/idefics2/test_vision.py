import jax.numpy as jnp
import torch
from einops import rearrange
from flax import nnx

from src.utils import torch2jax


## Global Defaults
rngs = nnx.Rngs(default=42)
param_dtype = jnp.float32

## Constants
HIDDEN_SIZE = 1152
NUM_HEADS = 12
SEQ_LEN = 4096


def port_idefics2_vision_attention(
    nnx_attention: nnx.Module, hf_attn: torch.nn.Module
) -> nnx.Module:
    nnx_attention.q_proj.kernel.value = torch2jax(
        rearrange(hf_attn.q_proj.weight, "i o -> o i")
    )
    nnx_attention.q_proj.bias.value = torch2jax(hf_attn.q_proj.bias)
    nnx_attention.k_proj.kernel.value = torch2jax(
        rearrange(hf_attn.k_proj.weight, "i o -> o i")
    )
    nnx_attention.k_proj.bias.value = torch2jax(hf_attn.k_proj.bias)
    nnx_attention.v_proj.kernel.value = torch2jax(
        rearrange(hf_attn.v_proj.weight, "i o -> o i")
    )
    nnx_attention.v_proj.bias.value = torch2jax(hf_attn.v_proj.bias)
    nnx_attention.out_proj.kernel.value = torch2jax(
        rearrange(hf_attn.out_proj.weight, "i o -> o i")
    )
    nnx_attention.out_proj.bias.value = torch2jax(hf_attn.out_proj.bias)
    return nnx_attention


# test_attention doesn't work refer to bin/finicky_matmul.py
