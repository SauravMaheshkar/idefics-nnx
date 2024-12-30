import chex
import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from flax import nnx
from flax.nnx import RMSNorm as NNXRMSNorm
from torch import nn

from src.modules import MLP
from src.utils import torch2jax


## Global Defaults
rngs = nnx.Rngs(default=42)
param_dtype = jnp.float32

## Constants
HIDDEN_SIZE = 4096
INPUT_DIM = 1152
INTERMEDIATE_DIM = 14336
HEAD_DIM = 96
NUM_HEADS = 16
NUM_KEY_VALUE_HEADS = 4


def port_mlp(nnx_mlp: nnx.Module, hf_mlp: nn.Module) -> nnx.Module:
    nnx_mlp.gate_proj.kernel.value = torch2jax(
        rearrange(hf_mlp.gate_proj.weight, "i o -> o i")
    )
    nnx_mlp.up_proj.kernel.value = torch2jax(
        rearrange(hf_mlp.up_proj.weight, "i o -> o i")
    )
    nnx_mlp.down_proj.kernel.value = torch2jax(
        rearrange(hf_mlp.down_proj.weight, "i o -> o i")
    )
    return nnx_mlp


def port_rmsnorm(nnx_rmsnorm: nnx.Module, hf_rmsnorm: nn.Module) -> nnx.Module:
    nnx_rmsnorm.scale.value = torch2jax(hf_rmsnorm.weight)
    return nnx_rmsnorm


def test_mlp(get_idefics2_from_hub):
    ## Declare Modules
    nnx_mlp = MLP(
        in_features=INPUT_DIM,
        intermediate_features=INTERMEDIATE_DIM,
        out_features=HIDDEN_SIZE,
        rngs=rngs,
        param_dtype=param_dtype,
    )
    hf_mlp = get_idefics2_from_hub.model.connector.modality_projection
    nnx_mlp = port_mlp(nnx_mlp, hf_mlp)

    ## Generate Inputs
    nnx_input = jax.random.normal(
        key=jax.random.PRNGKey(42), shape=(1, INPUT_DIM), dtype=jnp.float32
    )
    hf_input = torch.from_numpy(nnx_input.__array__()).to(torch.float32)
    chex.assert_trees_all_close(
        nnx_input,
        torch2jax(hf_input),
        rtol=1e-5,
        atol=1e-5,
    )

    ## Compare Outputs
    nnx_output = nnx_mlp(nnx_input)
    hf_output = hf_mlp(hf_input)
    chex.assert_trees_all_close(
        nnx_output,
        torch2jax(hf_output),
        rtol=1e-5,
        atol=1e-5,
    )


def test_rms_norm(get_idefics2_from_hub):
    ## Declare Modules
    nnx_rmsnorm = NNXRMSNorm(num_features=HIDDEN_SIZE, rngs=rngs, param_dtype=param_dtype)
    hf_rmsnorm = get_idefics2_from_hub.model.connector.perceiver_resampler.norm
    nnx_rmsnorm = port_rmsnorm(nnx_rmsnorm, hf_rmsnorm)

    ## Generate Inputs
    jax_input = jax.random.normal(
        key=jax.random.PRNGKey(42), shape=(1, HIDDEN_SIZE), dtype=jnp.float32
    )
    torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)
    chex.assert_trees_all_close(
        jax_input,
        torch2jax(torch_input),
        rtol=1e-5,
        atol=1e-5,
    )

    ## Compare Outputs
    jax_output = nnx_rmsnorm(jax_input)
    torch_output = hf_rmsnorm(torch_input)
    chex.assert_trees_all_close(
        jax_output,
        torch2jax(torch_output),
        rtol=1e-5,
        atol=1e-5,
    )
