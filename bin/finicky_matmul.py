"""
This script aims to reproduce a error I encountered while porting the attention module
of the vision encoder of Idefics2. Usually simply setting the weights (and biases) from
a pre-trained pytorch model to the corresponding nnx layers does the trick even though
matrix multiplication is finicky.

References:
* https://github.com/pytorch/pytorch/issues/17678
"""

import chex
import jax
import jax.numpy as jnp
import torch
from einops import rearrange
from flax import nnx
from transformers import AutoModelForImageTextToText

from src.utils import torch2jax


base_model = AutoModelForImageTextToText.from_pretrained("HuggingFaceM4/idefics2-8b")
hf_attn = base_model.model.vision_model.encoder.layers[0].self_attn

## Global Defaults
rngs = nnx.Rngs(default=42)
param_dtype = jnp.float32

## Constants
HIDDEN_SIZE = 1152
NUM_HEADS = 12
SEQ_LEN = 4096
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS
SCALE = HEAD_DIM**-0.5

## Assert similar input tensors
jax_input = jax.random.normal(
    key=jax.random.PRNGKey(42),
    shape=(1, SEQ_LEN, HIDDEN_SIZE),
    dtype=jnp.float32,
)
torch_input = torch.from_numpy(jax_input.__array__()).to(torch.float32)
chex.assert_trees_all_close(
    jax_input,
    torch2jax(torch_input),
    rtol=1e-5,
    atol=1e-5,
)

########################################
##### Check Query Projection Layer #####
########################################
jax_q_proj = nnx.Linear(
    in_features=HIDDEN_SIZE,
    out_features=HIDDEN_SIZE,
    rngs=rngs,
    param_dtype=param_dtype,
)
torch_q_proj = hf_attn.q_proj

# Assert same weight matrices
jax_q_proj.kernel.value = torch2jax(rearrange(hf_attn.q_proj.weight, "i o -> o i"))
jax_q_proj.bias.value = torch2jax(hf_attn.q_proj.bias)

# Forward pass
jax_query_states = jax_q_proj(jax_input)
torch_query_states = torch_q_proj(torch_input)

# Compare outputs
chex.assert_trees_all_close(
    jax_query_states,
    torch2jax(torch_query_states),
    rtol=1e-5,
    atol=1e-5,
)

torch_query_states = torch_query_states.view(1, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(
    1, 2
)
jax_query_states = jnp.transpose(
    jax_query_states.reshape((1, SEQ_LEN, NUM_HEADS, HEAD_DIM)), (0, 2, 1, 3)
)

chex.assert_trees_all_close(
    jax_query_states,
    torch2jax(torch_query_states),
    rtol=1e-5,
    atol=1e-5,
)

######################################
##### Check Key Projection Layer #####
######################################
jax_k_proj = nnx.Linear(
    in_features=HIDDEN_SIZE,
    out_features=HIDDEN_SIZE,
    rngs=rngs,
    param_dtype=param_dtype,
)
torch_k_proj = hf_attn.k_proj

# Assert same weight matrices
jax_k_proj.kernel.value = torch2jax(rearrange(hf_attn.k_proj.weight, "i o -> o i"))
jax_k_proj.bias.value = torch2jax(hf_attn.k_proj.bias)

# Forward Pass
jax_key_states = jax_k_proj(jax_input)
torch_key_states = torch_k_proj(torch_input)

jax_key_states = jnp.transpose(
    jax_key_states.reshape((1, SEQ_LEN, NUM_HEADS, HEAD_DIM)), (0, 2, 1, 3)
)
torch_key_states = torch_key_states.view(1, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(1, 2)

# Compare outputs
chex.assert_trees_all_close(
    jax_key_states,
    torch2jax(torch_key_states),
    rtol=1e-5,
    atol=1e-5,
)

########################################
##### Check Value Projection Layer #####
########################################
jax_v_proj = nnx.Linear(
    in_features=HIDDEN_SIZE,
    out_features=HIDDEN_SIZE,
    rngs=rngs,
    param_dtype=param_dtype,
)
torch_v_proj = hf_attn.v_proj

# Assert same weight matrices
jax_v_proj.kernel.value = torch2jax(rearrange(hf_attn.v_proj.weight, "i o -> o i"))
jax_v_proj.bias.value = torch2jax(hf_attn.v_proj.bias)

# Forward Pass
jax_value_states = jax_v_proj(jax_input)
torch_value_states = torch_v_proj(torch_input)

jax_value_states = jnp.transpose(
    jax_value_states.reshape((1, SEQ_LEN, NUM_HEADS, HEAD_DIM)), (0, 2, 1, 3)
)
torch_value_states = torch_value_states.view(1, SEQ_LEN, NUM_HEADS, HEAD_DIM).transpose(
    1, 2
)

# Compare outputs
chex.assert_trees_all_close(
    jax_value_states,
    torch2jax(torch_value_states),
    rtol=1e-5,
    atol=1e-5,
)

#################################
##### Check Attention Layer #####
#################################
torch_attn_weights = (
    torch.matmul(torch_query_states, torch_key_states.transpose(2, 3)) * SCALE
)
jax_attn_weights = (
    jnp.matmul(
        jax_query_states,
        jnp.transpose(jax_key_states, (0, 1, 3, 2)),
        preferred_element_type=param_dtype,
    )
    * SCALE
)

######################
## !! This fails !! ##
######################
chex.assert_trees_all_close(
    jax_attn_weights,
    torch2jax(torch_attn_weights),
    rtol=1e-5,
    atol=1e-5,
)
