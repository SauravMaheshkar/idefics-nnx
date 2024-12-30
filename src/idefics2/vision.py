import jax.numpy as jnp
from chex import Array
from flax import nnx
from jax.typing import DTypeLike


class Idefics2VisionAttention(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        if self.head_dim * num_heads != hidden_size:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.scale = self.head_dim**-0.5

        self.k_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.v_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.q_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.out_proj = nnx.Linear(
            in_features=hidden_size,
            out_features=hidden_size,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, hidden_states: Array) -> Array:
        B, L, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = jnp.transpose(
            query_states.reshape((B, L, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        key_states = jnp.transpose(
            key_states.reshape((B, L, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )
        value_states = jnp.transpose(
            value_states.reshape((B, L, self.num_heads, self.head_dim)), (0, 2, 1, 3)
        )

        # TODO: Consider flax.nnx.dot_product_attention
        attn_weights = (
            jnp.matmul(query_states, jnp.transpose(key_states, (0, 1, 3, 2))) * self.scale
        )
        attn_weights = nnx.softmax(attn_weights, axis=-1)
        attn_output = jnp.matmul(attn_weights, value_states)

        attn_output = jnp.transpose(attn_output, (0, 2, 1, 3)).reshape(
            (B, L, self.hidden_size)
        )
        return self.out_proj(attn_output)
