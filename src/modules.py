import jax.numpy as jnp
from chex import Array
from flax import nnx
from jax.typing import DTypeLike


class MLP(nnx.Module):
    """
    References:
    * https://github.com/huggingface/transformers/blob/bc6ae0d55e11e46eaed4da71b6bc5087d38cec70/src/transformers/models/idefics2/modeling_idefics2.py#L388
    * https://github.com/Blaizzy/mlx-vlm/blob/3f5e1620072440afb7496940f67ac1c7fc64056f/mlx_vlm/models/idefics2/vision.py#L104
    """

    def __init__(
        self,
        in_features: int,
        intermediate_features: int,
        out_features: int,
        rngs: nnx.Rngs,
        param_dtype: DTypeLike = jnp.bfloat16,
    ) -> None:
        self.gate_proj = nnx.Linear(
            in_features=in_features,
            out_features=intermediate_features,
            use_bias=False,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.up_proj = nnx.Linear(
            in_features=in_features,
            out_features=intermediate_features,
            use_bias=False,
            rngs=rngs,
            param_dtype=param_dtype,
        )
        self.down_proj = nnx.Linear(
            in_features=intermediate_features,
            out_features=out_features,
            use_bias=False,
            rngs=rngs,
            param_dtype=param_dtype,
        )

    def __call__(self, input: Array) -> Array:
        return self.down_proj(nnx.silu(self.gate_proj(input)) * self.up_proj(input))
