import dataclasses
from typing import List, NamedTuple, Optional, Dict

import jax
import jax.numpy as jnp

from gemma.gm.nn import _functional as F
from gemma.gm.nn import config as config_lib


class Layer(NamedTuple):
    """Weights for a single transformer layer."""
    attn: Dict[str, jax.Array]
    mlp: Dict[str, jax.Array]
    pre_attention_norm: Dict[str, jax.Array]
    post_attention_norm: Dict[str, jax.Array]
    pre_ffw_norm: Dict[str, jax.Array]
    post_ffw_norm: Dict[str, jax.Array]


class Gemma3(NamedTuple):
    """Container for the Gemma3 model weights."""
    input_embedding_table: jax.Array
    mm_input_projection: jax.Array | None = None
    mm_soft_embedding_norm: jax.Array | None = None
    final_norm_scale: jax.Array | None = None
    blocks: List[Layer] = []


@dataclasses.dataclass
class ForwardOutput:
    logits: jax.Array
    cache: Optional[config_lib.Cache]
    hidden_states: Optional[jax.Array]


def _rms_norm(x: jax.Array, scale: jax.Array) -> jax.Array:
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(var + 1e-6)
    scale = jnp.expand_dims(scale, axis=tuple(range(x.ndim - 1)))
    return x * (1 + scale)


def transformer_forward(
    model: Gemma3,
    tokens: jax.Array,
    *,
    config: config_lib.TransformerConfig,
    cache: Optional[config_lib.Cache] = None,
    positions: Optional[jax.Array] = None,
    attention_mask: Optional[jax.Array] = None,
    return_last_only: bool = False,
    return_hidden_states: bool = False,
) -> Output:
    """Run a forward pass of the Gemma model."""
    x = F.embed_tokens({'input_embedding': model.input_embedding_table}, tokens, config)
    batch_size, seq_len = x.shape[:2]

    if positions is None:
        positions = jnp.arange(seq_len)[None, :]
        positions = jnp.broadcast_to(positions, (batch_size, seq_len))

    cache_len = seq_len if cache is None else next(iter(cache.values()))['k'].shape[1]
    if attention_mask is None:
        i = jnp.arange(seq_len)[None, :, None]
        j = jnp.arange(cache_len)[None, None, :]
        attention_mask = i >= (j - (cache_len - seq_len))
        attention_mask = jnp.broadcast_to(attention_mask, (batch_size, seq_len, cache_len))

    old_cache = cache or {}
    new_cache = {}
    for i, layer in enumerate(model.blocks):
        layer_cache, x = F.block(
            {
                'attn': layer.attn,
                'mlp': layer.mlp,
                'pre_attention_norm': layer.pre_attention_norm,
                'post_attention_norm': layer.post_attention_norm,
                'pre_ffw_norm': layer.pre_ffw_norm,
                'post_ffw_norm': layer.post_ffw_norm,
            },
            x,
            positions,
            old_cache.get(f'layer_{i}'),
            attention_mask,
            attn_type=config.attention_types[i].name,
            config=config,
        )
        new_cache[f'layer_{i}'] = layer_cache

    if model.final_norm_scale is not None:
        x = _rms_norm(x, model.final_norm_scale)

    logits = jnp.einsum('...d,vd->...v', x, model.input_embedding_table)

    if config.final_logit_softcap is not None:
        logits = jnp.tanh(logits / config.final_logit_softcap) * config.final_logit_softcap

    if return_last_only:
        logits = logits[:, -1]
        x = x[:, -1]

    return ForwardOutput(
        logits=logits,
        cache=None if cache is None else new_cache,
        hidden_states=x if return_hidden_states else None,
    )

