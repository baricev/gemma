import jax
import jax.numpy as jnp
from gemma.gm.math import apply_rope


K_MASK = -2.3819763e38  # Same constant as the module version.


def init_cache(batch_size: int, cache_length: int, config, *, dtype=jnp.bfloat16):
    """Initialize attention caches for all layers."""
    if cache_length is None:
        raise ValueError('cache_length must be specified')
    caches = {}
    for i in range(len(config.attention_types)):
        caches[f'layer_{i}'] = {
            'v': jnp.zeros((batch_size, cache_length, config.num_kv_heads, config.head_dim), dtype=dtype),
            'k': jnp.zeros((batch_size, cache_length, config.num_kv_heads, config.head_dim), dtype=dtype),
            'end_index': jnp.zeros((batch_size,), dtype=jnp.int32),
        }
    return caches


def _rms_norm(x, scale):
    var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
    x = x * jax.lax.rsqrt(var + 1e-6)
    scale = jnp.expand_dims(scale, axis=tuple(range(x.ndim - 1)))
    return x * (1 + scale)


def embed_tokens(params_embedder, tokens, config):
    table = params_embedder['input_embedding']
    out = table[tokens]
    out = out * jnp.sqrt(config.embed_dim).astype(out.dtype)
    return out


def attention(params_attn, x, segment_pos, cache, mask, *, attn_type, config):
    if 'qkv_einsum' in params_attn:
        qkv = jnp.einsum('BTD,SNDH->SBTNH', x, params_attn['qkv_einsum'])
        query_proj, key_proj, value_proj = qkv
    else:
        query_proj = jnp.einsum('BTD,NDH->BTNH', x, params_attn['q_einsum'])
        kv = jnp.einsum('BSD,CKDH->CBSKH', x, params_attn['kv_einsum'])
        key_proj, value_proj = kv

    if config.use_qk_norm:
        query_proj = _rms_norm(query_proj, params_attn['_query_norm']['scale'])
        key_proj = _rms_norm(key_proj, params_attn['_key_norm']['scale'])

    query_proj = apply_rope(
        query_proj,
        segment_pos,
        base_frequency=config.local_base_frequency,
        scale_factor=config.local_scale_factor,
    )
    query_scaled = query_proj * config.query_pre_attn_scalar()

    key_proj = apply_rope(
        key_proj,
        segment_pos,
        base_frequency=config.local_base_frequency,
        scale_factor=config.local_scale_factor,
    )

    if cache is not None:
        end_index = cache['end_index'][0]
        cache_size = cache['v'].shape[1]
        slice_indices = (0, end_index % cache_size, 0, 0)
        value_proj = jax.lax.dynamic_update_slice(cache['v'], value_proj, slice_indices)
        key_proj = jax.lax.dynamic_update_slice(cache['k'], key_proj, slice_indices)

    if config.num_kv_heads != config.num_heads and config.num_kv_heads > 1:
        b, t, kg, h = query_scaled.shape
        query_scaled = query_scaled.reshape((b, t, config.num_kv_heads, kg // config.num_kv_heads, h))
        logits = jnp.einsum('BTKGH,BSKH->BTKGS', query_scaled, key_proj)
        b, t, k, g, s = logits.shape
        logits = logits.reshape((b, t, k * g, s))
    else:
        logits = jnp.einsum('BTNH,BSNH->BTNS', query_scaled, key_proj)

    if config.attn_logits_soft_cap is not None:
        logits = jnp.tanh(logits / config.attn_logits_soft_cap)
        logits = logits * config.attn_logits_soft_cap

    if attn_type == 'LOCAL_SLIDING':
        if config.sliding_window_size is None:
            raise ValueError('sliding_window_size must be set for local sliding attention')
        sliding_mask = _create_sliding_mask(
            segment_pos,
            end_index=cache['end_index'][0] if cache is not None else 0,
            cache_len=mask.shape[-1],
            sliding_window_size=config.sliding_window_size,
        )
        mask = mask * sliding_mask

    padded_logits = jnp.where(jnp.expand_dims(mask, -2), logits, K_MASK)
    probs = jax.nn.softmax(padded_logits, axis=-1).astype(key_proj.dtype)

    if config.num_kv_heads != config.num_heads and config.num_kv_heads > 1:
        b, t, kg, h = probs.shape
        probs = probs.reshape((b, t, config.num_kv_heads, kg // config.num_kv_heads, h))
        encoded = jnp.einsum('BTKGS,BSKH->BTKGH', probs, value_proj)
        b, t, k, g, h = encoded.shape
        encoded = encoded.reshape((b, t, k * g, h))
    else:
        encoded = jnp.einsum('BTNS,BSNH->BTNH', probs, value_proj)

    out = jnp.einsum('BTNH,NHD->BTD', encoded, params_attn['attn_vec_einsum'])

    if cache is not None:
        seq_len = x.shape[1]
        new_cache = {
            'v': value_proj,
            'k': key_proj,
            'end_index': cache['end_index'] + seq_len,
        }
    else:
        new_cache = None

    return new_cache, out


def _create_sliding_mask(segment_pos, end_index, cache_len, sliding_window_size):
    total_tokens = end_index + segment_pos.shape[1]

    def _reconstruct_rotated_cache_positions():
        cache_positions = jnp.arange(cache_len) + total_tokens - cache_len
        cache_positions = jnp.zeros_like(cache_positions).at[cache_positions % cache_len].set(cache_positions)
        return cache_positions

    cache_positions = jax.lax.cond(
        total_tokens <= cache_len,
        lambda: jnp.arange(cache_len),
        _reconstruct_rotated_cache_positions,
    )

    cache_positions = cache_positions[None, None, :]
    segment_pos = segment_pos[:, :, None]
    sliding_mask = cache_positions > segment_pos - sliding_window_size
    sliding_mask = sliding_mask * (cache_positions < segment_pos + sliding_window_size)
    return sliding_mask


def feed_forward(params_ffw, x, config):
    if config.transpose_gating_einsum:
        gate = jnp.einsum('...F,NHF->...NH', x, params_ffw['gating_einsum'])
    else:
        gate = jnp.einsum('...F,NFH->...NH', x, params_ffw['gating_einsum'])
    activations = jax.nn.gelu(gate[..., 0, :]) * gate[..., 1, :]
    out = jnp.einsum('...H,HF->...F', activations, params_ffw['linear'])
    return out


def block(params_block, x, segment_pos, cache, mask, *, attn_type, config):
    x_norm = _rms_norm(x, params_block['pre_attention_norm']['scale'])
    layer_cache, attn_out = attention(
        params_block['attn'],
        x_norm,
        segment_pos,
        cache,
        mask,
        attn_type=attn_type,
        config=config,
    )

    if config.use_post_attn_norm:
        attn_out = _rms_norm(attn_out, params_block['post_attention_norm']['scale'])

    attn_out = attn_out + x

    out = _rms_norm(attn_out, params_block['pre_ffw_norm']['scale'])
    out = feed_forward(params_block['mlp'], out, config)

    if config.use_post_ffw_norm:
        out = _rms_norm(out, params_block['post_ffw_norm']['scale'])

    out = out + attn_out
    return layer_cache, out
