"""Small jax/flax models we lower to HLO and feed into egglog.

Each `make_*` function returns `(fn, sample_args)`. To get HLO text:

    text = jax.jit(fn).lower(*sample_args).as_text("hlo")

Models are intentionally small (B=1 or 2, hidden ~16/32) so HLO stays
readable while still containing the canonical patterns we care about
(softmax, dot, transpose, reshape, broadcast, reduce, residual, FFN).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import flax.linen as nn


# ---------- minimal flax modules -------------------------------------------------

class MLPBlock(nn.Module):
    hidden: int = 32
    out: int = 16

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden)(x)
        x = nn.relu(x)
        x = nn.Dense(self.out)(x)
        return x


class AttentionBlock(nn.Module):
    """Single-head dot-product self-attention without bias."""

    d_model: int = 16

    @nn.compact
    def __call__(self, x):
        q = nn.Dense(self.d_model, use_bias=False, name="q")(x)
        k = nn.Dense(self.d_model, use_bias=False, name="k")(x)
        v = nn.Dense(self.d_model, use_bias=False, name="v")(x)
        scores = jnp.einsum("bsd,btd->bst", q, k) / jnp.sqrt(float(self.d_model))
        weights = jax.nn.softmax(scores, axis=-1)
        out = jnp.einsum("bst,btd->bsd", weights, v)
        return out


class TransformerEncoderLayer(nn.Module):
    d_model: int = 16
    ff_hidden: int = 32

    @nn.compact
    def __call__(self, x):
        attn = AttentionBlock(d_model=self.d_model)(nn.LayerNorm()(x))
        x = x + attn
        ffn_in = nn.LayerNorm()(x)
        ff1 = nn.Dense(self.ff_hidden)(ffn_in)
        ff1 = nn.relu(ff1)
        ff2 = nn.Dense(self.d_model)(ff1)
        return x + ff2


class MultiHeadAttentionBlock(nn.Module):
    d_model: int = 16
    num_heads: int = 4

    @nn.compact
    def __call__(self, x):
        return nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, qkv_features=self.d_model
        )(x)


class StackedTransformer(nn.Module):
    n_layers: int = 4
    d_model: int = 16
    ff_hidden: int = 32

    @nn.compact
    def __call__(self, x):
        for _ in range(self.n_layers):
            x = TransformerEncoderLayer(
                d_model=self.d_model, ff_hidden=self.ff_hidden
            )(x)
        return x


# ---------- registry -------------------------------------------------------------

def _key():
    return jax.random.key(0)


def make_mlp():
    model = MLPBlock(hidden=32, out=16)
    x = jnp.ones((2, 8))
    params = model.init(_key(), x)
    return jax.jit(lambda p, x: model.apply(p, x)), (params, x)


def make_attention():
    model = AttentionBlock(d_model=16)
    x = jnp.ones((1, 8, 16))
    params = model.init(_key(), x)
    return jax.jit(lambda p, x: model.apply(p, x)), (params, x)


def make_multihead():
    model = MultiHeadAttentionBlock(d_model=16, num_heads=4)
    x = jnp.ones((1, 8, 16))
    params = model.init(_key(), x)
    return jax.jit(lambda p, x: model.apply(p, x)), (params, x)


def make_transformer_layer():
    model = TransformerEncoderLayer(d_model=16, ff_hidden=32)
    x = jnp.ones((1, 8, 16))
    params = model.init(_key(), x)
    return jax.jit(lambda p, x: model.apply(p, x)), (params, x)


def make_stacked_transformer(n_layers: int = 4):
    model = StackedTransformer(n_layers=n_layers, d_model=16, ff_hidden=32)
    x = jnp.ones((1, 8, 16))
    params = model.init(_key(), x)
    return jax.jit(lambda p, x: model.apply(p, x)), (params, x)


## ---- BERT / ViT-style models -----------------------------------------------

class TransformerEncoderBlockFull(nn.Module):
    """Standard pre-LayerNorm transformer block with multi-head attention."""

    d_model: int
    n_heads: int
    ff_hidden: int
    use_gelu: bool = True

    @nn.compact
    def __call__(self, x):
        h = nn.LayerNorm()(x)
        h = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads, qkv_features=self.d_model,
        )(h, h)
        x = x + h
        h = nn.LayerNorm()(x)
        h = nn.Dense(self.ff_hidden)(h)
        h = nn.gelu(h) if self.use_gelu else nn.relu(h)
        h = nn.Dense(self.d_model)(h)
        return x + h


class MiniBert(nn.Module):
    """Minimal BERT-style encoder.

    Includes token + positional embeddings, embedding LayerNorm, then
    `n_layers` of standard transformer blocks. Output is the final hidden
    state. Vocab/seq sizes are small to keep HLO readable but the op
    structure is the same as real BERT.
    """

    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    ff_hidden: int = 3072
    vocab_size: int = 1024
    max_seq_len: int = 32

    @nn.compact
    def __call__(self, input_ids):
        seq_len = input_ids.shape[1]
        token = nn.Embed(self.vocab_size, self.d_model)(input_ids)
        pos = jnp.arange(seq_len)[None, :]
        pos = jnp.broadcast_to(pos, input_ids.shape)
        position = nn.Embed(self.max_seq_len, self.d_model)(pos)
        x = nn.LayerNorm()(token + position)
        for _ in range(self.n_layers):
            x = TransformerEncoderBlockFull(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ff_hidden=self.ff_hidden,
            )(x)
        return x


class MiniViT(nn.Module):
    """Minimal ViT-style encoder.

    A patch-embedding conv, learned class token + position embedding, then
    `n_layers` transformer blocks. Output is the class-token hidden state.
    """

    image_size: int = 32
    patch_size: int = 4
    n_layers: int = 12
    d_model: int = 768
    n_heads: int = 12
    ff_hidden: int = 3072

    @nn.compact
    def __call__(self, image):
        x = nn.Conv(
            features=self.d_model,
            kernel_size=(self.patch_size, self.patch_size),
            strides=(self.patch_size, self.patch_size),
            padding="VALID",
        )(image)
        b, h, w, c = x.shape
        x = x.reshape((b, h * w, c))
        cls = self.param(
            "cls_token", nn.initializers.zeros, (1, 1, self.d_model)
        )
        cls = jnp.broadcast_to(cls, (b, 1, self.d_model))
        x = jnp.concatenate([cls, x], axis=1)
        n_tokens = x.shape[1]
        pos = self.param(
            "pos_embed", nn.initializers.zeros, (1, n_tokens, self.d_model)
        )
        x = x + pos
        for _ in range(self.n_layers):
            x = TransformerEncoderBlockFull(
                d_model=self.d_model,
                n_heads=self.n_heads,
                ff_hidden=self.ff_hidden,
            )(x)
        x = nn.LayerNorm()(x)
        return x[:, 0]


def make_bert(n_layers: int = 12, d_model: int = 768,
              n_heads: int = 12, ff_hidden: int = 3072,
              vocab_size: int = 1024, seq_len: int = 32):
    model = MiniBert(
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ff_hidden=ff_hidden,
        vocab_size=vocab_size,
    )
    input_ids = jnp.ones((1, seq_len), dtype=jnp.int32)
    params = model.init(_key(), input_ids)
    return jax.jit(lambda p, x: model.apply(p, x)), (params, input_ids)


def make_vit(n_layers: int = 12, d_model: int = 768,
             n_heads: int = 12, ff_hidden: int = 3072,
             image_size: int = 32, patch_size: int = 4):
    model = MiniViT(
        image_size=image_size,
        patch_size=patch_size,
        n_layers=n_layers,
        d_model=d_model,
        n_heads=n_heads,
        ff_hidden=ff_hidden,
    )
    image = jnp.ones((1, image_size, image_size, 3), dtype=jnp.float32)
    params = model.init(_key(), image)
    return jax.jit(lambda p, x: model.apply(p, x)), (params, image)


REGISTRY = {
    "mlp":               lambda: make_mlp(),
    "attention":         lambda: make_attention(),
    "multihead":         lambda: make_multihead(),
    "transformer1":      lambda: make_transformer_layer(),
    "transformer4":      lambda: make_stacked_transformer(4),
    "transformer8":      lambda: make_stacked_transformer(8),
    "transformer16":     lambda: make_stacked_transformer(16),
    "transformer32":     lambda: make_stacked_transformer(32),
    # smaller BERT/ViT for sanity-checking the bridge
    "bert_tiny":         lambda: make_bert(n_layers=2, d_model=128,
                                          n_heads=2, ff_hidden=512,
                                          vocab_size=512, seq_len=16),
    "bert_small":        lambda: make_bert(n_layers=6, d_model=384,
                                          n_heads=6, ff_hidden=1536,
                                          vocab_size=1024, seq_len=32),
    # canonical sizes
    "bert_base":         lambda: make_bert(n_layers=12, d_model=768,
                                          n_heads=12, ff_hidden=3072,
                                          vocab_size=1024, seq_len=32),
    "bert_large":        lambda: make_bert(n_layers=24, d_model=1024,
                                          n_heads=16, ff_hidden=4096,
                                          vocab_size=1024, seq_len=32),
    # GPT-3 sized depth (95 layers); pushes the e-graph past 10^4 nodes
    "bert_xxl":          lambda: make_bert(n_layers=96, d_model=1024,
                                          n_heads=16, ff_hidden=4096,
                                          vocab_size=1024, seq_len=32),
    "vit_small":         lambda: make_vit(n_layers=6, d_model=384,
                                         n_heads=6, ff_hidden=1536,
                                         image_size=32, patch_size=4),
    "vit_base":          lambda: make_vit(n_layers=12, d_model=768,
                                         n_heads=12, ff_hidden=3072,
                                         image_size=32, patch_size=4),
    "vit_large":         lambda: make_vit(n_layers=24, d_model=1024,
                                         n_heads=16, ff_hidden=4096,
                                         image_size=32, patch_size=4),
}


def lower_to_hlo(fn, args) -> str:
    return jax.jit(fn).lower(*args).as_text("hlo")
