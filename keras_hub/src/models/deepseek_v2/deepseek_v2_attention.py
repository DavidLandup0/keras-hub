"""DeepSeekV2 Multi-head Latent Attention (MLA) implementation."""

import keras
from keras import ops

from keras_hub.src.models.deepseek_v2.deepseek_v2_layers import (
    DeepSeekV2RotaryEmbedding,
    apply_rotary_pos_emb,
)


@keras.saving.register_keras_serializable(package="keras_hub")
class DeepSeekV2Attention(keras.layers.Layer):
    """Multi-head Latent Attention (MLA) for DeepSeekV2.

    This implements the novel MLA mechanism from DeepSeekV2, which uses
    low-rank compression for efficient attention computation.

    Architecture:
        Query path:
            hidden → q_a_proj → q_a_layernorm → q_b_proj → [q_nope | q_pe]

        Key-Value path:
            hidden → kv_a_proj_with_mqa → [compressed_kv | k_pe]
                     compressed_kv → kv_a_layernorm
                     compressed_kv → kv_b_proj → [k_nope | v]

        Attention:
            attn = softmax((q_pe @ k_pe.T + q_nope @ compressed_kv.T) * scale)
            out = attn @ (k_nope + v combined from kv_b_proj)

    Args:
        hidden_size: Model hidden size (default: 1280).
        num_attention_heads: Number of attention heads (default: 10).
        q_lora_rank: Rank for query LoRA compression (default: 384).
        kv_lora_rank: Rank for key-value compression (default: 256).
        qk_rope_head_dim: Dimension for RoPE in Q/K (default: 64).
        qk_nope_head_dim: Dimension for non-RoPE part of Q/K (default: 64).
        v_head_dim: Dimension for value head (default: 128).
        max_position_embeddings: Maximum sequence length (default: 8192).
        rope_theta: Base for RoPE frequencies (default: 10000).
        attention_bias: Whether to use bias in projections (default: False).
        attention_dropout: Dropout rate for attention (default: 0.0).
    """

    def __init__(
        self,
        hidden_size=1280,
        num_attention_heads=10,
        q_lora_rank=384,
        kv_lora_rank=256,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        v_head_dim=128,
        max_position_embeddings=8192,
        rope_theta=10000,
        attention_bias=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_nope_head_dim = qk_nope_head_dim
        self.v_head_dim = v_head_dim
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        # Derived dimensions
        self.q_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.softmax_scale = self.q_head_dim ** (-0.5)

    def build(self, input_shape):
        # Query projections (with LoRA-style compression)
        if self.q_lora_rank is not None:
            self.q_a_proj = keras.layers.Dense(
                self.q_lora_rank,
                use_bias=self.attention_bias,
                name="q_a_proj",
            )
            self.q_a_layernorm = keras.layers.RMSNormalization(
                epsilon=1e-6,
                name="q_a_layernorm",
            )
            self.q_b_proj = keras.layers.Dense(
                self.num_heads * self.q_head_dim,
                use_bias=False,
                name="q_b_proj",
            )
        else:
            # Direct projection (no compression)
            self.q_proj = keras.layers.Dense(
                self.num_heads * self.q_head_dim,
                use_bias=False,
                name="q_proj",
            )

        # Key-Value projections (compressed)
        # Projects to: [kv_lora_rank + qk_rope_head_dim]
        self.kv_a_proj_with_mqa = keras.layers.Dense(
            self.kv_lora_rank + self.qk_rope_head_dim,
            use_bias=self.attention_bias,
            name="kv_a_proj_with_mqa",
        )

        self.kv_a_layernorm = keras.layers.RMSNormalization(
            epsilon=1e-6,
            name="kv_a_layernorm",
        )

        # Projects compressed KV to: [num_heads * (qk_nope_head_dim + v_head_dim)]
        self.kv_b_proj = keras.layers.Dense(
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            use_bias=False,
            name="kv_b_proj",
        )

        # Output projection
        self.o_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=self.attention_bias,
            name="o_proj",
        )

        # RoPE
        self.rotary_emb = DeepSeekV2RotaryEmbedding(
            dim=self.qk_rope_head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
            name="rotary_emb",
        )

        # Dropout
        if self.attention_dropout > 0:
            self.dropout = keras.layers.Dropout(self.attention_dropout)

        self.built = True

    def call(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        training=False,
    ):
        """
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            attention_mask: Attention mask [batch, 1, seq_len, seq_len].
            position_ids: Position indices [batch, seq_len].
            training: Whether in training mode.

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Query compression
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)  # [B, S, q_lora_rank]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)  # [B, S, num_heads * q_head_dim]
        else:
            q = self.q_proj(hidden_states)

        # Reshape to multi-head: [B, num_heads, S, q_head_dim]
        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.q_head_dim))
        q = ops.transpose(q, (0, 2, 1, 3))

        # Split query into RoPE and non-RoPE parts
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        # Key-value compression
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)

        # Split into compressed_kv and k_pe
        compressed_kv_part = compressed_kv[..., : self.kv_lora_rank]
        k_pe = compressed_kv[..., self.kv_lora_rank :]

        # Normalize compressed KV
        compressed_kv_part = self.kv_a_layernorm(compressed_kv_part)

        # Reshape k_pe for RoPE: [B, 1, S, qk_rope_head_dim]
        k_pe = ops.reshape(k_pe, (batch_size, seq_len, 1, self.qk_rope_head_dim))
        k_pe = ops.transpose(k_pe, (0, 2, 1, 3))

        # Apply RoPE
        if position_ids is None:
            position_ids = ops.expand_dims(ops.arange(seq_len), axis=0)
            position_ids = ops.tile(position_ids, (batch_size, 1))

        cos, sin = self.rotary_emb(q_pe, seq_len=seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # Apply kv_b_proj to get k_nope and values
        # compressed_kv_part: [B, S, kv_lora_rank]
        kv_b_output = self.kv_b_proj(compressed_kv_part)  # [B, S, num_heads * (qk_nope + v)]

        # Reshape to multi-head: [B, S, num_heads, qk_nope + v]
        kv_b_output = ops.reshape(
            kv_b_output,
            (batch_size, seq_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        )
        kv_b_output = ops.transpose(kv_b_output, (0, 2, 1, 3))  # [B, num_heads, S, qk_nope + v]

        # Split into k_nope and values
        k_nope = kv_b_output[..., :self.qk_nope_head_dim]  # [B, H, S, qk_nope]
        values = kv_b_output[..., self.qk_nope_head_dim:]  # [B, H, S, v_head_dim]

        # Compute attention scores
        # Attention = (q_pe @ k_pe.T) + (q_nope @ k_nope.T)
        attn_pe = ops.matmul(q_pe, ops.transpose(k_pe, (0, 1, 3, 2)))  # [B, H, S_q, S_kv]
        attn_nope = ops.matmul(q_nope, ops.transpose(k_nope, (0, 1, 3, 2)))  # [B, H, S_q, S_kv]

        attn_weights = (attn_pe + attn_nope) * self.softmax_scale

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        attn_weights = ops.softmax(attn_weights, axis=-1)

        if self.attention_dropout > 0 and training:
            attn_weights = self.dropout(attn_weights, training=training)

        # Apply attention to values
        attn_output = ops.matmul(attn_weights, values)  # [B, H, S_q, v_head_dim]

        # Reshape output
        attn_output = ops.transpose(attn_output, (0, 2, 1, 3))
        attn_output = ops.reshape(
            attn_output, (batch_size, seq_len, self.num_heads * self.v_head_dim)
        )

        # Output projection
        output = self.o_proj(attn_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_attention_heads": self.num_heads,
            "q_lora_rank": self.q_lora_rank,
            "kv_lora_rank": self.kv_lora_rank,
            "qk_rope_head_dim": self.qk_rope_head_dim,
            "qk_nope_head_dim": self.qk_nope_head_dim,
            "v_head_dim": self.v_head_dim,
            "max_position_embeddings": self.max_position_embeddings,
            "rope_theta": self.rope_theta,
            "attention_bias": self.attention_bias,
            "attention_dropout": self.attention_dropout,
        })
        return config
