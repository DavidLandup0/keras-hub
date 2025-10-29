"""Layers for DeepSeekV2 decoder."""

import keras
from keras import ops


class DeepSeekV2RotaryEmbedding(keras.layers.Layer):
    """Rotary Position Embedding (RoPE) for DeepSeekV2.

    Based on the RoFormer paper: https://arxiv.org/abs/2104.09864

    Args:
        dim: Dimensionality of the rotary embeddings.
        max_position_embeddings: Maximum sequence length.
        base: Base for the inverse frequency computation.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

    def build(self, input_shape):
        # Compute inverse frequencies: 1.0 / (base^(i/dim)) for i in [0, 2, 4, ..., dim-2]
        # Shape: [dim/2]
        inv_freq = 1.0 / (
            self.base ** (ops.arange(0, self.dim, 2, dtype="float32") / self.dim)
        )
        self.inv_freq = self.add_weight(
            name="inv_freq",
            shape=inv_freq.shape,
            initializer=keras.initializers.Constant(inv_freq),
            trainable=False,
        )

        # Pre-compute cos and sin cache for max sequence length
        # t shape: [max_position_embeddings]
        t = ops.arange(self.max_position_embeddings, dtype="float32")
        # freqs shape: [max_position_embeddings, dim/2]
        freqs = ops.outer(t, self.inv_freq)
        # Concatenate to get [max_position_embeddings, dim]
        emb = ops.concatenate([freqs, freqs], axis=-1)

        self.cos_cached = self.add_weight(
            name="cos_cached",
            shape=emb.shape,
            initializer=keras.initializers.Constant(ops.cos(emb)),
            trainable=False,
        )
        self.sin_cached = self.add_weight(
            name="sin_cached",
            shape=emb.shape,
            initializer=keras.initializers.Constant(ops.sin(emb)),
            trainable=False,
        )

        self.built = True

    def call(self, x, seq_len=None):
        """
        Args:
            x: Input tensor (used to determine dtype).
            seq_len: Sequence length to return embeddings for.

        Returns:
            Tuple of (cos, sin) tensors of shape [seq_len, dim].
        """
        if seq_len is None:
            seq_len = self.max_position_embeddings

        # Return cached values up to seq_len
        cos = ops.cast(self.cos_cached[:seq_len], x.dtype)
        sin = ops.cast(self.sin_cached[:seq_len], x.dtype)

        return cos, sin

    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "max_position_embeddings": self.max_position_embeddings,
            "base": self.base,
        })
        return config


def rotate_half(x):
    """Rotates half the hidden dims of the input for RoPE.

    Args:
        x: Input tensor of shape [..., dim].

    Returns:
        Tensor with second half negated and concatenated with first half.
    """
    x1 = x[..., : ops.shape(x)[-1] // 2]
    x2 = x[..., ops.shape(x)[-1] // 2 :]
    return ops.concatenate([-x2, x1], axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    """Applies Rotary Position Embedding to query and key tensors.

    Args:
        q: Query tensor [batch, num_heads, seq_len, head_dim].
        k: Key tensor [batch, num_kv_heads, seq_len, head_dim].
        cos: Cosine embeddings [seq_len, head_dim].
        sin: Sine embeddings [seq_len, head_dim].
        position_ids: Position indices [batch, seq_len].

    Returns:
        Tuple of (q_embed, k_embed) with rotary embeddings applied.
    """
    # Gather cos/sin for the positions: [batch, seq_len, head_dim]
    cos = ops.take(cos, position_ids, axis=0)
    sin = ops.take(sin, position_ids, axis=0)

    # Add dimension for num_heads: [batch, 1, seq_len, head_dim]
    cos = ops.expand_dims(cos, axis=1)
    sin = ops.expand_dims(sin, axis=1)

    # DeepSeekV2 uses a special permutation for RoPE
    # Reshape to interleave dimensions: [b, h, s, d//2, 2] -> transpose -> reshape
    batch, num_heads, seq_len, head_dim = ops.shape(q)
    q_reshaped = ops.reshape(q, (batch, num_heads, seq_len, head_dim // 2, 2))
    q_reshaped = ops.transpose(q_reshaped, (0, 1, 2, 4, 3))  # Swap last two dims
    q_permuted = ops.reshape(q_reshaped, (batch, num_heads, seq_len, head_dim))

    batch_k, num_kv_heads, seq_len_k, head_dim_k = ops.shape(k)
    k_reshaped = ops.reshape(k, (batch_k, num_kv_heads, seq_len_k, head_dim_k // 2, 2))
    k_reshaped = ops.transpose(k_reshaped, (0, 1, 2, 4, 3))
    k_permuted = ops.reshape(k_reshaped, (batch_k, num_kv_heads, seq_len_k, head_dim_k))

    # Apply rotary embedding
    q_embed = (q_permuted * cos) + (rotate_half(q_permuted) * sin)
    k_embed = (k_permuted * cos) + (rotate_half(k_permuted) * sin)

    return q_embed, k_embed


class DeepSeekV2MLP(keras.layers.Layer):
    """MLP layer for DeepSeekV2 with gated activation (SwiGLU).

    Architecture:
        hidden -> gate_proj -> act_fn ──┐
                                         × -> down_proj -> hidden
        hidden -> up_proj ───────────────┘

    Args:
        hidden_size: Input/output hidden size.
        intermediate_size: Hidden size of the MLP.
        hidden_act: Activation function name (default: "silu").
    """

    def __init__(
        self,
        hidden_size,
        intermediate_size,
        hidden_act="silu",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act

    def build(self, input_shape):
        self.gate_proj = keras.layers.Dense(
            self.intermediate_size,
            use_bias=False,
            name="gate_proj",
        )
        self.up_proj = keras.layers.Dense(
            self.intermediate_size,
            use_bias=False,
            name="up_proj",
        )
        self.down_proj = keras.layers.Dense(
            self.hidden_size,
            use_bias=False,
            name="down_proj",
        )
        self.act_fn = keras.activations.get(self.hidden_act)
        self.built = True

    def call(self, x):
        """Forward pass: down_proj(act_fn(gate_proj(x)) * up_proj(x))."""
        gate = self.act_fn(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "hidden_act": self.hidden_act,
        })
        return config


class MoEGate(keras.layers.Layer):
    """MoE gating mechanism for expert routing.

    Computes routing weights for selecting top-k experts per token.

    Args:
        num_experts: Number of routed experts.
        num_experts_per_tok: Number of experts to route each token to (top-k).
        hidden_size: Hidden dimension size.
        scoring_func: Scoring function ("softmax" or "sigmoid").
        norm_topk_prob: Whether to normalize top-k probabilities to sum to 1.
        routed_scaling_factor: Scaling factor for routed expert weights.
    """

    def __init__(
        self,
        num_experts,
        num_experts_per_tok,
        hidden_size,
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.hidden_size = hidden_size
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

    def build(self, input_shape):
        # Gating weight: [num_experts, hidden_size]
        self.weight = self.add_weight(
            name="weight",
            shape=(self.num_experts, self.hidden_size),
            initializer=keras.initializers.VarianceScaling(
                scale=2.0, mode="fan_in", distribution="uniform"
            ),
            trainable=True,
        )
        self.built = True

    def call(self, hidden_states, training=False):
        """
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            training: Whether in training mode.

        Returns:
            topk_idx: Expert indices [batch*seq_len, top_k].
            topk_weight: Expert weights [batch*seq_len, top_k].
        """
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Flatten: [batch*seq_len, hidden_size]
        hidden_flat = ops.reshape(hidden_states, (-1, self.hidden_size))

        # Compute logits: [batch*seq_len, num_experts]
        hidden_flat_fp32 = ops.cast(hidden_flat, "float32")
        weight_fp32 = ops.cast(self.weight, "float32")
        logits = ops.matmul(hidden_flat_fp32, ops.transpose(weight_fp32))

        # Apply scoring function
        if self.scoring_func == "softmax":
            scores = ops.softmax(logits, axis=-1)
        elif self.scoring_func == "sigmoid":
            scores = ops.sigmoid(logits)
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_func}")

        # Select top-k experts (greedy method)
        topk_weight, topk_idx = ops.top_k(scores, k=self.num_experts_per_tok)

        # Normalize top-k probabilities
        if self.num_experts_per_tok > 1 and self.norm_topk_prob:
            denominator = ops.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
            topk_weight = (topk_weight / denominator) * self.routed_scaling_factor
        else:
            topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "hidden_size": self.hidden_size,
            "scoring_func": self.scoring_func,
            "norm_topk_prob": self.norm_topk_prob,
            "routed_scaling_factor": self.routed_scaling_factor,
        })
        return config


class DeepSeekV2MoE(keras.layers.Layer):
    """Mixture of Experts layer for DeepSeekV2.

    Combines routed experts and shared experts.

    Args:
        hidden_size: Hidden dimension size.
        num_experts: Number of routed experts.
        num_experts_per_tok: Number of experts to route to per token.
        expert_intermediate_size: Intermediate size for routed experts.
        shared_expert_intermediate_size: Intermediate size for shared experts.
        scoring_func: Scoring function for gating.
        norm_topk_prob: Whether to normalize top-k probabilities.
        routed_scaling_factor: Scaling factor for routed weights.
    """

    def __init__(
        self,
        hidden_size,
        num_experts,
        num_experts_per_tok,
        expert_intermediate_size,
        shared_expert_intermediate_size,
        scoring_func="softmax",
        norm_topk_prob=True,
        routed_scaling_factor=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.expert_intermediate_size = expert_intermediate_size
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.scoring_func = scoring_func
        self.norm_topk_prob = norm_topk_prob
        self.routed_scaling_factor = routed_scaling_factor

    def build(self, input_shape):
        # Create routed experts
        self.experts = []
        for i in range(self.num_experts):
            expert = DeepSeekV2MLP(
                hidden_size=self.hidden_size,
                intermediate_size=self.expert_intermediate_size,
                name=f"expert_{i}",
            )
            self.experts.append(expert)

        # Gating network
        self.gate = MoEGate(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            scoring_func=self.scoring_func,
            norm_topk_prob=self.norm_topk_prob,
            routed_scaling_factor=self.routed_scaling_factor,
            name="gate",
        )

        # Shared experts
        self.shared_experts = DeepSeekV2MLP(
            hidden_size=self.hidden_size,
            intermediate_size=self.shared_expert_intermediate_size,
            name="shared_experts",
        )

        self.built = True

    def call(self, hidden_states, training=False):
        """
        Args:
            hidden_states: Input tensor [batch, seq_len, hidden_size].
            training: Whether in training mode.

        Returns:
            Output tensor [batch, seq_len, hidden_size].
        """
        identity = hidden_states
        batch_size = ops.shape(hidden_states)[0]
        seq_len = ops.shape(hidden_states)[1]

        # Get routing decisions
        topk_idx, topk_weight = self.gate(hidden_states, training=training)

        # Flatten input: [batch*seq_len, hidden_size]
        hidden_flat = ops.reshape(hidden_states, (-1, self.hidden_size))

        # Process through routed experts
        # For each token, apply its top-k experts
        final_output = ops.zeros_like(hidden_flat)

        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask = ops.cast(ops.equal(topk_idx, expert_idx), "float32")
            expert_weight = ops.sum(expert_mask * topk_weight, axis=-1, keepdims=True)

            # Apply expert to all tokens (will be masked by weight)
            expert_output = self.experts[expert_idx](hidden_flat)
            final_output = final_output + expert_output * expert_weight

        # Reshape back: [batch, seq_len, hidden_size]
        final_output = ops.reshape(final_output, (batch_size, seq_len, self.hidden_size))

        # Add shared experts
        shared_output = self.shared_experts(identity)
        final_output = final_output + shared_output

        return final_output

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_experts": self.num_experts,
            "num_experts_per_tok": self.num_experts_per_tok,
            "expert_intermediate_size": self.expert_intermediate_size,
            "shared_expert_intermediate_size": self.shared_expert_intermediate_size,
            "scoring_func": self.scoring_func,
            "norm_topk_prob": self.norm_topk_prob,
            "routed_scaling_factor": self.routed_scaling_factor,
        })
        return config
