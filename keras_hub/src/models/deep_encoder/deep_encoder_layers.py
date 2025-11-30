"""Layers for DeepEncoder (SAM + modified CLIP)."""

import keras
from keras import ops

from keras_hub.src.models.vit_det.vit_det_backbone import ViTDetBackbone


@keras.saving.register_keras_serializable(package="keras_hub")
class DeepEncoderMlpProjector(keras.layers.Layer):
    """MLP projector for DeepEncoder vision features.

    Projects concatenated vision features (2048-dim) to language model
    embedding dimension (1280-dim). In the original DeepSeek-OCR, this
    is a simple linear projection.

    Args:
        input_dim: int. Input feature dimension (default 2048).
        output_dim: int. Output embedding dimension (default 1280).
    """

    def __init__(
        self,
        input_dim=2048,
        output_dim=1280,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def build(self, input_shape):
        # Simple linear projection as in original implementation
        self.projection = keras.layers.Dense(
            self.output_dim,
            use_bias=True,
            name="linear_proj",
        )
        self.built = True

    def call(self, x):
        return self.projection(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        })
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class DeepEncoderFeatureFusion(keras.layers.Layer):
    """Fuses CLIP and SAM features for DeepEncoder.

    Takes CLIP features [B, 257, 1024] and SAM features [B, 1024, 16, 16],
    removes CLS token from CLIP, flattens SAM, and concatenates them.

    Output: [B, 256, 2048]
    """

    def call(self, inputs):
        clip_features, sam_features = inputs

        # CLIP without CLS: [B, 256, 1024]
        clip_no_cls = clip_features[:, 1:, :]

        # SAM flattened: [B, 1024, 16, 16] → [B, 256, 1024]
        batch, channels, height, width = ops.shape(sam_features)
        sam_flat = ops.reshape(sam_features, (batch, channels, height * width))
        # Transpose to [B, 256, 1024]
        sam_flat = ops.transpose(sam_flat, (0, 2, 1))

        # Concatenate: [B, 256, 2048]
        return ops.concatenate([clip_no_cls, sam_flat], axis=-1)


@keras.saving.register_keras_serializable(package="keras_hub")
class SAMImageEncoder(keras.layers.Layer):
    """SAM ViT-B image encoder with additional downsampling layers.

    This is based on ViTDetBackbone but adds two extra convolutional
    downsampling layers to match the DeepSeek-OCR architecture.

    The architecture:
    1. ViTDetBackbone: Image → [B, H/16, W/16, 256]
    2. net_2: Conv 256→512, stride=2
    3. net_3: Conv 512→1024, stride=2
    4. Output: [B, 1024, 16, 16] (NCHW format)
    """

    def __init__(
        self,
        image_size=1024,
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        global_attn_indexes=[2, 5, 8, 11],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads
        self.global_attn_indexes = global_attn_indexes

    def build(self, input_shape):
        # Use ViTDetBackbone as base (outputs 256 channels)
        self.vit_det = ViTDetBackbone(
            hidden_size=self.embed_dim,
            num_layers=self.depth,
            intermediate_dim=self.embed_dim * 4,
            num_heads=self.num_heads,
            global_attention_layer_indices=self.global_attn_indexes,
            patch_size=self.patch_size,
            num_output_channels=256,
            window_size=14,
            image_shape=(self.image_size, self.image_size, 3),
        )

        # Extra downsampling layers
        # net_2: 256→512, stride=2 (64x64 → 32x32)
        self.net_2 = keras.layers.Conv2D(
            filters=512,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="net_2",
        )

        self.net_2_pad = keras.layers.ZeroPadding2D(padding=1, name="net_2_pad")

        # net_3: 512→1024, stride=2 (32x32 → 16x16)
        self.net_3 = keras.layers.Conv2D(
            filters=1024,
            kernel_size=3,
            strides=2,
            padding="valid",
            use_bias=False,
            name="net_3",
        )

        # Explicit padding to match PyTorch padding=1
        self.net_3_pad = keras.layers.ZeroPadding2D(padding=1, name="net_3_pad")

        self.built = True

    def call(self, images):
        # ViTDet output: [B, H/16, W/16, 256] = [B, 64, 64, 256]
        x = self.vit_det(images)

        x = self.net_2_pad(x)  # Pad before conv
        x = self.net_2(x)  # [B, 32, 32, 512]

        x = self.net_3_pad(x)  # Pad before conv
        x = self.net_3(x)  # [B, 16, 16, 1024]

        # Permute to NCHW for compatibility: [B, 1024, 16, 16]
        x = ops.transpose(x, (0, 3, 1, 2))

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "patch_size": self.patch_size,
            "embed_dim": self.embed_dim,
            "depth": self.depth,
            "num_heads": self.num_heads,
            "global_attn_indexes": self.global_attn_indexes,
        })
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class DeepEncoderCLIPImageEncoder(keras.layers.Layer):
    """Modified CLIP ViT-L encoder that accepts SAM features.

    This encoder takes both image and SAM features as input. When SAM features
    are provided, they replace the standard patch embeddings, allowing the CLIP
    encoder to process high-resolution features from SAM.

    Forward:
        inputs: list of [images, sam_features]
            - images: [B, H, W, 3]
            - sam_features: [B, 1024, 16, 16]
        outputs: [B, 257, 1024] (256 patches + 1 CLS token)
    """

    def __init__(
        self,
        hidden_size=1024,
        num_layers=24,
        num_heads=16,
        intermediate_dim=4096,
        image_size=224,
        patch_size=14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

    def build(self, input_shape):
        # CLS token
        self.class_embedding = self.add_weight(
            name="class_embedding",
            shape=(self.hidden_size,),
            initializer="random_normal",
            trainable=True,
        )

        # Positional embeddings (for 256 patches + 1 CLS)
        self.position_embedding = self.add_weight(
            name="position_embedding",
            shape=(self.num_patches + 1, self.hidden_size),
            initializer="random_normal",
            trainable=True,
        )

        # Pre-layer norm
        self.pre_layernorm = keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="pre_layernorm",
        )

        # Transformer blocks
        self.transformer_blocks = []
        for i in range(self.num_layers):
            block = DeepEncoderCLIPTransformerBlock(
                hidden_size=self.hidden_size,
                num_heads=self.num_heads,
                intermediate_dim=self.intermediate_dim,
                name=f"transformer_block_{i}",
            )
            self.transformer_blocks.append(block)

        self.built = True

    def call(self, inputs):
        """
        Args:
            inputs: list of [images, sam_features]
                - images: [B, H, W, 3] (not used if SAM features provided)
                - sam_features: [B, 1024, 16, 16]
        """
        images, sam_features = inputs

        # Convert SAM features to patch embeddings
        # SAM features: [B, 1024, 16, 16]
        # Convert to [B, 256, 1024] for CLIP (16x16 = 256 patches)
        batch_size = ops.shape(sam_features)[0]
        patch_embeds = ops.transpose(sam_features, (0, 2, 3, 1))  # [B, 16, 16, 1024]
        patch_embeds = ops.reshape(patch_embeds, (batch_size, 256, 1024))

        # Add CLS token
        cls_token = ops.expand_dims(self.class_embedding, axis=0)  # [1, hidden_size]
        cls_token = ops.tile(cls_token, (batch_size, 1, 1))  # [B, 1, hidden_size]
        embeddings = ops.concatenate([cls_token, patch_embeds], axis=1)  # [B, 257, hidden_size]

        # Add positional embeddings
        embeddings = embeddings + self.position_embedding

        # Pre-layer norm
        hidden_states = self.pre_layernorm(embeddings)

        # Transformer blocks
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states)

        return hidden_states

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "intermediate_dim": self.intermediate_dim,
            "image_size": self.image_size,
            "patch_size": self.patch_size,
        })
        return config


@keras.saving.register_keras_serializable(package="keras_hub")
class DeepEncoderCLIPTransformerBlock(keras.layers.Layer):
    """CLIP Transformer block with self-attention and FFN."""

    def __init__(
        self,
        hidden_size,
        num_heads,
        intermediate_dim,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.intermediate_dim = intermediate_dim

    def build(self, input_shape):
        # Layer norm 1
        self.layer_norm1 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="layer_norm1",
        )

        # Self-attention
        self.self_attention = keras.layers.MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.hidden_size // self.num_heads,
            name="self_attention",
        )

        # Layer norm 2
        self.layer_norm2 = keras.layers.LayerNormalization(
            epsilon=1e-5,
            name="layer_norm2",
        )

        # FFN
        self.dense1 = keras.layers.Dense(
            self.intermediate_dim,
            name="dense1",
        )
        self.dense2 = keras.layers.Dense(
            self.hidden_size,
            name="dense2",
        )

        self.built = True

    def call(self, x):
        # Self-attention with residual
        residual = x
        x = self.layer_norm1(x)
        x = self.self_attention(x, x)
        x = x + residual

        residual = x
        x = self.layer_norm2(x)
        x = self.dense1(x)
        # quick_gelu: x * sigmoid(1.702 * x)
        x = x * ops.sigmoid(1.702 * x)
        x = self.dense2(x)
        x = x + residual

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "intermediate_dim": self.intermediate_dim,
        })
        return config
