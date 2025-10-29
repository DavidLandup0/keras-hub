import keras
from keras import ops

from keras_hub.src.api_export import keras_hub_export
from keras_hub.src.models.backbone import Backbone
from keras_hub.src.models.deep_encoder.deep_encoder_layers import (
    DeepEncoderCLIPImageEncoder,
    DeepEncoderFeatureFusion,
    SAMImageEncoder,
)


@keras_hub_export("keras_hub.models.DeepEncoderBackbone")
class DeepEncoderBackbone(Backbone):
    """DeepEncoder vision backbone combining SAM ViT-B and modified CLIP ViT-L.

    DeepEncoder is a vision encoder that combines two vision transformers:
    1. SAM ViT-B: Processes images and outputs spatial features [B, 1024, 16, 16]
    2. Modified CLIP ViT-L: Takes image and SAM features, outputs [B, 257, 1024]

    The final output concatenates CLIP tokens (without CLS) and SAM features,
    resulting in [B, 256, 2048] features.

    This architecture is used in DeepSeek-OCR for document understanding.

    Args:
        image_size: int. Input image size (should be 1024 for DeepSeek-OCR).
        sam_patch_size: int. SAM patch size (default: 16).
        sam_embed_dim: int. SAM embedding dimension (default: 768).
        sam_depth: int. Number of SAM layers (default: 12).
        sam_num_heads: int. Number of SAM attention heads (default: 12).
        sam_global_attn_indexes: list. SAM layers with global attention
            (default: [2, 5, 8, 11]).
        clip_hidden_size: int. CLIP hidden dimension (default: 1024).
        clip_num_layers: int. Number of CLIP layers (default: 24).
        clip_num_heads: int. Number of CLIP attention heads (default: 16).
        clip_intermediate_dim: int. CLIP FFN dimension (default: 4096).
        clip_image_size: int. CLIP processes image at this size (default: 224).
        clip_patch_size: int. CLIP patch size (default: 14).
        dtype: string or keras.mixed_precision.DTypePolicy. The dtype to use
            for model computations and weights.

    Example:
    ```python
    # Create DeepEncoder
    deep_encoder = keras_hub.models.DeepEncoderBackbone(
        image_size=1024,
        sam_embed_dim=768,
        sam_depth=12,
        clip_hidden_size=1024,
        clip_num_layers=24,
    )

    # Forward pass
    images = np.random.rand(1, 1024, 1024, 3).astype("float32")
    features = deep_encoder(images)
    print(features.shape)  # (1, 256, 2048)
    ```
    """

    def __init__(
        self,
        image_size=1024,
        sam_patch_size=16,
        sam_embed_dim=768,
        sam_depth=12,
        sam_num_heads=12,
        sam_global_attn_indexes=[2, 5, 8, 11],
        clip_hidden_size=1024,
        clip_num_layers=24,
        clip_num_heads=16,
        clip_intermediate_dim=4096,
        clip_image_size=224,
        clip_patch_size=14,
        dtype=None,
        **kwargs,
    ):
        # === Functional Model ===
        image_input = keras.layers.Input(
            shape=(image_size, image_size, 3),
            name="images"
        )

        sam_features = SAMImageEncoder(
            image_size=image_size,
            patch_size=sam_patch_size,
            embed_dim=sam_embed_dim,
            depth=sam_depth,
            num_heads=sam_num_heads,
            global_attn_indexes=sam_global_attn_indexes,
            name="sam_encoder",
        )(image_input)  # Output: [B, 1024, 16, 16]

        clip_features = DeepEncoderCLIPImageEncoder(
            hidden_size=clip_hidden_size,
            num_layers=clip_num_layers,
            num_heads=clip_num_heads,
            intermediate_dim=clip_intermediate_dim,
            image_size=clip_image_size,
            patch_size=clip_patch_size,
            name="clip_encoder",
        )([image_input, sam_features])  # Output: [B, 257, 1024]

        # Fuse features: remove CLIP CLS, flatten SAM, concatenate
        combined = DeepEncoderFeatureFusion(
            name="feature_fusion"
        )([clip_features, sam_features])  # Output: [B, 256, 2048]

        super().__init__(inputs=image_input, outputs=combined, dtype=dtype, **kwargs)

        # === Config ===
        self.image_size = image_size
        self.sam_patch_size = sam_patch_size
        self.sam_embed_dim = sam_embed_dim
        self.sam_depth = sam_depth
        self.sam_num_heads = sam_num_heads
        self.sam_global_attn_indexes = sam_global_attn_indexes
        self.clip_hidden_size = clip_hidden_size
        self.clip_num_layers = clip_num_layers
        self.clip_num_heads = clip_num_heads
        self.clip_intermediate_dim = clip_intermediate_dim
        self.clip_image_size = clip_image_size
        self.clip_patch_size = clip_patch_size

    def get_config(self):
        config = super().get_config()
        config.update({
            "image_size": self.image_size,
            "sam_patch_size": self.sam_patch_size,
            "sam_embed_dim": self.sam_embed_dim,
            "sam_depth": self.sam_depth,
            "sam_num_heads": self.sam_num_heads,
            "sam_global_attn_indexes": self.sam_global_attn_indexes,
            "clip_hidden_size": self.clip_hidden_size,
            "clip_num_layers": self.clip_num_layers,
            "clip_num_heads": self.clip_num_heads,
            "clip_intermediate_dim": self.clip_intermediate_dim,
            "clip_image_size": self.clip_image_size,
            "clip_patch_size": self.clip_patch_size,
        })
        return config
