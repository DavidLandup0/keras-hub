"""Convert DeepEncoder weights from DeepSeek-OCR PyTorch implementation.

Usage:
python tools/checkpoint_conversion/convert_deepencoder.py
"""

import os
import sys

os.environ["KERAS_BACKEND"] = "jax"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from keras_hub.src.models.deep_encoder.deep_encoder_backbone import (
    DeepEncoderBackbone,
)


def load_hf_models():
    """Load PyTorch models from DeepSeek-OCR."""
    print("Loading HuggingFace DeepSeek-OCR model...")
    print("  (This will download the model if not cached)")

    from transformers import AutoModel

    # Load full model
    hf_model = AutoModel.from_pretrained(
        "deepseek-ai/DeepSeek-OCR",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    hf_model.eval()

    print("  Model loaded successfully!")
    print(f"  Model type: {type(hf_model).__name__}")

    # The model is wrapped in DeepseekOCRForCausalLM
    # Vision components are under model.sam_model, model.vision_model, model.projector
    if hasattr(hf_model, 'model'):
        base_model = hf_model.model
    else:
        base_model = hf_model

    print(f"  Base model type: {type(base_model).__name__}")
    print(f"  Base model attributes: {[a for a in dir(base_model) if not a.startswith('_')][:10]}")

    # Extract components
    if hasattr(base_model, 'sam_model'):
        torch_sam = base_model.sam_model
    else:
        raise AttributeError(f"Cannot find sam_model in {type(base_model).__name__}")

    if hasattr(base_model, 'vision_model'):
        torch_clip = base_model.vision_model
    else:
        raise AttributeError(f"Cannot find vision_model in {type(base_model).__name__}")

    if hasattr(base_model, 'projector'):
        torch_projector = base_model.projector
    else:
        raise AttributeError(f"Cannot find projector in {type(base_model).__name__}")

    print(f"  SAM model: {type(torch_sam).__name__}")
    print(f"  CLIP model: {type(torch_clip).__name__}")
    print(f"  Projector: {type(torch_projector).__name__}")

    return torch_sam, torch_clip, torch_projector


def create_keras_model():
    """Create KerasHub DeepEncoder model."""
    print("Creating KerasHub DeepEncoder...")
    model = DeepEncoderBackbone(
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
    )
    return model


def convert_sam_weights(keras_sam, torch_sam):
    """Convert SAM weights from PyTorch to Keras."""
    print("  Converting SAM weights...")

    torch_dict = torch_sam.state_dict()

    print(f"    PyTorch SAM has {len(torch_dict)} weights")

    # Get the ViTDet backbone inside SAMImageEncoder
    vit_det = keras_sam.vit_det
    print(f"    ViTDet has {len(vit_det.layers)} layers")

    # Helper function to port weights
    def port_weight(keras_var, torch_key, transpose=False):
        if torch_key not in torch_dict:
            print(f"    Warning: {torch_key} not found")
            return False
        torch_tensor = torch_dict[torch_key].cpu().numpy()
        if transpose and torch_tensor.ndim >= 2:
            torch_tensor = torch_tensor.T
        keras_var.assign(torch_tensor)
        return True

    # Find patch embedding layer (ViTDetPatchingAndEmbedding)
    patch_layers = [l for l in vit_det.layers if 'patching' in l.name.lower()]
    print(f"    Found {len(patch_layers)} patching layers")
    patch_layer = patch_layers[0]

    # Convert patch embedding
    # PyTorch: Conv2d weight is (out_ch, in_ch, H, W), Keras: (H, W, in_ch, out_ch)
    patch_proj_weight = torch_dict["patch_embed.proj.weight"].cpu().numpy()
    patch_proj_weight = np.transpose(patch_proj_weight, (2, 3, 1, 0))  # OIHW -> HWIO
    patch_layer.weights[0].assign(patch_proj_weight)  # kernel

    if "patch_embed.proj.bias" in torch_dict:
        patch_layer.weights[1].assign(torch_dict["patch_embed.proj.bias"].cpu().numpy())  # bias

    # Find positional embedding layer (AddPositionalEmbedding)
    pos_layer = [l for l in vit_det.layers if 'positional' in l.name.lower()][0]

    # Convert positional embeddings
    # PyTorch: (1, H, W, C), Keras: (1, H, W, C) - same format
    if "pos_embed" in torch_dict:
        pos_layer.weights[0].assign(torch_dict["pos_embed"].cpu().numpy())

    # Find transformer encoder layers (WindowedTransformerEncoder)
    transformer_layers = [l for l in vit_det.layers if 'windowed_transformer_encoder' in l.name]

    # Convert transformer blocks
    for i in range(12):  # 12 layers
        keras_block = transformer_layers[i]

        # Get weights in order: [ln1_gamma, ln1_beta, ln2_gamma, ln2_beta, qkv_kernel, qkv_bias,
        #                         out_kernel, out_bias, rel_pos_h, rel_pos_w, mlp1_kernel, mlp1_bias,
        #                         mlp2_kernel, mlp2_bias]
        weights = keras_block.weights

        # Layer norm 1
        port_weight(weights[0], f"blocks.{i}.norm1.weight")  # gamma
        port_weight(weights[1], f"blocks.{i}.norm1.bias")    # beta

        # Layer norm 2
        port_weight(weights[2], f"blocks.{i}.norm2.weight")  # gamma
        port_weight(weights[3], f"blocks.{i}.norm2.bias")    # beta

        # QKV projection (combined in PyTorch, split in Keras)
        qkv_weight = torch_dict[f"blocks.{i}.attn.qkv.weight"].cpu().numpy()  # (2304, 768)
        qkv_bias = torch_dict[f"blocks.{i}.attn.qkv.bias"].cpu().numpy()  # (2304,)

        # Keras stores as (768, 2304) and (2304,)
        weights[4].assign(qkv_weight.T)  # qkv kernel
        weights[5].assign(qkv_bias)      # qkv bias

        # Output projection
        proj_weight = torch_dict[f"blocks.{i}.attn.proj.weight"].cpu().numpy()  # (768, 768)
        proj_bias = torch_dict[f"blocks.{i}.attn.proj.bias"].cpu().numpy()
        weights[6].assign(proj_weight.T)  # output kernel
        weights[7].assign(proj_bias)      # output bias

        # Relative position embeddings (if present)
        if f"blocks.{i}.attn.rel_pos_h" in torch_dict:
            port_weight(weights[8], f"blocks.{i}.attn.rel_pos_h")  # rel_pos_h
            port_weight(weights[9], f"blocks.{i}.attn.rel_pos_w")  # rel_pos_w

        # MLP
        mlp1_weight = torch_dict[f"blocks.{i}.mlp.lin1.weight"].cpu().numpy()
        mlp1_bias = torch_dict[f"blocks.{i}.mlp.lin1.bias"].cpu().numpy()
        mlp2_weight = torch_dict[f"blocks.{i}.mlp.lin2.weight"].cpu().numpy()
        mlp2_bias = torch_dict[f"blocks.{i}.mlp.lin2.bias"].cpu().numpy()

        weights[10].assign(mlp1_weight.T)  # mlp1 kernel
        weights[11].assign(mlp1_bias)      # mlp1 bias
        weights[12].assign(mlp2_weight.T)  # mlp2 kernel
        weights[13].assign(mlp2_bias)      # mlp2 bias

    # Find neck layer (Sequential)
    neck = [l for l in vit_det.layers if 'sequential' in l.name.lower()][-1]

    # Convert neck layers
    # Neck conv1: 768 -> 256 (1x1 conv)
    neck_conv1_weight = torch_dict["neck.0.weight"].cpu().numpy()  # (256, 768, 1, 1)
    neck_conv1_weight = np.transpose(neck_conv1_weight, (2, 3, 1, 0))  # OIHW -> HWIO
    neck.layers[0].weights[0].assign(neck_conv1_weight)

    # Neck LayerNorm 1
    port_weight(neck.layers[1].weights[0], "neck.1.weight")  # gamma
    port_weight(neck.layers[1].weights[1], "neck.1.bias")    # beta

    # Neck conv2: 256 -> 256 (3x3 conv)
    neck_conv2_weight = torch_dict["neck.2.weight"].cpu().numpy()  # (256, 256, 3, 3)
    neck_conv2_weight = np.transpose(neck_conv2_weight, (2, 3, 1, 0))  # OIHW -> HWIO
    neck.layers[2].weights[0].assign(neck_conv2_weight)

    # Neck LayerNorm 2
    port_weight(neck.layers[3].weights[0], "neck.3.weight")  # gamma
    port_weight(neck.layers[3].weights[1], "neck.3.bias")    # beta

    # Convert downsampling layers net_2 and net_3
    net2_weight = torch_dict["net_2.weight"].cpu().numpy()  # (512, 256, 3, 3)
    net2_weight = np.transpose(net2_weight, (2, 3, 1, 0))
    keras_sam.net_2.weights[0].assign(net2_weight)

    net3_weight = torch_dict["net_3.weight"].cpu().numpy()  # (1024, 512, 3, 3)
    net3_weight = np.transpose(net3_weight, (2, 3, 1, 0))
    keras_sam.net_3.weights[0].assign(net3_weight)

    print("    SAM weights converted!")


def convert_clip_weights(keras_clip, torch_clip):
    """Convert CLIP weights from PyTorch to Keras."""
    print("  Converting CLIP weights...")

    torch_dict = torch_clip.state_dict()

    print(f"    Keras CLIP has {len(keras_clip.weights)} weights")
    print(f"    PyTorch CLIP has {len(torch_dict)} weights")

    def port_weight(keras_var, torch_key, transpose=False):
        if torch_key not in torch_dict:
            print(f"    Warning: {torch_key} not found")
            return False
        torch_tensor = torch_dict[torch_key].cpu().numpy()
        if transpose and torch_tensor.ndim >= 2:
            torch_tensor = torch_tensor.T
        keras_var.assign(torch_tensor)
        return True

    # Class embedding (stored as (1024,) in Keras)
    print(f"    Converting class embedding...")
    port_weight(keras_clip.class_embedding, "embeddings.class_embedding")

    # Position embedding (stored as (257, 1024) in Keras)
    port_weight(keras_clip.position_embedding, "embeddings.position_embedding.weight")

    # Pre-layer norm
    port_weight(keras_clip.pre_layernorm.gamma, "pre_layrnorm.weight")
    port_weight(keras_clip.pre_layernorm.beta, "pre_layrnorm.bias")

    # Transformer blocks
    for i in range(24):  # 24 layers
        keras_block = keras_clip.transformer_blocks[i]

        # Get block weights
        weights = keras_block.weights

        # Layer norm 1
        port_weight(weights[0], f"transformer.layers.{i}.layer_norm1.weight")  # gamma
        port_weight(weights[1], f"transformer.layers.{i}.layer_norm1.bias")    # beta

        # Self-attention - QKV projection (combined in PyTorch)
        qkv_weight = torch_dict[f"transformer.layers.{i}.self_attn.qkv_proj.weight"].cpu().numpy()
        qkv_bias = torch_dict[f"transformer.layers.{i}.self_attn.qkv_proj.bias"].cpu().numpy()

        # Split into Q, K, V (3 * 1024 = 3072)
        hidden = 1024
        num_heads = 16
        head_dim = hidden // num_heads

        q_weight, k_weight, v_weight = np.split(qkv_weight, 3, axis=0)
        q_bias, k_bias, v_bias = np.split(qkv_bias, 3, axis=0)

        # Reshape for multi-head attention
        # PyTorch: (1024, 1024) -> Keras: (1024, 16, 64)
        q_weight = q_weight.T.reshape(hidden, num_heads, head_dim)
        k_weight = k_weight.T.reshape(hidden, num_heads, head_dim)
        v_weight = v_weight.T.reshape(hidden, num_heads, head_dim)

        q_bias = q_bias.reshape(num_heads, head_dim)
        k_bias = k_bias.reshape(num_heads, head_dim)
        v_bias = v_bias.reshape(num_heads, head_dim)

        # Weights order: [ln1_gamma, ln1_beta, q_kernel, q_bias, k_kernel, k_bias,
        #                 v_kernel, v_bias, out_kernel, out_bias, ln2_gamma, ln2_beta,
        #                 mlp1_kernel, mlp1_bias, mlp2_kernel, mlp2_bias]
        weights[2].assign(q_weight)  # query kernel
        weights[3].assign(q_bias)    # query bias
        weights[4].assign(k_weight)  # key kernel
        weights[5].assign(k_bias)    # key bias
        weights[6].assign(v_weight)  # value kernel
        weights[7].assign(v_bias)    # value bias

        # Output projection
        out_weight = torch_dict[f"transformer.layers.{i}.self_attn.out_proj.weight"].cpu().numpy()
        out_bias = torch_dict[f"transformer.layers.{i}.self_attn.out_proj.bias"].cpu().numpy()

        # PyTorch: (1024, 1024) -> Keras: (16, 64, 1024)
        out_weight = out_weight.T.reshape(num_heads, head_dim, hidden)
        weights[8].assign(out_weight)  # output kernel
        weights[9].assign(out_bias)    # output bias

        # Layer norm 2
        port_weight(weights[10], f"transformer.layers.{i}.layer_norm2.weight")  # gamma
        port_weight(weights[11], f"transformer.layers.{i}.layer_norm2.bias")    # beta

        # MLP
        mlp1_weight = torch_dict[f"transformer.layers.{i}.mlp.fc1.weight"].cpu().numpy()
        mlp1_bias = torch_dict[f"transformer.layers.{i}.mlp.fc1.bias"].cpu().numpy()
        mlp2_weight = torch_dict[f"transformer.layers.{i}.mlp.fc2.weight"].cpu().numpy()
        mlp2_bias = torch_dict[f"transformer.layers.{i}.mlp.fc2.bias"].cpu().numpy()

        weights[12].assign(mlp1_weight.T)  # mlp1 kernel
        weights[13].assign(mlp1_bias)      # mlp1 bias
        weights[14].assign(mlp2_weight.T)  # mlp2 kernel
        weights[15].assign(mlp2_bias)      # mlp2 bias

    print("    CLIP weights converted!")


def convert_projector_weights(keras_projector, torch_projector):
    """Convert projector weights from PyTorch to Keras."""
    print("  Converting projector weights...")

    torch_dict = torch_projector.state_dict()
    print(f"    PyTorch projector has {len(torch_dict)} weights")

    # In DeepSeek-OCR, projector is: MlpProjector(projector_type="linear")
    # Which creates: nn.Linear(2048, 1280)
    # Weight key should be: "layers.weight" and "layers.bias"

    weight_key = "layers.weight" if "layers.weight" in torch_dict else "weight"
    bias_key = "layers.bias" if "layers.bias" in torch_dict else "bias"

    if weight_key not in torch_dict:
        print(f"    Available keys: {list(torch_dict.keys())}")
        raise KeyError(f"Weight key '{weight_key}' not found in projector state dict")

    # Get PyTorch weights
    proj_weight = torch_dict[weight_key].cpu().numpy()  # [1280, 2048]
    proj_bias = torch_dict[bias_key].cpu().numpy()  # [1280]

    print(f"    PyTorch weight shape: {proj_weight.shape}")
    print(f"    PyTorch bias shape: {proj_bias.shape}")

    # Keras Dense expects [in_features, out_features]
    # PyTorch Linear has [out_features, in_features]
    # Need to transpose
    keras_projector.projection.kernel.assign(proj_weight.T)  # [2048, 1280]
    keras_projector.projection.bias.assign(proj_bias)

    print("    Projector weights converted!")


def convert_weights(keras_model, torch_sam, torch_clip, torch_projector=None):
    """Convert weights from PyTorch to Keras."""
    print("\nConverting weights...")

    # Find SAM, CLIP encoders, and projector in the model
    sam_encoder = None
    clip_encoder = None
    projector = None

    for layer in keras_model.layers:
        if hasattr(layer, 'name'):
            if 'sam_encoder' in layer.name:
                sam_encoder = layer
                print(f"  Found SAM encoder: {layer.name}, type: {type(layer).__name__}")
            elif 'clip_encoder' in layer.name:
                clip_encoder = layer
                print(f"  Found CLIP encoder: {layer.name}, type: {type(layer).__name__}")
            elif 'projector' in layer.name:
                projector = layer
                print(f"  Found projector: {layer.name}, type: {type(layer).__name__}")

    if sam_encoder is None:
        raise ValueError("SAM encoder not found in model!")
    if clip_encoder is None:
        raise ValueError("CLIP encoder not found in model!")
    if projector is None:
        raise ValueError("Projector not found in model!")

    # Convert weights
    convert_sam_weights(sam_encoder, torch_sam)
    convert_clip_weights(clip_encoder, torch_clip)

    if torch_projector is not None:
        convert_projector_weights(projector, torch_projector)
    else:
        print("  ⚠️  No PyTorch projector provided, skipping projector weight conversion")

    print("  Weight conversion complete!")
    return keras_model


def test_numerical_equivalence(keras_model, torch_sam, torch_clip, torch_projector=None):
    """Test that outputs match between Keras and PyTorch."""
    print("\nTesting numerical equivalence...")

    # Create test input with fixed seed
    np.random.seed(42)
    test_input = np.random.rand(1, 1024, 1024, 3).astype("float32")

    # Keras forward
    print("  Running Keras forward pass...")
    import time
    start = time.time()
    keras_output = keras_model(test_input)
    print(f"    Keras output shape: {keras_output.shape} (took {time.time()-start:.1f}s)")

    # Get intermediate outputs from Keras
    print("  Getting intermediate Keras outputs...")
    sam_encoder = [l for l in keras_model.layers if 'sam_encoder' in l.name][0]
    clip_encoder = [l for l in keras_model.layers if 'clip_encoder' in l.name][0]

    keras_sam_output = sam_encoder(test_input)
    print(f"    Keras SAM output shape: {keras_sam_output.shape}")
    keras_clip_output = clip_encoder([test_input, keras_sam_output])
    print(f"    Keras CLIP output shape: {keras_clip_output.shape}")

    # PyTorch forward
    print("  Running PyTorch forward pass...")
    with torch.no_grad():
        torch_input = torch.from_numpy(test_input).permute(0, 3, 1, 2)  # NHWC -> NCHW

        # SAM forward
        sam_features = torch_sam(torch_input)
        print(f"    PyTorch SAM output shape: {sam_features.shape}")

        # CLIP forward
        clip_features = torch_clip(torch_input, sam_features)
        print(f"    PyTorch CLIP output shape: {clip_features.shape}")

        # Concatenate (matching DeepEncoder fusion)
        clip_no_cls = clip_features[:, 1:, :]  # Remove CLS token
        sam_flat = sam_features.flatten(2).permute(0, 2, 1)  # [B, 1024, 256] -> [B, 256, 1024]
        fused = torch.cat([clip_no_cls, sam_flat], dim=-1)
        print(f"    PyTorch fused output shape: {fused.shape}")

        # Project (if projector provided)
        if torch_projector is not None:
            torch_output = torch_projector(fused)
            print(f"    PyTorch projected output shape: {torch_output.shape}")
        else:
            torch_output = fused
            print(f"    PyTorch final output shape (no projector): {torch_output.shape}")

        torch_output_np = torch_output.cpu().numpy()
        torch_sam_np = sam_features.cpu().numpy()
        torch_clip_np = clip_features.cpu().numpy()

    # Detailed comparison
    print("\n  SAM Comparison:")
    keras_sam_np = keras_sam_output.numpy() if hasattr(keras_sam_output, 'numpy') else np.array(keras_sam_output)
    print(f"    Keras SAM shape: {keras_sam_np.shape}, mean={np.mean(keras_sam_np):.6f}, std={np.std(keras_sam_np):.6f}")
    print(f"    PyTorch SAM shape: {torch_sam_np.shape}, mean={np.mean(torch_sam_np):.6f}, std={np.std(torch_sam_np):.6f}")
    # Both should be NCHW format
    sam_close = np.allclose(keras_sam_np, torch_sam_np, atol=1e-3)
    print(f"    Match (atol=1e-3): {sam_close}")
    if not sam_close:
        print(f"    Max diff: {np.max(np.abs(keras_sam_np - torch_sam_np)):.6f}")

    print("\n  CLIP Comparison:")
    keras_clip_np = keras_clip_output.numpy() if hasattr(keras_clip_output, 'numpy') else np.array(keras_clip_output)
    print(f"    Keras CLIP shape: {keras_clip_np.shape}, mean={np.mean(keras_clip_np):.6f}, std={np.std(keras_clip_np):.6f}")
    print(f"    PyTorch CLIP shape: {torch_clip_np.shape}, mean={np.mean(torch_clip_np):.6f}, std={np.std(torch_clip_np):.6f}")
    clip_close = np.allclose(keras_clip_np, torch_clip_np, atol=1e-3)
    print(f"    Match (atol=1e-3): {clip_close}")
    if not clip_close:
        print(f"    Max diff: {np.max(np.abs(keras_clip_np - torch_clip_np)):.6f}")

    print(f"\n  Final Output Comparison:")
    keras_output_np = keras_output.numpy() if hasattr(keras_output, 'numpy') else np.array(keras_output)
    print(f"    Keras shape: {keras_output_np.shape}, mean={np.mean(keras_output_np):.6f}, std={np.std(keras_output_np):.6f}")
    print(f"    PyTorch shape: {torch_output_np.shape}, mean={np.mean(torch_output_np):.6f}, std={np.std(torch_output_np):.6f}")

    # Check if weights were converted
    final_match = np.allclose(keras_output_np, torch_output_np, atol=1e-3)
    print(f"    Match (atol=1e-3): {final_match}")
    if not final_match:
        max_diff = np.max(np.abs(keras_output_np - torch_output_np))
        print(f"    Max difference: {max_diff:.6f}")
        print("\n  ⚠ Outputs don't match - check SAM and CLIP comparisons above")
    else:
        print("\n  ✓ All outputs match!")


def main():
    print("="*80)
    print("DeepEncoder Weight Conversion")
    print("="*80)

    # Load PyTorch models
    torch_sam, torch_clip, torch_projector = load_hf_models()

    # Create Keras model
    keras_model = create_keras_model()
    print(f"  Model created: input={keras_model.input_shape}, output={keras_model.output_shape}")

    # Build the model with test input before converting weights
    print("  Building model...")
    np.random.seed(42)
    test_input = np.random.rand(1, 1024, 1024, 3).astype("float32")
    _ = keras_model(test_input)
    print("  Model built!")

    # Convert weights
    keras_model = convert_weights(keras_model, torch_sam, torch_clip, torch_projector)

    # IMPORTANT: Functional models in JAX cache the computation graph with initial weights
    # We need to save and reload to update the cached graph
    print("\n  Saving converted weights...")
    weights_path = "/tmp/deepencoder_converted.weights.h5"
    keras_model.save_weights(weights_path)
    print(f"  Weights saved to {weights_path}")

    # Create a fresh model and load the converted weights
    print("  Creating fresh model and loading weights...")
    keras_model_fresh = create_keras_model()
    _build = keras_model_fresh(test_input)  # Build it first
    keras_model_fresh.load_weights(weights_path)
    print("  Weights loaded!")

    # Test equivalence with the fresh model
    test_numerical_equivalence(keras_model_fresh, torch_sam, torch_clip, torch_projector)

    print("\n" + "="*80)
    print("✅ DeepEncoder weight conversion and verification complete!")
    print("="*80)
    print("\nConverted weights saved to: /tmp/deepencoder_converted.weights.h5")
    print("\nTo use the converted model:")
    print("  model = DeepEncoderBackbone(...)")
    print("  model.load_weights('/tmp/deepencoder_converted.weights.h5')")
    print("="*80)


if __name__ == "__main__":
    main()
