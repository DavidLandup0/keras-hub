from transformers import SegformerForSemanticSegmentation

import keras_hub

DOWNLOAD_URL = "nvidia/segformer-b0-finetuned-ade-512-512"


def set_conv_weights(conv_layer, state_dict):
    conv_weights = state_dict["weight"].numpy().transpose(2, 3, 1, 0)
    conv_bias = state_dict["bias"].numpy()
    conv_layer.set_weights([conv_weights, conv_bias])


def set_dwconv_weights(conv_layer, state_dict):
    conv_weights = state_dict["dwconv.weight"].numpy().transpose(2, 3, 0, 1)
    conv_bias = state_dict["dwconv.bias"].numpy()
    conv_layer.set_weights([conv_weights, conv_bias])


def set_layer_norm_weights(layer_norm, state_dict):
    gamma = state_dict["weight"].numpy()
    beta = state_dict["bias"].numpy()
    layer_norm.set_weights([gamma, beta])


def set_dense_weights(dense_layer, state_dict):
    weight = state_dict["weight"].numpy().T
    bias = state_dict["bias"].numpy()
    dense_layer.set_weights([weight, bias])


def set_hierarchical_encoder_weights(keras_layer, pytorch_layer, key):

    set_layer_norm_weights(
        keras_layer.norm1, pytorch_layer.layer_norm_1.state_dict()
    )

    set_dense_weights(
        keras_layer.attn.q, pytorch_layer.attention.self.query.state_dict()
    )
    set_dense_weights(
        keras_layer.attn.k, pytorch_layer.attention.self.key.state_dict()
    )
    set_dense_weights(
        keras_layer.attn.v, pytorch_layer.attention.self.value.state_dict()
    )
    set_dense_weights(
        keras_layer.attn.proj, pytorch_layer.attention.output.dense.state_dict()
    )

    if keras_layer.attn.sr_ratio > 1:
        set_conv_weights(
            keras_layer.attn.sr, pytorch_layer.attention.self.sr.state_dict()
        )
        set_layer_norm_weights(
            keras_layer.attn.norm,
            pytorch_layer.attention.self.layer_norm.state_dict(),
        )

    set_layer_norm_weights(
        keras_layer.norm2, pytorch_layer.layer_norm_2.state_dict()
    )

    set_dense_weights(
        keras_layer.mlp.fc1, pytorch_layer.mlp.dense1.state_dict()
    )
    set_dwconv_weights(
        keras_layer.mlp.dwconv, pytorch_layer.mlp.dwconv.state_dict()
    )
    set_dense_weights(
        keras_layer.mlp.fc2, pytorch_layer.mlp.dense2.state_dict()
    )


def main():
    model = SegformerForSemanticSegmentation.from_pretrained(DOWNLOAD_URL)
    original_mit = original_mit = model.segformer.encoder

    keras_mit = keras_hub.models.MiTBackbone(
        depths=[2, 2, 2, 2],
        image_shape=(224, 224, 3),
        hidden_dims=[32, 64, 160, 256],
        num_layers=4,
        blockwise_num_heads=[1, 2, 5, 8],
        blockwise_sr_ratios=[8, 4, 2, 1],
        max_drop_path_rate=0.1,
        patch_sizes=[7, 3, 3, 3],
        strides=[4, 2, 2, 2],
    )

    # Indices for the different patch embeddings and layer norms
    proj_indices = [1, 6, 11, 16]
    layer_norm_indices = [4, 9, 14, 19]
    hierarchical_encoder_indices = [
        (2, 0, 0),
        (3, 0, 1),
        (7, 1, 0),
        (8, 1, 1),
        (12, 2, 0),
        (13, 2, 1),
        (17, 3, 0),
        (18, 3, 1),
    ]

    # Loop through the indices to set convolutional and normalization weights
    for i, idx in enumerate(proj_indices):
        set_conv_weights(
            keras_mit.layers[idx].proj,
            original_mit.patch_embeddings[i].proj.state_dict(),
        )
        set_layer_norm_weights(
            keras_mit.layers[idx].norm,
            original_mit.patch_embeddings[i].layer_norm.state_dict(),
        )

    # Set layer normalization weights
    for i, idx in enumerate(layer_norm_indices):
        set_layer_norm_weights(
            keras_mit.layers[idx], original_mit.layer_norm[i].state_dict()
        )

    # Set hierarchical encoder weights
    for layer_idx, block_idx, key in hierarchical_encoder_indices:
        set_hierarchical_encoder_weights(
            keras_mit.layers[layer_idx],
            original_mit.block[block_idx][int(key)],
            key=key,
        )
