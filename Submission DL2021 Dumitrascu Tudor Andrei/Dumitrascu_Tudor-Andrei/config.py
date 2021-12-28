from lightning_models import (
    CLASSIFICATION,
    ADNet_lightning,
    CNN_lightning,
    ConvMixerLightning,
    ViTLigthning,
)

REGRESSION = "regression"
config = {
    "cnn": {
        "name": "cnn",
        "model": CNN_lightning,
        "hyps": {
            "size": [3, 16, 32, 64, 128],
            "kernel_size": 3,
            "strides": 2,
            "padding": 1,
            "op": CLASSIFICATION,
        },
    },
    "conv_mixer": {
        "name": "conv_mixer",
        "model": ConvMixerLightning,
        "hyps": {
            "size": 1024,
            "num_blocks": 1,
            "kernel_size": 3,
            "patch_size": 64,
            "num_classes": 5,
            "lr": 0.003,
            "res_type": "add",
            "op": REGRESSION,
        },
    },
    "conv_repeater": {
        "name": "conv_repeater",
        "model": ConvMixerLightning,
        "hyps": {
            "size": 16,
            "num_blocks": 2,
            "kernel_size": 5,
            "patch_size": 2,
            "num_classes": 5,
            "lr": 0.03,
            "res_type": "cat",
            "op": CLASSIFICATION,
        },
    },
    "vit": {
        "name": "vit",
        "model": ViTLigthning,
        "hyps": {
            "num_classes": 5,
            "image_size": 128,
            "patch_size": 8,
            "lr": 0.03,
            "dim": 8,
            "depth": 4,
            "heads": 3,
            "mlp_dim": 8,
            "dropout": 0.1,
            "emb_dropout": 0.1,
            "op": CLASSIFICATION,
        },
    },
    "adnet": {
        "name": "adnet",
        "model": ADNet_lightning,
        "hyps": {
            "lr": 0.003,
            "op": CLASSIFICATION,
        },
    },
}
