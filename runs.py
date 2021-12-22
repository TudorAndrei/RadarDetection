from lightning_models import (
    ADNet_lightning,
    CNN_lightning,
    ConvMixerLightning,
    ViTLigthning,
)

REGRESSION = "regression"
runs = {
    "cnn": {
        "model": CNN_lightning,
        "hyps": {
            "size": [3, 16, 32, 64],
            "num_blocks": 1,
            "kernel_size": 3,
            "strides": 2,
            "padding": 1,
            "op": "regression",
        },
    },
    "convmixer": {
        "model": ConvMixerLightning,
        "hyps": {
            "size": 32,
            "num_blocks": 4,
            "kernel_size": 11,
            "patch_size": 5,
            "num_classes": 5,
            "lr": 0.3,
            # "res_type": "add",
            "res_type": "cat",
            # "op": "regression",
            "op": "classification",
        },
    },
    # "convrepeater": {
    #     "model": ConvMixerLightning,
    #     "hyps": {
    #         "size": 32,
    #         "num_blocks": 4,
    #         "kernel_size": 11,
    #         "patch_size": 5,
    #         "num_classes": 5,
    #         "lr": 0.3,
    #         "res_type": "add",
    #         "op": "classification",
    #     },
    # },
    "adnet": {
        "model": ADNet_lightning,
        "hyps": {"lr": 0.003, "op": REGRESSION, "optim": "adam"},
    },
    # "vit": {
    #     "model": ViTLigthning,
    #     "hyps": {
    #         "num_classes": 5,
    #         "image_size": 128,
    #         "patch_size": 32,
    #         "lr": 0.03,
    #         "dim": 256,
    #         "depth": 20,
    #         "heads": 9,
    #         "mlp_dim": 256,
    #         "dropout": 0.2,
    #         "emb_dropout": 0.2,
    #     },
    # },
}
