import os
import pretty_errors

import pandas as pd
import pytorch_lightning as pl
import torch

from convmixer_expanded import ConvMixerModule
from data_utils import get_submission_dataloader

pl.seed_everything(42)

BATCH_SIZE = 64
NW = 1
sub_dir = "submissions"

best_model_path = f"models/model_size_16_num_blocks_3_kernel_size_3_patch_size_9_num_classes_5_lr_0.003_res_type_cat/radar-epoch11-val_loss1.32.ckpt"

hyps = {
    "size": 16,
    "num_blocks": 3,
    "kernel_size": 3,
    "patch_size": 9,
    "num_classes": 5,
    "lr": 0.003,
    "res_type": "cat"
}

hyp_print = ''
for key, value in hyps.items():
    hyp_print += f"_{key}_{value}"

if __name__ == "__main__":

    img_dir = "test/*"
    file_name = lambda x: x.split("/")[-1]

    model_name = best_model_path.split("/")[-1]
    model = ConvMixerModule(**hyps).load_from_checkpoint(
        best_model_path, **hyps
    )

    model.eval()
    model.freeze()
    preds = []
    data = get_submission_dataloader(img_dir, BATCH_SIZE, NW)
    for batch in data:
        img, img_path = batch
        img = img.permute(0, 3, 1, 2).float()
        out = model(img)
        predictions = torch.argmax(out, 1).numpy()
        # print(predictions)
        # preds = [(file_name(img_path[i]), pred + 1) for i,pred in enumerate(predictions)]
        for i, pred in enumerate(predictions):
            preds.append((file_name(img_path[i]), pred + 1))
    result_df = pd.DataFrame(preds, columns=["id", "label"])
    result_df.to_csv(os.path.join(sub_dir, f"{model_name}{hyp_print}.csv"), index=False)
    print("submission created")
