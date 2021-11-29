from convmixer import ConvMixerModule
import os
from data_utils import get_submission_dataloader
import pytorch_lightning as pl
import pandas as pd

# from tqdm import tqdm
import torch
from glob import glob

pl.seed_everything(42)

BATCH_SIZE = 64
NW = 1
sub_dir = "submissions"

best_model_path = glob("models/*")[-1]
print(best_model_path)
hyps = {
    "size": 7,
    "num_blocks": 5,
    "kernel_size": 9,
    "patch_size": 8,
    "num_classes": 5,
}
hyp_print = ''
for key, value in hyps.items():
    hyp_print += f"_{key}_{value}"

if __name__ == "__main__":

    img_dir = "test/*"
    file_name = lambda x: x.split("/")[-1]

    model_name = best_model_path.split("/")[-1]
    model = ConvMixerModule().load_from_checkpoint(
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
        for i, pred in enumerate(predictions):
            preds.append((file_name(img_path[i]), pred + 1))
    result_df = pd.DataFrame(preds, columns=["id", "label"])
    result_df.to_csv(os.path.join(sub_dir, f"{model_name}{hyp_print}.csv"), index=False)
