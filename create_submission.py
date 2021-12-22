import os

import pandas as pd
import pretty_errors
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype, Normalize, Pad, ToTensor

from data_utils import get_submission_dataloader
from lightning_models import ADNet_lightning, ViTLigthning

BATCH_SIZE = 128
NW = 1
sub_dir = "submissions"

best_model_path = f"models/adnet/radar-epoch22-val_loss1.28.ckpt"


trans_ = transforms.Compose(
    [
        ToTensor(),
        Pad(
            [5, 0, 4, 0],
        ),
        ConvertImageDtype(torch.float),
    ]
)

hyps = {
    "num_classes": 5,
    "dropout": 0.1,
    "emb_dropout": 0.1,
}

hyp_print = ""
for key, value in hyps.items():
    hyp_print += f"_{key}_{value}"

if __name__ == "__main__":

    img_dir = "test/*"
    file_name = lambda x: x.split("/")[-1]

    model_name = best_model_path.split("/")[-2]
    model = ADNet_lightning().load_from_checkpoint(best_model_path)

    model.eval()
    model.freeze()
    preds = []
    data = get_submission_dataloader(img_dir, BATCH_SIZE, NW, transform=trans_)
    for batch in data:
        img, img_path = batch
        out = model(img)
        predictions = torch.argmax(out, 1).numpy()
        # print(predictions)
        # preds = [(file_name(img_path[i]), pred + 1) for i,pred in enumerate(predictions)]
        for i, pred in enumerate(predictions):
            preds.append((file_name(img_path[i]), pred + 1))
    result_df = pd.DataFrame(preds, columns=["id", "label"])
    result_df.to_csv(os.path.join(sub_dir, f"{model_name}{hyp_print}.csv"), index=False)
    print("submission created")
