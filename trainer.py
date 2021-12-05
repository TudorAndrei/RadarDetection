import torch
from data_utils import DataGenerator
from convmixer import ConvMixer
import numpy as np

device = torch.device('cuda')
BATCH_SIZE = 64
NW = 8
device = torch.device('cuda')
model = ConvMixer(size=8, num_blocks=5, kernel_size=9, patch_size=9, num_classes=5).to(device)
# model = Model().to(device)
lossf = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

dataset = DataGenerator(
    img_dir="./train", path=r"train.csv", bs=BATCH_SIZE, nw=NW
)
train_ = dataset.get_train()
val_ = dataset.get_val()

epochs = 100
for epoch in range(epochs):
    model.train()
    for img, label in train_:

        model.zero_grad()

        img = img.float().to(device).permute(0,3,1,2)
        label = label.to(device)
        output = model(img)

        loss = lossf(output, label)
        loss.backward()
        optimizer.step()
    accs = []
    model.eval()
    for batch in val_:

        img, label = batch
        img = img.float().to(device).permute(0,3,1,2)
        label = label.to(device)
        with torch.no_grad():
            output = model(img)

        predict = output.argmax(1)
        acc = (predict == label).float().mean().detach().cpu().numpy()
        accs.append(acc)
    print(np.mean(accs))


