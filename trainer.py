import torch
from data_utils import DataGenerator
from model import Model
import numpy as np

device = torch.device('cuda')
BATCH_SIZE = 64
NW = 2
device = torch.device('cuda')
model = Model().to(device)
lossf = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

dataset = DataGenerator(
    img_path="./train", path=r"train.csv", bs=BATCH_SIZE, nw=NW
)
train_ = dataset.get_train()
val_ = dataset.get_val()

epochs = 10
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


