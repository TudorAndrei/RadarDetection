import torch


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten_size = 64 * 29 * 11
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 64, kernel_size=5),
            torch.nn.Dropout(0.2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
        )

        self.cls = torch.nn.Sequential(
            torch.nn.Linear(self.flatten_size, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 5),
        )

    def forward(self, x):
        out = self.conv(x)
        print(out.shape)
        out = out.reshape(-1, self.flatten_size)
        out = self.cls(out)
        return out


