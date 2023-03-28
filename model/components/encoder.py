import torch
from torch import Tensor
from torchvision import models
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        resnet = models.resnet50(pretrained=True)

        for param in resnet.parameters():
            param.requires_grad = False

        modules = list(resnet.children())[:-2]

        self.pretrained_model = nn.Sequential(*modules)
        self.conv_output = nn.Conv2d(in_channels=2048, out_channels=d_model, kernel_size=1, stride=1)
        self.linear = nn.Linear(in_features=256, out_features=40)
        self.d_model = d_model
        self.to(device)

    def forward(self, x: Tensor):
        batch_size = x.size(0)

        x = self.pretrained_model(x) # (batch_size, channels, width, height)

        x = self.conv_output(x) # (batch_size, d_model, width, height)
        print(x.size())
        x = x.reshape((batch_size, self.d_model, x.size(2)*x.size(3))) # (batch_size, d_model, width*height)
        x = self.linear(x)
        x = x.permute((0, 2, 1))
        return x    