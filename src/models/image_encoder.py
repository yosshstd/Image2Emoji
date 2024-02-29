import timm
import torch
from torch import nn


class ImageEncoder(nn.Module):
    def __init__(
        self, model_name: str, pretrained: bool = True, trainable: bool = True
    ) -> None:
        super().__init__()

        self.model = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )

        for param in self.model.parameters():
            param.requires_grad = trainable

    def forward(self, x):
        return self.model(x)
