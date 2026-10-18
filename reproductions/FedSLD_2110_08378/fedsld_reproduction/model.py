"""Four-layer CNN reconstructed from the architecture stated in the paper."""

import torch
from torch import nn


class PaperCNN(nn.Module):
    """Two 5x5 convolutions followed by two fully connected layers.

    The paper specifies the layer count, kernel sizes, and 500-unit hidden
    layer. The 20/50 channel widths, ReLU, and max-pooling are documented
    reconstruction assumptions based on the classic LeNet/PySyft model.
    """

    def __init__(self, in_channels: int, num_classes: int, image_size: int = 28):
        super().__init__()
        if image_size < 16:
            raise ValueError("image_size is too small for two 5x5 convolutions")
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 20, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, in_channels, image_size, image_size)
            flattened = int(self.features(dummy).numel())
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))

