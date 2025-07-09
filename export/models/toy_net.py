# -*- coding: utf-8 -*-

"""
@Time    : 2025/7/9 14:09
@File    : toy_classify.py
@Author  : zj
@Description:

Toy CNN model inspired by LeNet-5 for flexible testing.

Designed to work with various input sizes and channels.
Suitable for quick export tests (ONNX, TorchScript).

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ToyNet(nn.Module):
    """
    A small CNN inspired by LeNet-5 architecture,
    designed for flexible input sizes and channel counts.

    Args:
        num_classes (int): Number of output classes. Default: 10.
        in_channels (int): Number of input image channels. Default: 1.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1):
        super(ToyNet, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels

        # Feature extractor
        self.features = nn.Sequential(
            # Layer 1: Conv2d
            # Input: [B, C, H, W] e.g., [B, 1, 32, 32]
            # Output: [B, 32, 16, 16] if stride=2
            nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Layer 2: MaxPool
            # Input: [B, 32, 16, 16]
            # Output: [B, 32, 8, 8]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Layer 3: Conv2d
            # Input: [B, 32, 8, 8]
            # Output: [B, 64, 4, 4]
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Layer 4: MaxPool
            # Input: [B, 64, 4, 4]
            # Output: [B, 64, 2, 2]
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Classifier head
        self.classifier = nn.Sequential(
            # Dynamic Linear layer based on feature map size
            # Will be initialized in forward() or reset_classifier()
            nn.Linear(64 * 2 * 2, 128),  # placeholder
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

        # Initialize final linear layer dynamically
        self._initialize_head()

    def _initialize_head(self):
        """Dynamically initialize the first Linear layer based on feature map size."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.in_channels, 32, 32)  # base size
            features = self.features(dummy_input)
            flattened_size = features.view(1, -1).shape[1]
            self.classifier[0] = nn.Linear(flattened_size, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.

        Args:
            x (Tensor): Input tensor of shape [batch_size, in_channels, height, width]

        Returns:
            Tensor: Log-probabilities over classes.
        """
        # Validate input dimensions
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input [B, C, H, W], got {x.shape}")

        # Feature extraction
        x = self.features(x)

        # Flatten except batch dim
        x = torch.flatten(x, 1)

        # Classification head
        x = self.classifier(x)

        return F.log_softmax(x, dim=1)

    def get_info(self) -> str:
        """Returns a summary string of the model configuration."""
        return f"ToyClassifier(num_classes={self.num_classes}, in_channels={self.in_channels})"
