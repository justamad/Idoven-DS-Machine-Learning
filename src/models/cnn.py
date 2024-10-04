import torch
import torch.nn as nn

from typing import Tuple


class CNNModel(nn.Module):

    def __init__(
            self,
            input_channels: int,
            num_classes: int,
            conv_filters: Tuple[int, int, int],
            dropout_rate: float,
    ):
        super(CNNModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=conv_filters[0], kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=conv_filters[0], out_channels=conv_filters[1], kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=conv_filters[1], out_channels=conv_filters[2], kernel_size=3)

        # Adaptive pooling to dynamically handle different input window sizes
        self.pool = nn.AdaptiveMaxPool1d(output_size=10)  # Output size of 10 for the time dimension

        # Fully connected layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(conv_filters[2] * 10, 128)  # conv_filters[2] * 10 time steps after adaptive pooling
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input x shape: (batch_size, channels, window_size)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))

        # Adaptive pooling ensures that we get a fixed output size (10) for the time dimension
        x = self.pool(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
