from torch import nn
import torch

class TinyCNN(nn.Module):
    """
    A small CNN for MNIST digit classification.
    Input: (batch, 1, 28, 28)
    Output: (batch, 10) logits
    """
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)  # (B,32,28,28)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1) # (B,64,28,28)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)               # (B,64,14,14)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, 28, 28)
        Returns:
            logits: (batch, 10)
        """
        x = self.relu(self.conv1(x))     # (B,32,28,28)
        x = self.pool(self.relu(self.conv2(x)))  # (B,64,14,14)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)        # (B,64*14*14)
        x = self.relu(self.fc1(x))       # (B,128)
        x = self.fc2(x)                  # (B,10)
        return x
