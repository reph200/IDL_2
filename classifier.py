import torch.nn as nn
from encoder import Encoder
class Classifier(nn.Module):
    def __init__(self, input_dim=64, num_classes=10):
        super(Classifier, self).__init__()
        self.encoder = Encoder()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        # print(f"Flattened shape before FC: {x.shape}")  # Debugging statement
        x = x.view(x.size(0), -1)  # Flatten the encoder output

        x = self.fc(x)
        return x


