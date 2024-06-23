import torch.nn as nn
from encoder import Encoder
class Classifier(nn.Module):
    def __init__(self, input_dim=12, num_classes=10):
        super(Classifier, self).__init__()
        self.encoder = Encoder()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return x


