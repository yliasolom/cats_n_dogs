import torch.nn as nn
from torchvision.models import resnet50

class ImageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageClassifier, self).__init__()
        self.resnet50 = resnet50(pretrained=True)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(2048, 128),  # 2048 - размерность признаков перед слоем
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),  # num_classes - количество классов
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.resnet50(x)
