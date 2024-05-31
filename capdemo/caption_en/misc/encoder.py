import torch
import torch.nn as nn
import torchvision.models as models

class Encoder(nn.Module):
    """ resnet101
    18 34 50 101 152
    """
    def __init__(self):
        super().__init__()
        pretrained_model = models.resnet101(pretrained=True, progress=True)
        sub_modules = list(pretrained_model.children())
        self.feature_map = nn.Sequential(*sub_modules[:-2])
        self.feature_vector = sub_modules[-2]

    # input (batch_size, 2, 224, 224) output (batch_size, 2048, 7, 7) (batch_size, 2048)
    def forward(self, x):
        x = self.feature_map(x)
        return x, torch.flatten(self.feature_vector(x), 1)