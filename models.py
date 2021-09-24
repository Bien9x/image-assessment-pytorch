import torch
from torch import nn
import timm
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Nima(nn.Module):
    def __init__(self, base_model_name, dropout_rate, n_classes=10):
        super(Nima, self).__init__()
        self.base_model_name = base_model_name
        self.base_model = timm.create_model(base_model_name, pretrained=True)
        modules = list(self.base_model.children())
        n_features = modules[-1].in_features
        modules = modules[:-1]
        self.backbone = nn.Sequential(*modules)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(n_features, n_classes)

    def forward(self, img):
        x = self.backbone(img)
        x = self.dropout(x)
        x = self.fc(x)
        return F.softmax(x, 1)

    def set_parameter_requires_grad(self, fine_tune=False):
        if not fine_tune:
            for p in self.backbone.parameters():
                p.requires_grad = False
