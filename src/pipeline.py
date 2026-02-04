import torch.nn as nn

from src.backbone import ResNet50
from src.head import ClassifyHead


class HydraNet(nn.Module):
    def __init__(self, cfg):
        super(HydraNet, self).__init__()
        self.backbone = ResNet50()
        self.head = ClassifyHead(
            n_class=cfg["general"]["n_class"],
            input_dim=self.backbone.feature_dim,
            feature_dim=cfg["model"]["feature_dim"],
        )

    def forward(self, x):
        feat = self.backbone(x)
        out = self.head(feat)
        return out
