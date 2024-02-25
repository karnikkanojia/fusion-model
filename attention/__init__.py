import torch.nn as nn
from .cbam import CBAMBlock
from .coordinate import CoordAtt
from .eca import ECAAttention
from .external import ExternalAttention
from .squeezeandexcitation import SEAttention

__all__ = [
    "CBAMBlock",
    "CoordAtt",
    "ECAAttention",
    "ExternalAttention",
    "SEAttention",
]


class Attention(nn.Module):
    def __init__(self, attention_type, **kwargs):
        super().__init__()
        if attention_type == "se":
            self.attention = SEAttention(**kwargs)
        elif attention_type == "eca":
            self.attention = ECAAttention(**kwargs)
        elif attention_type == "cbam":
            self.attention = CBAMBlock(**kwargs)
        elif attention_type == "external":
            self.attention = ExternalAttention(**kwargs)
        elif attention_type == "coordatt":
            self.attention = CoordAtt(**kwargs)
        else:
            raise ValueError(f"Unrecognized attention type {attention_type}")

    def forward(self, x):
        return self.attention(x)
