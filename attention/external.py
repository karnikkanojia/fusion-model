import numpy as np
import torch
from torch import nn
from torch.nn import init


class ExternalAttention(nn.Module):
    def __init__(self, d_model, S=64):
        super().__init__()
        self.d_model = d_model
        self.mk = nn.Linear(d_model, S, bias=False)
        self.mv = nn.Linear(S, d_model, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        init.normal_(self.mk.weight, std=0.001)
        if self.mk.bias is not None:
            init.constant_(self.mk.bias, 0)
        init.normal_(self.mv.weight, std=0.001)
        if self.mv.bias is not None:
            init.constant_(self.mv.bias, 0)

    def forward(self, x):
        n, c, h, w = x.size()
        x = x.view(n, c, -1).permute(0, 2, 1)  # (n, hw, c)
        attn = self.mk(x)  # (n, hw, S)
        attn = self.softmax(attn)  # (n, hw, S)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))  # (n, hw, S)
        x = self.mv(attn)  # (n, hw, c)
        x = x.permute(0, 2, 1).contiguous().view(n, self.d_model, h, w)  # (n, c, h, w)
        return x


if __name__ == "__main__":
    input = torch.randn(1, 1024, 7, 7)
    ea = ExternalAttention(d_model=1024, S=8)
    output = ea(input)
    print(output.shape)
