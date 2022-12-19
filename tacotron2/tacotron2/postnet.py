from dataclasses import dataclass
import torch
from torch import nn
import torch.functional as F

from tacotron2.layers import ConvNorm


@dataclass
class PostNetConfig:
    postnet_embedding_dim: int
    postnet_kernel_size: int
    postnet_n_convolutions: int
    n_mel_channels: int


class PostNet(nn.Module):
    """Postnet
        - Five 1-d convolution with 512 channels and kernel size 5
    """

    def __init__(self, config: PostNetConfig):
        super().__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config.n_mel_channels, config.postnet_embedding_dim,
                         kernel_size=config.postnet_kernel_size, stride=1,
                         padding=int((config.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='tanh'),
                nn.BatchNorm1d(config.postnet_embedding_dim))
        )

        for _ in range(1, config.postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(config.postnet_embedding_dim,
                             config.postnet_embedding_dim,
                             kernel_size=config.postnet_kernel_size, stride=1,
                             padding=int((config.postnet_kernel_size - 1) / 2),
                             dilation=1, w_init_gain='tanh'),
                    nn.BatchNorm1d(config.postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(config.postnet_embedding_dim, config.n_mel_channels,
                         kernel_size=config.postnet_kernel_size, stride=1,
                         padding=int((config.postnet_kernel_size - 1) / 2),
                         dilation=1, w_init_gain='linear'),
                nn.BatchNorm1d(config.n_mel_channels))
            )

    def forward(self, x):
        for i in range(len(self.convolutions) - 1):
            x = F.dropout(torch.tanh(self.convolutions[i](x)), 0.5, self.training)
        x = F.dropout(self.convolutions[-1](x), 0.5, self.training)

        return x
