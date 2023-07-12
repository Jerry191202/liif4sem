import math
from argparse import Namespace
from torch import nn
from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias=bias)


class ConvSeq(nn.Module):
    def __init__(self, args, conv=default_conv):
        super().__init__()
        self.args = args
        n_feats = args.n_feats
        kernel_size = 3
        scale = args.scale

        self.head = conv(1, 1, kernel_size)
        m_body = [
            nn.Sequential(conv(scale**i, scale**(i + 1), kernel_size),
                          nn.ReLU())
            for i in range(round(math.log(n_feats, scale)))
        ]
        self.body = nn.Sequential(*m_body)
        self.tail = conv(n_feats, n_feats, kernel_size)
        self.out_dim = n_feats

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x


@register('conv-seq')
def make_conv(n_feats=64, scale=4):
    args = Namespace()
    args.n_feats = n_feats
    args.scale = scale
    args.n_colors = 1
    return ConvSeq(args)
