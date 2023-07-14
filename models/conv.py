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
        feats_list = args.feats_list
        n_colors = args.n_colors
        kernel_size = 3

        m_body = []
        pred = n_colors
        for i, n_feats in enumerate(feats_list):
            m_body.append(conv(pred, n_feats, kernel_size))
            pred = n_feats
            if i + 1 < len(feats_list):
                m_body.append(nn.ReLU())
        self.body = nn.Sequential(*m_body)
        self.out_dim = feats_list[-1]

    def forward(self, x):
        x = self.body(x)
        return x


@register('conv-seq')
def make_conv(feats_list):
    args = Namespace()
    args.feats_list = feats_list
    args.n_colors = 1
    return ConvSeq(args)
