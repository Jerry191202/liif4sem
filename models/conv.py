from torch import nn
from models import register


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels,
                     out_channels,
                     kernel_size,
                     padding=(kernel_size // 2),
                     bias=bias)


@register('conv-seq')
class ConvSeq(nn.Module):
    def __init__(self, feats_list, kernel_size=3, n_colors=1,
                 drop_rate=None, conv=default_conv):
        super().__init__()

        m_body = []
        pred = n_colors
        for i, n_feats in enumerate(feats_list):
            m_body.append(conv(pred, n_feats, kernel_size))
            pred = n_feats
            if drop_rate is not None:
                m_body.append(nn.Dropout(drop_rate))
            if i + 1 < len(feats_list):
                m_body.append(nn.ReLU())
        self.body = nn.Sequential(*m_body)
        self.out_dim = feats_list[-1]

    def forward(self, x):
        x = self.body(x)
        return x
