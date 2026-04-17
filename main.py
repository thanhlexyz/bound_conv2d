import torch.nn.functional as F
import torch
from functools import partial

import tool


if __name__ == '__main__':
    # args
    device = 'cpu'
    n_sample = 30

    # create input BoundTensor
    x0 = torch.randn(1, 4, 10, 10)
    eps = torch.ones_like(x0) * 0.02
    b = tool.BoundInterval(x0, eps)
    x = tool.BoundTensor(x0, b)

    # create net
    weight = torch.randn(2, 4, 3, 3)
    bias = torch.randn(2)
    attr = dict(stride=2, padding=2, dilation=2, groups=1)

    # compile F_conv and check get example output
    F_conv = partial(F.conv2d, **attr)
    y0 = F_conv(x.sample(), weight, bias)

    # create identity wedge on out
    wedge_out = tool.Conv2dWedge.init_identity(y0)
    print(f'{wedge_out=}')

    # accumulate weight of wedge to map in -> out
    wedge_in = wedge_out.accumulate_weight(weight, bias, attr)
    print(f'{wedge_in=}')
