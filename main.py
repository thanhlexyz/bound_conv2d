import torch.nn.functional as F
import torch
from functools import partial

import tool


if __name__ == '__main__':
    # args
    device = 'cpu'
    n_sample = 1000

    # create input BoundTensor
    x0 = torch.randn(1, 4, 10, 10)
    eps = torch.ones_like(x0) * 0.02
    b = tool.BoundInterval(x0, eps)
    x = tool.BoundTensor(x0, b)
    b1 = tool.BoundInterval(x0, eps * 3)
    x1 = tool.BoundTensor(x0, b1)

    # create net
    weight = torch.randn(2, 4, 3, 3)
    bias = torch.randn(2)
    attr = dict(stride=2, padding=2, dilation=2, groups=1)

    # compile F_conv and check get example output
    y0 = F.conv2d(x.sample(), weight, bias, **attr)
    print(f'{y0.shape=}')

    # create identity wedge on out
    wedge_out = tool.Conv2dWedge.init_identity(y0)
    print(f'{wedge_out=}')

    # accumulate weight of wedge to map in -> out
    wedge_in = wedge_out.accumulate_layer(weight, bias, attr)
    print(f'{wedge_in=}')

    #
    y = wedge_in.to_bound_tensor(x)
    print(y)

    correct = 0
    for i in range(n_sample):
        y0 = F.conv2d(x.sample(), weight, bias, **attr)
        if y.contain(y0):
            correct += 1
    print(f'input in interval {correct/n_sample=}')

    correct = total = 0
    for i in range(n_sample):
        x0 = x1.sample()
        if not x.contain(x0):
            total += 1
            y0 = F.conv2d(x0, weight, bias, **attr)
            if y.contain(y0):
                correct += 1
    print(f'input not in interval {correct/total=}')
