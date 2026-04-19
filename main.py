import torch.nn.functional as F
import torch
from functools import partial

import tool


if __name__ == '__main__':
    # args
    device = 'cpu'
    n_sample = 1000

    # create input BoundTensor
    x0 = torch.randn(8, 3, 32, 32)
    eps = torch.ones_like(x0) * 0.02
    b = tool.BoundInterval(x0, eps)
    x = tool.BoundTensor(x0, b)
    b1 = tool.BoundInterval(x0, eps * 3)
    x1 = tool.BoundTensor(x0, b1)

    # define network
    weight1 = torch.randn(8, 3, 3, 3)
    bias1   = torch.randn(8)
    attr1   = dict(stride=2, padding=2, dilation=2, groups=1)
    weight2 = torch.randn(16, 8, 3, 3)
    bias2   = torch.randn(16)
    attr2   = dict(stride=2, padding=0, dilation=2, groups=1)

    # forward
    out = F.conv2d(x.sample(), weight1, bias1, **attr1)
    out = F.conv2d(out, weight2, bias2, **attr2)

    # create identity wedge on out
    wedge_out = tool.Conv2dWedge.init_identity(out)
    print(f'{wedge_out=}')
    # accumulate weight of wedge to map in -> out
    wedge_2 = wedge_out.accumulate_layer(weight2, bias2, attr2)
    print(f'{wedge_2=}')
    wedge_1 = wedge_2.accumulate_layer(weight1, bias1, attr1)
    print(f'{wedge_1=}')

    # get bound on y using wedge
    y = wedge_1.to_bound_tensor(x)
    print(f'{y=}')
    print(f'{out.shape=}')
    print(f'{y.contain(out)=}')

    assert y.shape == out.shape
    assert y.contain(out)

#     correct = 0
#     for i in range(n_sample):
#         y0 = F.conv2d(x.sample(), weight, bias, **attr)
#         if y.contain(y0):
#             correct += 1
#     print(f'input in interval {correct/n_sample=}')
#
#     # create layer 1
#     weight = torch.randn(4, 4, 3, 3)
#     bias = torch.randn(4)
#     attr = dict(stride=2, padding=2, dilation=2, groups=1)
#
#     # accumulate weight of wedge to map in -> out
#     wedge_layer_2 = wedge_out.accumulate_layer(weight, bias, attr)
