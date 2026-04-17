import torch.nn.functional as F
import torch
from functools import partial

import tool


if __name__ == '__main__':
    # args
    device = 'cpu'
    n_sample = 30

    # create input BoundTensor
    L = torch.randn(1, 4, 10, 10)
    U = L + 0.02
    p = tool.BoundLinf(L, U)
    x = tool.BoundTensor(L, p)

    # create net
    weight = torch.randn(2, 4, 3, 3)
    bias = torch.randn(2)
    conv_attr = dict(stride=2, padding=2, dilation=2, groups=1)
    F_conv = partial(F.conv2d, **conv_attr)

    # sample
    y0 = F_conv(x.sample(), weight, bias)

    # identity wedge uses default conv attr (stride=1, padding='same'); pass attr=... to match a specific op
    wedge_out = tool.Conv2dWedge.init_identity(y0)
    print(f'{wedge_out=}')

    # accumulate to compute wedge_in
    wedge_in = wedge_out.accumulate_weight(weight, bias)
    print(f'{wedge_in=}')

#     # compute output BoundTensor
#     y = wedge_in.to_bound_tensor(x)
#
#     print(f'{x=}')
#     print(f'{y=}')
#
#     # try to forward sample of x
#     correct = 0
#     for i in range(n_sample):
#         x0 = x.sample()
#         assert x.contain(x0)
#         y0 = F_conv(x0, weight, bias)
#         if y.contain(y0):
#             correct += 1
#     print(f'[@sample] {correct=} {n_sample=}')
#
#     # try to forward edge sample of x
#     correct = 0
#     for i in range(n_sample):
#         x0 = x.sample_edge_case(eps=1e-6)
#         assert x.contain(x0)
#         y0 = F_conv(x0, weight, bias)
#         if y.contain(y0):
#             correct += 1
#     print(f'[@sample_edge_case] {correct=} {n_sample=}')
