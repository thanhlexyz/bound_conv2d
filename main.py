import torch.nn.functional as F
import torch
from functools import partial

import tool


if __name__ == '__main__':
    # args
    device = 'cpu'
    n_sample = 100


    # create input BoundTensor
    L = torch.randn(128, 3, 10, 10)
    U = L + 0.02
    p = tool.BoundLinf(L, U)
    x = tool.BoundTensor(L, p)

    # create net
    weight = torch.randn(2, 3, 5, 5)
    bias = torch.randn(2)

    F_conv = partial(F.conv2d, stride=1, padding=1, dilation=1, groups=1)

    # create identity wedge for output
    C_out = weight.shape[0]
    W_I = torch.zeros(C_out, C_out, 1, 1)
    W_I[torch.arange(C_out), torch.arange(C_out), 0, 0] = 1.0
    b_I = torch.zeros(C_out).to(weight)
    W_I = W_I.to(weight)
    wedge_out = tool.Conv2dWedge(W_I.clone(), b_I.clone(), W_I.clone(), b_I.clone())

    # accumulate to compute wedge_in
    wedge_in = wedge_out.accumulate_weight(weight, bias)

    # compute output BoundTensor
    y = wedge_in.to_bound_tensor(F_conv, x)

    print(f'{x=}')
    print(f'{y=}')

    # try to forward sample of x
    correct = 0
    for i in range(n_sample):
        x0 = x.sample()
        assert x.contain(x0)
        y0 = F_conv(x0, weight, bias)
        if y.contain(y0):
            correct += 1
    print(f'{correct=} {n_sample=}')

    # try to forward sample not in x
    def neg_sample(p, eps=1e-4):
        noise = torch.rand_like(p.L) * eps
        samples = [
            p.L - noise,
            p.U + noise,
        ]
        return samples
    correct = 0
    for i in range(n_sample):
        for x0 in neg_sample(x.perturbation):
            assert not x.contain(x0)
            y0 = F_conv(x0, weight, bias)
            if not y.contain(y0):
                correct += 1
    print(f'{correct=} {n_sample*2=}')





#     # y: BoundTensor whose perturbation is the propagated L∞ box on y
#     y = y_wedge.to_bound_tensor(F_conv, x)
#     L_y, U_y = y.concretize()
#     p_y = y.perturbation
#
#     y_ref = x.conv(F_conv, x, weight, bias)
#     L_ref, U_ref = y_ref.concretize()
#     print(
#         "y (wedge) matches interval conv:",
#         torch.allclose(L_y, L_ref) and torch.allclose(U_y, U_ref),
#         (L_y - L_ref).abs().max().item(),
#         (U_y - U_ref).abs().max().item(),
#     )
#     print(f"y BoundTensor perturbation: {p_y}")
#
#     n = 100
#     inside_hits = 0
#     for _ in range(n):
#         x_s = torch.rand_like(L_x) * (U_x - L_x) + L_x
#         y_s = F_conv(x_s, weight, bias=bias)
#         if _in_output_box(y_s, L_y, U_y):
#             inside_hits += 1
#     print(f"samples inside x box → output in y box (from y BoundTensor): {inside_hits}/{n}")
#
#     outside_hits = 0
#     for _ in range(n):
#         margin = 0.05 + torch.rand(1).item() * 0.2
#         x_s = U_x + margin * torch.rand_like(U_x)
#         y_s = F_conv(x_s, weight, bias=bias)
#         if not _in_output_box(y_s, L_y, U_y):
#             outside_hits += 1
#     print(f"samples outside x box → output NOT in y box (from y BoundTensor): {outside_hits}/{n}")
