import torch.nn.functional as F
import torch


if __name__ == '__main__':
    # import
    from bound_tensor import BoundTensor
    from bound_linf import BoundLinf

    # initialize x as bound tensor
    L = torch.randn(128, 3, 10, 10)
    U = L + 0.02
    p = BoundLinf(L, U)
    x = BoundTensor(L, p)
    # initialize weight and bias for conv2d
    weight = torch.randn(2, 3, 5, 5)
    bias = torch.randn(2)

    # forward
    x0 = x.sample()
    y0 = F.conv2d(x0, weight, bias=bias, stride=1, padding=1, dilation=1, groups=1)
    print(y0.shape)

    # identity conv: F.conv2d(y0, W_L, b_L) == y0 (same shape, stride=1, padding=0)
    C = y0.shape[1]
    W_L = torch.zeros(C, C, 1, 1, dtype=y0.dtype, device=y0.device)
    W_L[torch.arange(C, device=y0.device), torch.arange(C, device=y0.device), 0, 0] = 1.0
    b_L = torch.zeros(C, dtype=y0.dtype, device=y0.device)

    y_id = F.conv2d(y0, W_L, bias=b_L, stride=1, padding=0)
    same = torch.allclose(y_id, y0)
    max_err = (y_id - y0).abs().max().item()
    print("identity conv matches y0:", same, "max_abs_err:", max_err)

#
#     print(y.shape)
#     print(
#         torch.allclose(y, y_hat, rtol=1e-4, atol=1e-5),
#         (y - y_hat).abs().max().item(),
#     )
