import torch.nn.functional as F
import torch
from functools import partial


if __name__ == '__main__':
    # import
    from conv2d_wedge import Conv2dWedge
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
    # print(y0.shape)

    # identity conv: F.conv2d(y0, W_L, b_L) == y0 (same shape, stride=1, padding=0)
    C = y0.shape[1]
    W_L = torch.zeros(C, C, 1, 1, dtype=y0.dtype, device=y0.device)
    W_L[torch.arange(C, device=y0.device), torch.arange(C, device=y0.device), 0, 0] = 1.0
    b_L = torch.zeros(C, dtype=y0.dtype, device=y0.device)
    W_U = W_L.clone()
    b_U = b_L.clone()
    wedge = Conv2dWedge(W_L, b_L, W_U, b_U)
    # print(wedge)

    # test if input for wedge is actually identity
    y1 = F.conv2d(y0, W_L, bias=b_L, stride=1, padding=0)
    same = torch.allclose(y1, y0)
    err = (y1 - y0).abs().max().item()
    # print(f'[y1 vs y0] {same=} {err=}')

    # initialize y as bound tensor
    L = torch.randn_like(y0)
    U = L + 0.02
    p = BoundLinf(L, U)
    y = BoundTensor(L, p)
    print(y)

    # realize bound tensor for y using identity conv2d wedge (same L/U as y)
    F_id = partial(F.conv2d, stride=1, padding=0, dilation=1, groups=1)
    identical_y = wedge.to_bound_tensor(F_id, y)
    Ly, Uy = y.concretize()
    Li, Ui = identical_y.concretize()
    print(
        "wedge to_bound_tensor matches y bounds:",
        torch.allclose(Ly, Li) and torch.allclose(Uy, Ui),
        (Ly - Li).abs().max().item(),
        (Uy - Ui).abs().max().item(),
    )
