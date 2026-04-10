import torch.nn.functional as F
import torch
from functools import partial


def _in_output_box(y, L_y, U_y):
    return bool(((y >= L_y) & (y <= U_y)).all().item())


if __name__ == '__main__':
    from conv2d_wedge import Conv2dWedge
    from bound_tensor import BoundTensor
    from bound_linf import BoundLinf

    L_x = torch.randn(128, 3, 10, 10)
    U_x = L_x + 0.02
    p_x = BoundLinf(L_x, U_x)
    x = BoundTensor(L_x, p_x)

    weight = torch.randn(2, 3, 5, 5)
    bias = torch.randn(2)

    F_conv = partial(F.conv2d, stride=1, padding=1, dilation=1, groups=1)

    c_out = weight.shape[0]
    W_I = torch.zeros(c_out, c_out, 1, 1, dtype=weight.dtype, device=weight.device)
    W_I[torch.arange(c_out, device=weight.device), torch.arange(c_out, device=weight.device), 0, 0] = 1.0
    b0 = torch.zeros(c_out, dtype=weight.dtype, device=weight.device)
    wedge_out = Conv2dWedge(W_I, b0.clone(), W_I.clone(), b0.clone())
    y_wedge = wedge_out.accumulate_weight(weight, bias)

    # y: BoundTensor whose perturbation is the propagated L∞ box on y
    y = y_wedge.to_bound_tensor(F_conv, x)
    L_y, U_y = y.concretize()
    p_y = y.perturbation

    y_ref = x.conv(F_conv, x, weight, bias)
    L_ref, U_ref = y_ref.concretize()
    print(
        "y (wedge) matches interval conv:",
        torch.allclose(L_y, L_ref) and torch.allclose(U_y, U_ref),
        (L_y - L_ref).abs().max().item(),
        (U_y - U_ref).abs().max().item(),
    )
    print(f"y BoundTensor perturbation: {p_y}")

    n = 100
    inside_hits = 0
    for _ in range(n):
        x_s = torch.rand_like(L_x) * (U_x - L_x) + L_x
        y_s = F_conv(x_s, weight, bias=bias)
        if _in_output_box(y_s, L_y, U_y):
            inside_hits += 1
    print(f"samples inside x box → output in y box (from y BoundTensor): {inside_hits}/{n}")

    outside_hits = 0
    for _ in range(n):
        margin = 0.05 + torch.rand(1).item() * 0.2
        x_s = U_x + margin * torch.rand_like(U_x)
        y_s = F_conv(x_s, weight, bias=bias)
        if not _in_output_box(y_s, L_y, U_y):
            outside_hits += 1
    print(f"samples outside x box → output NOT in y box (from y BoundTensor): {outside_hits}/{n}")
