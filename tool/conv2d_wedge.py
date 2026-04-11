import torch.nn.functional as F
import torch


def _compose_conv_kernels(w_second, w_first):
    """
    Kernel W with conv(x, W, pad=p0+p1) equals conv(conv(x, w_first, pad=p0), w_second, pad=p1)
    in the interior (stride=1, dilation=1, groups=1). Same order as linear Wedge:
    new_W = W_L @ weight  =>  z = W_L ( conv(x, weight) + bias ) + b_L.
    """
    c2, c1, k2, kw2 = w_second.shape
    c1a, c0, k1, kw1 = w_first.shape
    if c1 != c1a or k2 != kw2 or k1 != kw1:
        raise ValueError("incompatible conv weights for composition")
    kh, kw = k1 + k2 - 1, k1 + k2 - 1
    out = torch.zeros(c2, c0, kh, kw, dtype=w_first.dtype, device=w_first.device)
    for i in range(c2):
        for j in range(c0):
            for l in range(c1):
                a = w_first[l, j].view(1, 1, k1, k1)
                b = w_second[i, l].view(1, 1, k2, k2).flip(-2, -1)
                out[i, j] += F.conv2d(a, b, padding=k2 - 1).squeeze()
    return out


def _conv_backproject_bias(W_wedge, bias):
    """conv(spatial-constant bias map, W_wedge) reduces to einsum over input channels and kernel."""
    return torch.einsum("coij, o -> c", W_wedge, bias)


class Conv2dWedge:

    def __init__(self, W_L, b_L, W_U, b_U):
        self.W_L = W_L
        self.W_U = W_U
        self.b_L = b_L
        self.b_U = b_U

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'Conv2dWedge(shape={self.shape})'

    @property
    def shape(self):
        return self.W_L.shape

    def accumulate_weight(self, weight, bias, *, attr=None):
        # attr: reserved (e.g. stride/padding metadata); composition assumes stride=dilation=1, groups=1.
        _ = attr
        if self.W_L.shape[1] != weight.shape[0] or self.W_U.shape[1] != weight.shape[0]:
            raise ValueError(
                "weight must have shape [C_mid, C_in, ...] matching wedge input channels C_mid"
            )
        if bias.shape[0] != weight.shape[0]:
            raise ValueError("bias length must match weight.shape[0]")
        new_W_L = _compose_conv_kernels(self.W_L, weight)
        new_W_U = _compose_conv_kernels(self.W_U, weight)
        new_b_L = self.b_L + _conv_backproject_bias(self.W_L, bias)
        new_b_U = self.b_U + _conv_backproject_bias(self.W_U, bias)
        return type(self)(new_W_L, new_b_L, new_W_U, new_b_U)

    def to_bound_tensor(self, F_conv, x):
        # x: BoundTensor (or single-element tuple/list, same as linear Wedge)
        if isinstance(x, (tuple, list)):
            x = x[0]
        L_x, U_x = x.concretize()
        # Same center / half-width as wedge.py / BoundLinf.conv
        mean = 0.5 * (L_x + U_x)
        std = 0.5 * (U_x - L_x)

        def concretize_one_side(W, b, sign):
            # Linear: bound = W @ mean + sign * |W| @ std + b
            # Conv:   bound = conv(mean, W, b) + sign * conv(std, |W|, 0)
            mid = F_conv(mean, W, bias=b)
            rad = F_conv(std, torch.abs(W), bias=None)
            return mid + sign * rad

        L = concretize_one_side(self.W_L, self.b_L, -1)
        U = concretize_one_side(self.W_U, self.b_U, 1)
        p = type(x.perturbation)(L, U)
        return type(x)(L, p)
