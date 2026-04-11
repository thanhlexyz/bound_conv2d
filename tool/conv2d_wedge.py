import torch.nn.functional as F
import torch

def compose_weight(W_bound, weight):
    """Compose conv kernels (stride 1, dilation 1, groups 1, square kernels).

    With ``h = conv2d(x, weight)`` and ``y = conv2d(h, W_bound)`` (wedge / interior
    semantics), returns the composed kernel ``W_composed`` with
    ``conv2d(x, W_composed)`` equivalent in the interior.

    Args:
        W_bound: in shape ``(C_out_bound, C_in_bound, Kh_bound, Kw_bound)``;
            square kernels: ``Kh_bound == Kw_bound``.
        weight: in shape ``(C_out, C_in, Kh, Kw)``; square kernels ``Kh == Kw``;
            and ``C_in_bound == C_out`` (middle channels match).

    Returns:
        out shape ``(C_out_bound, C_in, kh, kw)`` with
        ``kh = Kh + Kh_bound - 1``, ``kw = Kw + Kw_bound - 1``.
    """
    C_out_bound, C_in_bound, Kh_bound, Kw_bound = W_bound.shape
    C_out, C_in, Kh, Kw = weight.shape
    if C_in_bound != C_out or Kh_bound != Kw_bound or Kh != Kw:
        raise ValueError('incompatible conv weights for composition')
    kh, kw = Kh + Kh_bound - 1, Kw + Kw_bound - 1
    out = torch.zeros(C_out_bound, C_in, kh, kw, dtype=weight.dtype, device=weight.device)
    for i in range(C_in_bound):
        # ``(1, C_out_bound, Kh_bound, Kw_bound)``: wedge maps one mid-channel plane to all bound outputs.
        kernel_wedge_in_i = W_bound[:, i, :, :].unsqueeze(0)
        for j in range(C_in):
            # ``(1, 1, Kh, Kw)``: first conv maps input channel j into mid channel l.
            kernel_layer_in_j_to_in_i = weight[i, j].view(1, 1, Kh, Kh)
            out[:, j, :, :] += F.conv_transpose2d(
                kernel_layer_in_j_to_in_i,
                kernel_wedge_in_i,
                bias=None,
                stride=1,
                padding=0,
                output_padding=0,
                groups=1,
            ).squeeze(0)
    return out

def compose_bias(W_bound, bias):
    """Back-project a per-channel bias through ``W_bound`` (spatially constant bias map).

    Args:
        W_bound: shape ``(C_out_bound, C_mid, Kh_bound, Kw_bound)`` — same layout
            as in :func:`compose_weight` (``c`` = wedge output channels, ``o`` = middle /
            conv output channels summed in the einsum).
        bias: shape ``(C_mid,)``, one bias per channel after ``conv2d(..., weight)``.

    Returns:
        shape ``(C_out_bound,)`` — contribution to the composed bias on wedge outputs.
    """
    return torch.einsum('coij, o -> c', W_bound, bias)

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
        # (channel_start_node, channel_end_node, kernel_height, kerner_width)
        return self.W_L.shape

    def accumulate_weight(self, weight, bias, *, attr=None):
        # check shape
        assert self.shape[1] == weight.shape[0]
        assert self.shape[0] == bias.shape[0]
        # compose weight and bias
        new_W_L = compose_weight(self.W_L, weight)
        new_W_U = compose_weight(self.W_U, weight)
        new_b_L = self.b_L + compose_bias(self.W_L, bias)
        new_b_U = self.b_U + compose_bias(self.W_U, bias)
        return type(self)(new_W_L, new_b_L, new_W_U, new_b_U)

    def to_bound_tensor(self, F_conv, x):
        # x: BoundTensor
        if isinstance(x, (tuple, list)):
            x = x[0]
        L, U = x.concretize()
        mean = 0.5 * (L + U)
        std = 0.5 * (U - L)
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
