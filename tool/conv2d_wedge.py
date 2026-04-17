from functools import partial

import torch
import torch.nn.functional as F

_DEFAULT_CONV_ATTR = dict(stride=1, padding=0, dilation=1, groups=1)


def _normalize_conv_attr(attr):
    if attr is None:
        return dict(_DEFAULT_CONV_ATTR)
    merged = dict(_DEFAULT_CONV_ATTR)
    if isinstance(attr, dict):
        merged.update(attr)
    else:
        for key in ('stride', 'padding', 'dilation', 'groups'):
            if hasattr(attr, key):
                merged[key] = getattr(attr, key)
    return merged


class Conv2dWedge:

    def __init__(self, W_L, b_L, W_U, b_U, attr=None):
        self.W_L = W_L
        self.W_U = W_U
        self.b_L = b_L
        self.b_U = b_U
        self.attr = _normalize_conv_attr(attr)
        self.F_conv = partial(F.conv2d, **self.attr)
        self.F_conv_transpose = partial(F.conv_transpose2d, **self.attr)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'Conv2dWedge(shape={self.shape})'

    @property
    def shape(self):
        # (batch_size, channel_start_node, channel_end_node, kernel_height, kerner_width)
        return self.W_L.shape

    def accumulate_weight(self, weight, bias):
        # extract and check shape
        B, S, E, H, W = self.shape
        O, I, H1, W1 = weight.shape
        assert E == O
        # compose weight
        W_L = torch.zeros(B, S, I, H1, W1)
        W_U = torch.zeros(B, S, I, H1, W1)
        # Per-batch conv2d(X, W[b]) with X = weight^T (I,O,H1,W1), W[b] (S,O,H,W).
        # Not replaceable by conv_transpose2d with the same stride/padding/dilation: transpose
        # conv is the adjoint (gradient) w.r.t. X, not this forward map; shapes/values differ.
        X = weight.transpose(0, 1)
        for b in range(B):
            W_U[b] = self.F_conv(X, self.W_L[b], None).transpose(0, 1)
            W_L[b] = self.F_conv(X, self.W_U[b], None).transpose(0, 1)
        # compose bias
        b_L = self.b_L
        b_U = self.b_L
        if bias is not None:
            bias = bias[None].repeat(B, 1)
            b_L += torch.einsum('bsehw,be->bs', self.W_L, bias)
            b_U += torch.einsum('bsehw,be->bs', self.W_U, bias)
        return type(self)(W_L, b_L, W_U, b_U, attr=self.attr)

    def to_bound_tensor(self, x):
        # x: BoundTensor
        if isinstance(x, (tuple, list)):
            x = x[0]
        L, U = x.concretize()
        mean = 0.5 * (L + U)
        std = 0.5 * (U - L)
        def concretize_one_side(W, b, sign):
            # Linear: bound = W @ mean + sign * |W| @ std + b
            # Conv:   bound = conv(mean, W, b) + sign * conv(std, |W|, 0)
            mid = self.F_conv(mean, W, bias=b)
            rad = self.F_conv(std, torch.abs(W), bias=None)
            return mid + sign * rad
        L = concretize_one_side(self.W_L, self.b_L, -1)
        U = concretize_one_side(self.W_U, self.b_U, 1)
        p = type(x.perturbation)(L, U)
        return type(x)(L, p)

    @staticmethod
    def init_identity(pre_value, attr=None):
        '''
        Initialize an identity Conv2dWedge that map ``pre_value`` with shape ``(B, C, H, W)`` to itself
        '''
        # extract dimension of predecessors
        bs, c, h, w = pre_value.shape
        dtype = pre_value.dtype
        device = pre_value.device
        # create identity wedge for output
        W_I = torch.eye(c).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(bs, 1, 1, 1, 1)
        b_I = torch.zeros(bs, c, dtype=dtype, device=device)
        if attr is None:
            attr = dict(stride=1, dilation=1, padding='same', groups=1)
        else:
            attr = _normalize_conv_attr(attr)
        return Conv2dWedge(W_I.clone(), b_I.clone(), W_I.clone(), b_I.clone(), attr=attr)
