import torch.nn.functional as F
import torch

from .wedge import Wedge

# keyword arguments accepted by ``torch.nn.functional.conv2d``
_CONV2D_ATTRS = ('stride', 'padding', 'dilation', 'groups', 'padding_mode')

class Conv2dWedge(Wedge):

    def __init__(self, W_L, b_L, W_U, b_U, attr=None):
        super().__init__(W_L, b_L, W_U, b_U)
        self.attr = attr

    def __str__(self):
        return f'Conv2dWedge(shape={self.shape})'

    @staticmethod
    def init_identity(pre_value):
        bs, c, h, w = pre_value.shape
        dtype = pre_value.dtype
        device = pre_value.device
        W_I = torch.eye(c, dtype=dtype, device=device)
        W_I = W_I.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).repeat(bs, 1, 1, 1, 1)
        b_I = torch.zeros(bs, c, dtype=dtype, device=device)
        attr = dict(stride=1, padding=0, dilation=1, groups=1)
        return Conv2dWedge(W_I.clone(), b_I.clone(), W_I.clone(), b_I.clone(), attr=attr)

    @staticmethod
    def _to_pair(value):
        # Expand a spatial parameter ``value`` to a ``(h, w)`` tuple.
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (tuple, list)):
            if len(value) == 1:
                v = int(value[0])
                return (v, v)
            if len(value) == 2:
                return (int(value[0]), int(value[1]))
            raise ValueError(f'value must be int or length-1/2 sequence, got length {len(value)}')
        raise TypeError(f'value must be int or sequence, got {type(value).__name__}')

    def _multiply_param(self, p1: tuple, p2: tuple):
        p1 = self._to_pair(p1)
        p2 = self._to_pair(p2)
        return (p1[0] * p2[0], p1[1] * p2[1])

    def _add_param(self, p1: tuple, p2: tuple):
        p1 = self._to_pair(p1)
        p2 = self._to_pair(p2)
        return (p1[0] + p2[0], p1[1] + p2[1])

    def _accumulate_attr(self, attr1, attr2):
        assert attr1['groups'] == 1
        assert attr2['groups'] == 1
        attr = {
            'stride': self._multiply_param(attr1['stride'], attr2['stride']),
            'dilation': self._multiply_param(attr1['dilation'], attr2['dilation']),
            'padding': self._add_param(attr1['padding'], attr2['padding']),
            'groups': 1,
        }
        return attr

    def accumulate_layer(self, weight, bias=None, attr=None):
        # self.attr + attr -> new_attr
        new_attr = self._accumulate_attr(self.attr, attr)

        # extract and check shapes
        B, S, E, _, _ = self.shape
        O, I, _, _ = weight.shape
        assert E == O, f'wedge end channels {E} must match weight out_channels {O}'

        # self.W + weight -> new_W
        X = weight.transpose(0, 1)
        fuse_attr = dict(stride=1, padding=0, dilation=self.attr['dilation'], groups=1)
        probe = F.conv2d(X, self.W_L[0], None, **fuse_attr)
        _, _, H, W = probe.shape
        dtype, device = self.W_L.dtype, self.W_L.device
        new_W_L = torch.zeros(B, S, I, H, W, dtype=dtype, device=device)
        new_W_U = torch.zeros(B, S, I, H, W, dtype=dtype, device=device)
        # TODO: vectorize this for loop on batch size
        for b in range(B):
            # this fuse_attr is for conv2d to combine weight of two conv2d
            new_W_U[b] = F.conv2d(X, self.W_L[b], bias=None, **fuse_attr).transpose(0, 1)
            new_W_L[b] = F.conv2d(X, self.W_U[b], bias=None, **fuse_attr).transpose(0, 1)
        # propagate layer bias: contribution at each wedge tap is sum_e W_*[b, s, e, ...] * bias[e]
        new_b_L, new_b_U = self.b_L, self.b_U
        if bias is not None:
            bias_b = bias.unsqueeze(0).expand(B, -1)
            new_b_L = torch.einsum('bsehw,be->bs', self.W_L, bias_b)
            new_b_U = torch.einsum('bsehw,be->bs', self.W_U, bias_b)
        return type(self)(new_W_L, new_b_L, new_W_U, new_b_U, attr=new_attr)

    def accumulate_relaxed_relu(self, pre_value, enable_alpha=False):
        raise NotImplementedError
