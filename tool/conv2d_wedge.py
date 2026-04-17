import torch.nn.functional as F
import torch

class Conv2dWedge:

    def __init__(self, W_L, b_L, W_U, b_U, F_conv=None, attr=None):
        self.W_L    = W_L
        self.W_U    = W_U
        self.b_L    = b_L
        self.b_U    = b_U
        self.F_conv = F_conv
        self.attr   = attr

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
        W_L = self.F_conv(weight, self.W_L, None)
        W_U = self.F_conv(weight, self.W_U, None)
        b_L = self.b_L
        b_U = self.b_L
        if bias is not None:
            # (batch size, in channel, out channel, kernel height, kernel width)
            # x (batch size, out channel)
            # -> (batch_size, in channel)
            b_L = torch.einsum('biohw,bo->bi', self.W_L, bias)
            b_U = torch.einsum('biohw,bo->bi', self.W_U, bias)
        return type(self)(W_L, b_L, W_U, b_U)

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

    @staticmethod
    def init_identity(pre_value):
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
        F_conv = lambda x, w, b: F.conv2d(x, w, b, stride=1, dilation=1, padding='same')
        return Conv2dWedge(W_I.clone(), b_I.clone(), W_I.clone(), b_I.clone(), F_conv=F_conv, attr=None)
