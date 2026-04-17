import torch.nn.functional as F
import torch

from .wedge import Wedge

class Conv2dWedge(Wedge):

    def __init__(self, W_L, b_L, W_U, b_U, F_conv=None, attr=None):
        super().__init__(W_L, b_L, W_U, b_U)
        self.F_conv = F_conv
        self.attr   = attr
        # self.W_L.shape = (batch_size, channel_start_node, channel_end_node, kernel_height, kerner_width)
        # self.b_L.shape = (batch_size, channel_start_node)

    def __str__(self):
        return f'Conv2dWedge(shape={self.shape})'

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
        return Conv2dWedge(W_I.clone(), b_I.clone(), W_I.clone(), b_I.clone(), F_conv=None, attr=None)

    def accumulate_weight(self, weight, bias, F_conv, *, attr=None):
        '''
        Compose this wedge with an F_conv map ``y = x @ weight + bias`` (layer backward).

        Given :math:`W_L x + b_L \le f(x) \le W_U x + b_U`.

        Then :math:`A W_L x + A b_L + b \le A f(x) + b \le A W_U x + A b_U + b`.

        The newly accumulated wedge bound is :math:`(A W_L, A b_L + b, A W_U, A b_U + b)`

        Parameters
        ----------
        weight : torch.Tensor
            Layer weight, shape ``(d_out, d_in)``.
        bias : torch.Tensor, optional
            Layer bias, shape ``(d_out,)``. If ``None``, only ``weight`` is applied.

        Returns
        -------
        Wedge
            Composed wedge after propagating through the affine map.
        '''
        # check shape
        # weight.shape = channel_out, channel_in, kernel_height, kernel_width
        # bias.shape   =
        assert self.shape[1] == weight.shape[0]
        assert bias is None or self.shape[1] == bias.shape[0]
        # compose weight and bias
        new_W_L = self.compose_weight(self.W_L, weight)
        new_W_U = self.compose_weight(self.W_U, weight)
        new_b_L, new_b_U = self.b_L, self.b_U
        if bias is not None:
            new_b_L = new_b_L + self.compose_bias(self.W_L, bias)
            new_b_U = new_b_U + self.compose_bias(self.W_U, bias)
        next_attr = attr if attr is not None else self.attr
        out = type(self)(new_W_L, new_b_L, new_W_U, new_b_U, F_conv, attr=next_attr)
        return out

    def to_bound_tensor(self, x):
        # x: BoundTensor
        if isinstance(x, (tuple, list)):
            x = x[0]
        L, U = x.concretize()
        F_conv = self.F_conv
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
        p = type(x.bound)(L, U)
        return type(x)(L, p)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _unfold_params_from_attr(attr, Kh, Kw):
        '''
        Extract unfold parameters (padding, stride, dilation) from a Conv attribute dict.

        Parameters
        ----------
        attr : dict
            ONNX Conv attribute dictionary containing ``'pads'``, ``'strides'``,
            and optionally ``'dilations'``.
        Kh : int
            Kernel height (unused directly; kept for API symmetry with
            :func:`_infer_unfold_params`).
        Kw : int
            Kernel width (unused directly).

        Returns
        -------
        padding : tuple of int
            ``(pad_h, pad_w)``.
        stride : tuple of int
            ``(stride_h, stride_w)``.
        dilation : tuple of int
            ``(dilation_h, dilation_w)``.
        '''
        pads = attr['pads']
        padding = (int(pads[0]), int(pads[1]))
        strides = attr['strides']
        if isinstance(strides, (list, tuple)):
            stride = (int(strides[0]), int(strides[-1]) if len(strides) > 1 else int(strides[0]))
        else:
            s = int(strides)
            stride = (s, s)
        dil = attr.get('dilations', [1, 1])
        if isinstance(dil, (list, tuple)):
            dilation = (int(dil[0]), int(dil[-1]) if len(dil) > 1 else int(dil[0]))
        else:
            d = int(dil)
            dilation = (d, d)
        return padding, stride, dilation

    @staticmethod
    def _infer_unfold_params(Hin, Win, Kh, Kw):
        '''
        Heuristically infer unfold parameters when no Conv attribute is available.

        Searches stride ``∈ [1, 4]``, padding ``∈ [0, 4]`` (with unit dilation)
        for combinations that yield a positive output spatial size, then picks
        the combination whose output size is closest to half the input size.

        Parameters
        ----------
        Hin : int
            Input height.
        Win : int
            Input width.
        Kh : int
            Kernel height.
        Kw : int
            Kernel width.

        Returns
        -------
        padding : tuple of int
            ``(pad_h, pad_w)``.
        stride : tuple of int
            ``(stride_h, stride_w)``.
        dilation : tuple of int
            Always ``(1, 1)``.

        Raises
        ------
        RuntimeError
            If no valid combination of parameters is found.
        '''
        dilation = (1, 1)
        tried = []
        for sh in range(1, 5):
            for sw in range(1, 5):
                for ph in range(0, 5):
                    for pw in range(0, 5):
                        oh = (Hin + 2 * ph - dilation[0] * (Kh - 1) - 1) // sh + 1
                        ow = (Win + 2 * pw - dilation[1] * (Kw - 1) - 1) // sw + 1
                        if oh < 1 or ow < 1:
                            continue
                        tried.append(((ph, pw), (sh, sw), dilation, oh, ow))
        if not tried:
            raise RuntimeError(
                f'Cannot infer unfold params for spatial ({Hin},{Win}), kernel ({Kh},{Kw})'
            )
        tried.sort(
            key=lambda t: abs(t[3] - Hin // 2) + abs(t[4] - Win // 2)
        )
        ph_pw, st, dil, _, _ = tried[0]
        return ph_pw, st, dil

    def _unfold_spatial(x_4d, Kh, Kw, last_wedge):
        '''
        Unfold a 4-D tensor into sliding local blocks for a given kernel size.

        Uses :func:`_unfold_params_from_attr` when ``last_wedge.attr``
        is available, otherwise falls back to :func:`_infer_unfold_params`.

        Parameters
        ----------
        x_4d : torch.Tensor
            Input tensor, shape ``(N, C, H, W)``.
        Kh : int
            Kernel height.
        Kw : int
            Kernel width.
        last_wedge : Conv2dWedge
            Wedge carrying the optional ``attr`` dictionary.

        Returns
        -------
        torch.Tensor
            Unfolded tensor, shape ``(N, C*Kh*Kw, L)`` where ``L`` is the
            number of output spatial positions.
        '''
        _, _, Hin, Win = x_4d.shape
        if last_wedge.attr is not None:
            padding, stride, dilation = _unfold_params_from_attr(
                last_wedge.attr, Kh, Kw
            )
        else:
            padding, stride, dilation = _infer_unfold_params(Hin, Win, Kh, Kw)
        return F.unfold(
            x_4d,
            kernel_size=(Kh, Kw),
            dilation=dilation,
            padding=padding,
            stride=stride,
        )

    def accumulate_relaxed_relu(self, pre_value, enable_alpha=False):
        # BEGIN: AI code
        W_L0, W_U0 = last_wedge.W_L, last_wedge.W_U
        d_out, C, Kh, Kw = W_U0.shape
        flat = C * Kh * Kw
        L, U = last_value.concretize()
        L = L.clamp(max=0)
        U = U.clamp(min=0)
        U = torch.max(U, L + 1e-8)
        upper_d = U / (U - L)
        upper_b = -L * upper_d
        N = upper_d.shape[0]
        if upper_d.dim() != 4:
            raise NotImplementedError(
                'Conv2dWedge ReLU backward expects a 4D pre-activation (N,C,H,W)'
            )
        _, Cin, Hin, Win = upper_d.shape
        if Cin != C:
            raise ValueError(
                f'channel mismatch: wedge C_in={C} vs pre-activation C={Cin}'
            )
        upper_d_unf = _unfold_spatial(upper_d, Kh, Kw, last_wedge)
        upper_b_unf = _unfold_spatial(upper_b, Kh, Kw, last_wedge)
        if upper_d_unf.shape[1] != flat:
            raise ValueError(
                f'unfold channels {upper_d_unf.shape[1]} != wedge flat {flat}'
            )
        Lcols = upper_d_unf.shape[2]
        if enable_alpha:
            raise NotImplementedError
        d_U = d_L = (upper_d_unf > 0.5).float()
        self.alpha_0 = d_L[0].mean(dim=-1).reshape(C, Kh, Kw)

        def widen(W):
            return W.reshape(d_out, flat).t().unsqueeze(0).expand(N, -1, -1)

        pos_W_U = W_U0.clamp(min=0)
        neg_W_U = W_U0.clamp(max=0)
        pos_W_L = W_L0.clamp(min=0)
        neg_W_L = W_L0.clamp(max=0)
        W_U_ = widen(pos_W_U)
        neg_U = widen(neg_W_U)
        W_L_ = widen(neg_W_L)
        neg_L = widen(pos_W_L)
        upper_d_exp = upper_d_unf.unsqueeze(2)
        d_U_exp = d_U.unsqueeze(2)
        d_L_exp = d_L.unsqueeze(2)
        W_U_exp = W_U_.unsqueeze(-1).expand(-1, -1, -1, Lcols)
        neg_U_exp = neg_U.unsqueeze(-1).expand(-1, -1, -1, Lcols)
        W_L_exp = W_L_.unsqueeze(-1).expand(-1, -1, -1, Lcols)
        neg_L_exp = neg_L.unsqueeze(-1).expand(-1, -1, -1, Lcols)
        W_U_n = upper_d_exp * W_U_exp + d_U_exp * neg_U_exp
        W_L_n = upper_d_exp * neg_L_exp + d_L_exp * W_L_exp
        upper_b_exp = upper_b_unf.unsqueeze(2)
        b_U = (W_U_exp * upper_b_exp).sum(dim=1).mean(dim=-1)
        b_L = (neg_L_exp * upper_b_exp).sum(dim=1).mean(dim=-1)
        b_U = b_U[0] + last_wedge.b_U
        b_L = b_L[0] + last_wedge.b_L
        W_U_n = W_U_n.mean(dim=-1)
        W_L_n = W_L_n.mean(dim=-1)

        def unwiden(Wn):
            return Wn[0].transpose(0, 1).reshape(d_out, C, Kh, Kw)

        W_U_f = unwiden(W_U_n)
        W_L_f = unwiden(W_L_n)
        out = Conv2dWedge(
            W_L_f, b_L, W_U_f, b_U, last_wedge.F_conv, attr=last_wedge.attr
        )
        # END
        return out
