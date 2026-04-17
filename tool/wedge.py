import torch

class Wedge:

    def __init__(self, W_L, b_L, W_U, b_U):
        '''
        Batched linear bounds of the form ``bound = W @ x + b``.

        Two sides ``(W_L, b_L)`` and ``(W_U, b_U)`` capture lower and upper bound of
        a computational node w.r.t input variable ``x``.

        Parameters
        ----------
        W_L : torch.Tensor
            Lower weight, shape ``(batch, out_dim, in_dim)``.
        b_L : torch.Tensor
            Lower bias, shape ``(batch, out_dim)``.
        W_U : torch.Tensor
            Upper weight, same shape as ``W_L``.
        b_U : torch.Tensor
            Upper bias, same shape as ``b_L``.
        '''
        self.W_L = W_L
        self.W_U = W_U
        self.b_L = b_L
        self.b_U = b_U

    def __str__(self):
        return f'Wedge(shape={self.shape})'

    def __repr__(self):
        return str(self)

    @property
    def shape(self):
        return self.W_L.shape

    def accumulate_weight(self, weight, bias=None):
        '''
        Compose this wedge with an affine map ``y = x @ weight + bias`` (layer backward).

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
        new_W_L = self.W_L.matmul(weight)
        new_W_U = self.W_U.matmul(weight)
        new_b_L = self.b_L
        new_b_U = self.b_U
        if bias is not None:
            new_b_L = new_b_L + self.W_L.matmul(bias)
            new_b_U = new_b_U + self.W_U.matmul(bias)
        return type(self)(new_W_L, new_b_L, new_W_U, new_b_U)

    def accumulate_relaxed_relu(self, pre_value, enable_alpha=False):
        '''
        CROWN backward pass through ReLU.

        Linearises the ReLU with the standard triangle relaxation:

        .. math::

            \\lambda^U = \\frac{U}{U - L}, \\quad
            b^U = -\\frac{L \\cdot U}{U - L}

        where ``L`` and ``U`` are the concrete pre-activation bounds from
        ``pre_value``.  The lower slope ``lambda^L`` is set to ``1`` when
        ``lambda^U > 0.5``, else ``0``.

        Parameters
        ----------
        last_wedge : Wedge
            Linear bound from the predecessor layer.
        last_value : BoundTensor
            Concrete interval value of the pre-activation (used to compute
            the relaxation slopes).
        start_name : str
            Name of the starting node (unused here, kept for API consistency).
        enable_alpha : bool, optional
            If ``True``, use per-neuron alpha parameters (not yet implemented).

        Returns
        -------
        Wedge
            Updated linear bound after propagating through the ReLU relaxation.
        '''
        # extract shape
        batch_size, in_features, out_features = self.shape
        # concretize pre-activation value
        L, U = pre_value.concretize()
        L = L.clamp(max=0)
        U = U.clamp(min=0)
        U = torch.max(U, L + 1e-8)
        # compute lower and upper bound of relu
        upper_d = U / (U - L)
        upper_b = - L * upper_d
        upper_d = upper_d.unsqueeze(1)
        if enable_alpha:
            raise NotImplementedError
        else:
            d_U = d_L = (upper_d > 0.5).float()
            self.alpha_0 = d_L.squeeze(1)
        pos_W_U  = self.W_U.clamp(min=0)
        neg_W_U  = self.W_U.clamp(max=0)
        W_U      = upper_d * pos_W_U + d_U * neg_W_U
        mult_W_U = pos_W_U.view(batch_size, in_features, -1)
        b_U      = mult_W_U.matmul(upper_b.view(batch_size, -1, 1)).squeeze(-1)
        pos_W_L  = self.W_L.clamp(min=0)
        neg_W_L  = self.W_L.clamp(max=0)
        W_L      = upper_d * neg_W_L + d_L * pos_W_L
        mult_W_L = neg_W_L.view(batch_size, in_features, -1)
        b_L      = mult_W_L.matmul(upper_b.view(batch_size, -1, 1)).squeeze(-1)
        b_L     += self.b_L
        b_U     += self.b_U
        # initialize new wedge
        out = Wedge(W_L, b_L, W_U, b_U)
        return out

    def to_bound_tensor(self, x):
        '''
        Concretize bounds on the input :class:`tool.bound.bound_tensor.BoundTensor` box.

        For input box ``[x0 ± eps]``,
        evaluate batched linear forms ``[W @ x0 ± |W| @ eps + b]``.

        Parameters
        ----------
        x : BoundTensor or tuple
            If a tuple/list, ``x[0]`` is used (same convention as :meth:`crown_forward`).

        Returns
        -------
        tool.bound.bound_tensor.BoundTensor
            Output with ``bound`` of type ``type(x.bound)`` and concrete bounds ``(L, U)``.
        '''
        x = x[0]
        x0 = x.bound.x0.unsqueeze(-1)
        eps = x.bound.eps.unsqueeze(-1)
        def concretize_one_side(W, b, sign):
            W = W.view(W.size(0), W.size(1), -1)
            bound = W.bmm(x0) + sign * W.abs().bmm(eps)
            bound = bound.squeeze(-1) + b
            return bound
        L = concretize_one_side(self.W_L, self.b_L, -1)
        U = concretize_one_side(self.W_U, self.b_U, 1)
        x0 = 0.5 * (U + L)
        eps = 0.5 * (U - L)
        b = type(x.bound)(x0, eps)
        b = type(x)(x0, b)
        return b

    @staticmethod
    def init_identity(pre_value):
        '''
        Identity wedge for activations with concrete shape ``(batch, d)``.

        Sets ``W_L = W_U`` to stacked identity per batch row and zero biases.

        Parameters
        ----------
        pre_value : torch.Tensor
            Tensor whose leading dimensions are ``(batch, d)`` (e.g. post-linear activation).

        Returns
        -------
        Wedge
            Identity linear map from the ``d``-dimensional box to itself.
        '''
        bs, d = pre_value.shape
        dtype = pre_value.dtype
        device = pre_value.device
        C = torch.eye(d, dtype=dtype, device=device)[None].repeat(bs, 1, 1)
        W_L = W_U = C
        b_L = b_U = torch.zeros(bs, d, dtype=dtype, device=device)
        return Wedge(W_L, b_L, W_U, b_U)
