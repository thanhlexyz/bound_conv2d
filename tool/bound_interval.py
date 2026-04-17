import torch.nn.functional as F
import torch

class BoundInterval:

    def __init__(self, x0: torch.Tensor, eps: torch.Tensor):
        '''
        This is the usual encoding of a high-dimensional interval.
        The central and diff of the interval ``x0`` and ``eps`` must have the same shape,
        commonly :math:`(B, d_1, \\ldots, d_k)` with batch size :math:`B`.

        Parameters
        ----------
        x0 : torch.Tensor
            Central point
        eps : torch.Tensor
            Diff from central point
        '''
        assert x0.shape == eps.shape
        self.x0  = x0
        self.eps = eps

    def __repr__(self):
        return str(self)

    def __str__(self):
        if self.x0.numel() > 10:
            return f'Interval({self.x0.mean():0.4f}±{self.eps.mean():0.4f}, shape={self.shape})'
        else:
            return f'Interval({self.x0}±{self.eps})'

    def concretize(self) -> (torch.Tensor, torch.Tensor):
        '''
        Retrieve lower and upper bounds ``(L, U)``.
        '''
        L = self.x0 - self.eps
        U = self.x0 + self.eps
        return L, U

    def sample(self) -> torch.Tensor:
        '''
        Uniformly sample within the interval.

        Returns
        -------
        torch.Tensor
            a concrete tensor of sampled value
        '''
        return (torch.rand_like(self.x0) * 2 - 1) * self.eps + self.x0

    def contain(self, x: torch.Tensor) -> bool:
        '''
        Check if concrete tensor x lies in this interval

        Parameters
        ----------
        x: torch.Tensor
            Input concrete tensor to check.

        Returns
        -------
        bool
            `True` if `x \in [L, U]`
        '''
        if x.shape != self.shape:
            return False
        if (x < self.x0 - self.eps).any():
            return False
        if (x > self.x0 + self.eps).any():
            return False
        return True

    @property
    def shape(self) -> tuple:
        '''
        Return the shape of this interval.
        '''
        return self.x0.shape

    def flatten(self, dim: int):
        '''
        To be called by FlattenNode.

        Returns
        -------
        BoundInterval
            Interval perturbation with flattened ``L`` and ``U``.
        '''
        x0 = torch.flatten(self.x0, dim)
        eps = torch.flatten(self.eps, dim)
        return type(self)(x0, eps)

    def squeeze(self, dim: int):
        '''
        To be called by SqueezeNode.

        Returns
        -------
        BoundInterval
            Interval perturbation with squeezed ``L`` and ``U``.
        '''
        x0 = self.x0.squeeze(dim)
        eps = self.eps.squeeze(dim)
        return type(self)(x0, eps)

    def addmm(self, weight: torch.Tensor, bias: torch.Tensor):
        '''
        Add matmul transformation of interval.

        ``new_x0 = x0 @ weight.T + bias``

        ``new_eps = eps @ weight.abs().T``

        Returns a :class:`BoundInterval(new_x0, new_eps)`.

        Parameters
        ----------
        weight : torch.Tensor
            The 2-dimensional weight
        bias : torch.Tensor
            The 1-dimensional bias

        Returns
        -------
        BoundInterval
            The transformed perturbation
        '''
        x0 = self.x0 @ weight.T + bias
        eps = self.eps @ weight.abs().T
        return type(self)(x0, eps)

    def mm(self, weight: torch.Tensor):
        '''
        Matmul transformation of :math:`\ell_\infty` pertubation.

        ``new_x0 = x0 @ weight.T``

        ``new_eps = eps @ weight.abs().T``

        Returns a :class:`BoundInterval(new_x0, new_eps)`.

        Parameters
        ----------
        weight : torch.Tensor
            The 2-dimensional weight

        Returns
        -------
        BoundInterval
            The transformed perturbation
        '''
        x0 = self.x0 @ weight
        eps = self.eps @ weight.abs()
        return type(self)(x0, eps)

    def relu(self):
        '''
        Relu transformation of :math:`\ell_\infty` perturbation.
        This transformation is interval bound propagation style transformation (IBP).
        For tighter bound, use ``bound_net.crown_forward()`` to invoke ``bound.Wedge``.

        Returns
        -------
        BoundInterval
            The transformed perturbation
        '''
        L, U = self.concretize()
        L, U = F.relu(L), F.relu(U)
        x0 = 0.5 * (U + L)
        eps = 0.5 * (U - L)
        return type(self)(x0, eps)

    def conv(self, F_conv, x, weight, bias):
        x0  = F_conv(self.x0, weight, bias)
        eps = F_conv(self.eps, weight.abs(), None)
        return type(self)(x0, eps)
