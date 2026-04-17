import torch

class BoundTensor(torch.Tensor):

    def __init__(self, x: torch.Tensor, bound=None):
        '''
        This implement general bounded version of torch.Tensor,
        wrapping on top of a bound (e.g., ``Interval``, ``Zonotope``, etc).

        x.shape = bound.shape = (batch_size, dim_1, dim_2, ...)

        Parameters
        ----------
        x : torch.Tensor
            a dummy value
        bound : BoundInterval, BoundZonotope
            a bound object to be wrapped
        '''
        self.x = x
        self.bound = bound

    @staticmethod
    def __new__(cls, x, bound=None, *args, **kwargs):
        if isinstance(x, torch.Tensor):
            tensor = super().__new__(cls, [], *args, **kwargs)
            tensor.data = x.data
            tensor.requires_grad = x.requires_grad
            return tensor
        else:
            return super().__new__(cls, x, *args, **kwargs)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return f'BoundTensor({self.bound})'

    def concretize(self, *x):
        return self.bound.concretize(*x)

    def sample(self):
        return self.bound.sample()

    def contain(self, x):
        return self.bound.contain(x)

    @property
    def shape(self):
        return self.bound.shape

    def squeeze(self, dim):
        b = self.bound.squeeze(dim)
        return type(self)(b.x0, b)

    def flatten(self, dim):
        b = self.bound.flatten(dim)
        return type(self)(b.x0, b)

    def addmm(self, weight, bias):
        b = self.bound.addmm(weight, bias)
        return type(self)(b.x0, b)

    def mm(self, y):
        b = self.bound.mm(y)
        return type(self)(b.x0, b)

    def relu(self):
        b = self.bound.relu()
        return type(self)(b.x0, b)

    def conv(self, F_conv, x, weight, bias):
        b = self.bound.conv(F_conv, x, weight, bias)
        return type(self)(b.x0, b)
