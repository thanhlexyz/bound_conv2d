import torch

class BoundTensor(torch.Tensor):
    # designed to wrap and handle different input data requirements
    # for different perturbation on inputs
    def __init__(self, x, perturbation=None):
        self.perturbation = perturbation

    @staticmethod
    def __new__(cls, x, perturbation=None, *args, **kwargs):
        if isinstance(x, torch.Tensor):
            tensor = super().__new__(cls, [], *args, **kwargs)
            tensor.data = x.data
            tensor.requires_grad = x.requires_grad
            return tensor
        else:
            return super().__new__(cls, x, *args, **kwargs)

    def __repr__(self):
        return f'BoundedTensor({self.perturbation})'

    def addmm(self, weight, bias):
        p = self.perturbation.addmm(weight, bias)
        return type(self)(p.L, p)

    def matmul(self, y):
        p = self.perturbation.matmul(y)
        return type(self)(p.L, p)

    def relu(self):
        p = self.perturbation.relu()
        return type(self)(p.L, p)

    def conv(self, F_conv, x, weight, bias):
        p = self.perturbation.conv(F_conv, x, weight, bias)
        return type(self)(p.L, p)

    def squeeze(self, dim):
        p = self.perturbation.squeeze(dim)
        return type(self)(p.L, p)

    def flatten(self, dim):
        p = self.perturbation.flatten(dim)
        return type(self)(p.L, p)

    @property
    def shape(self):
        return self.perturbation.shape

    def concretize(self, *x):
        return self.perturbation.concretize(*x)
