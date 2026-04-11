import torch.nn.functional as F
import torch

class BoundLinf:

    def __init__(self, L, U):
        self.L = L
        self.U = U

    def __str__(self):
        if self.L.numel() > 10:
            return f'PerturbationLInfNorm(mean={self.L.mean():0.4f}->{self.U.mean():0.4f}, shape={self.shape})'
        else:
            return f'PerturbationLInfNorm({self.L}->{self.U})'

    def concretize(self):
        return self.L, self.U

    def sample(self):
        return torch.rand_like(self.L) * (self.U - self.L) + self.L

    def sample_edge_case(self, eps=1e-6):
        return torch.rand_like(self.L) * (self.U - self.L) + self.L

    def flatten(self, dim):
        self.L = torch.flatten(self.L, dim)
        self.U = torch.flatten(self.U, dim)
        return self

    @property
    def mid(self):
        return 0.5 * (self.L + self.U)

    @property
    def diff(self):
        return 0.5 * (self.U - self.L)

    def addmm(self, weight, bias):
        mid = torch.addmm(bias, self.mid, weight.T)
        diff = self.diff.matmul(weight.abs().T)
        L = mid - diff
        U = mid + diff
        return type(self)(L, U)

    def matmul(self, y):
        L = self.L.matmul(y)
        U = self.U.matmul(y)
        return type(self)(L, U)

    def conv(self, F_conv, x, weight, bias):
        L, U = self.concretize()
        mid  = 0.5 * (L + U)
        diff = 0.5 * (U - L)
        diff = F_conv(diff, weight.abs(), None)
        mid = F_conv(mid, weight, bias)
        L = mid - diff
        U = mid + diff
        return type(self)(L, U)

    def matmul_final_coeff(self, y):
        # Interval propagation for a fixed coefficient matrix that may
        # contain mixed signs (e.g., verification objective C).
        y_pos = torch.clamp(y, min=0)
        y_neg = torch.clamp(y, max=0)
        L = self.L.matmul(y_pos) + self.U.matmul(y_neg)
        U = self.U.matmul(y_pos) + self.L.matmul(y_neg)
        return type(self)(L, U)

    def relu(self):
        return type(self)(F.relu(self.L), F.relu(self.U))

    def squeeze(self, dim):
        return type(self)(self.L.squeeze(dim), self.U.squeeze(dim))

    @property
    def shape(self):
        return self.L.shape

    def contain(self, x):
        if x.shape != self.shape:
            return False
        if (x < self.L).any():
            return False
        if (x > self.U).any():
            return False
        return True
