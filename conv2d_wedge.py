import torch.nn.functional as F
import torch

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

    def accumulate_weight(self, weight, bias, *, attr=None):
        raise NotImplementedError
        return type(self)(new_W_L, new_b_L, new_W_U, new_b_U)

    def to_bound_tensor(self, x):
        # x: BoundTensor, input bound tensor
        # return output bound tensor
        x = x[0]
        x_L, x_U = x.concretize()
        x_mean = 0.5 * (x_U - x_L)
        x_std  = 0.5 * (x_U - x_L)
        # extract dimension
        bs, c_in, h_in, w_in = x.shape
        _, c_out, h_out, w_out, _, k_h, k_w = self.shape
        # BEGIN: not done yet
        L = torch.empty(bs, c_out, h_out, w_out)
        U = torch.empty(bs, c_out, h_out, w_out)
        def concretize_one_side(W, b, sign):
            for i in range(bs):
                for h in range(h_out):
                    for w in range(w_out):
                        temp = (x_mean[bs, ...] * self.W_L[i, :, h, w, ...])
                        L[i, :, h, w] = temp + self.b_L
            return bound
        # END: not done yet
        L = concretize_one_side(self.W_L, self.b_L, -1)
        U = concretize_one_side(self.W_U, self.b_U, 1)
        # return a BoundTensor of correct type
        p = type(x.perturbation)(L, U)
        z = type(x)(L, p)
        return z
