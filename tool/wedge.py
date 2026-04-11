
class Wedge:

    def __init__(self, W_L, b_L, W_U, b_U):
        self.W_L = W_L
        self.W_U = W_U
        self.b_L = b_L
        self.b_U = b_U

    def __str__(self):
        return f'Wedge(shape={self.shape})'

    def __repr__(self):
        return str(self)

    def accumulate_weight(self, weight, bias): # weight, bias: nn.Parameter
        new_W_L = self.W_L.matmul(weight)
        new_b_L = self.b_L + self.W_L.matmul(bias)
        new_W_U = self.W_U.matmul(weight)
        new_b_U = self.b_U + self.W_U.matmul(bias)
        return type(self)(new_W_L, new_b_L, new_W_U, new_b_U)

    def matmul(self, y):
        new_W_L = self.W_L.matmul(y.T)
        new_b_L = self.b_L
        new_W_U = self.W_U.matmul(y.T)
        new_b_U = self.b_U
        return type(self)(new_W_L, new_b_L, new_W_U, new_b_U)

    @property
    def shape(self):
        return self.W_L.shape

    def to_bound_tensor(self, x): # x: BoundTensor
        # TODO: this is also a place to implement optimize
        # inner problem
        x = x[0]
        L, U = x.concretize()
        # reshape for bmm
        L, U = L.unsqueeze(-1), U.unsqueeze(-1)
        # center and diff for closed form solution of output linear bound
        mean = 0.5 * (U + L)
        std  = 0.5 * (U - L)
        #
        def concretize_one_side(W, b, sign):
            W = W.view(W.size(0), W.size(1), -1)
            bound = W.bmm(mean) + sign * W.abs().bmm(std)
            bound = bound.squeeze(-1) + b
            return bound
        L = concretize_one_side(self.W_L, self.b_L, -1).squeeze(-1)
        U = concretize_one_side(self.W_U, self.b_U, 1).squeeze(-1)
        # return a BoundTensor of correct type
        p = type(x.perturbation)(L, U)
        z = type(x)(L, p)
        return z
