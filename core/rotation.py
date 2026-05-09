import torch

class RandomRotation:
    def __init__(self, d, device="cpu", seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        A = torch.randn(d, d, device=device)
        Q, _ = torch.linalg.qr(A)
        self.P = Q  

    def apply(self, x):
        # x: (batch, d)
        return x @ self.P.T

    def invert(self, y):
        return y @ self.P
