import torch
import numpy as np

class QJL:
    def __init__(self, d, device="cpu", seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.d = d
        S = torch.randn(d, d, device=device)
        self.S = S / S.norm(dim=1, keepdim=True)  # ← normalize rows to unit norm

    def quantize(self, r):
        return torch.sign(r @ self.S.T)

    def dequantize(self, z, norm):
        scale = np.sqrt(np.pi / 2) / self.d
        recon = z @ self.S
        return scale * norm.unsqueeze(1) * recon