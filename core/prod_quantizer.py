import torch
from .mse_quantizer import TurboQuantMSE
from .qjl import QJL

class TurboQuantProd:
    def __init__(self, d, codebook_path, device="cpu"):
        self.device = device
        self.mse = TurboQuantMSE(d, codebook_path, device=device)
        self.qjl = QJL(d, device=device)

    def quantize(self, x):
        # x: (batch, d)

        idx = self.mse.quantize(x)
        x_hat = self.mse.dequantize(idx)

        r = x - x_hat
        z = self.qjl.quantize(r)
        norm = torch.norm(r, dim=1)

        return (idx, z, norm)

    def dequantize(self, packed):
        idx, z, norm = packed

        x_mse = self.mse.dequantize(idx)
        x_qjl = self.qjl.dequantize(z, norm)

        return x_mse + x_qjl
    def quantize_mse_only(self, x):
        idx = self.mse.quantize(x)
        return idx

    def dequantize_mse_only(self, idx):
        return self.mse.dequantize(idx)