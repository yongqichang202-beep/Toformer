import torch
import math

def morlet_cwt(x, scales, w0=6.0):
    if x.dim() == 1:
        x = x.unsqueeze(0)
    B, L = x.shape
    t = torch.arange(L, device=x.device).float() - L // 2
    Xf = torch.fft.fft(x, dim=1)
    outputs = []
    for a in scales:
        psi = torch.exp(1j * w0 * t / a) * torch.exp(-t**2 / (2 * a * a)) / math.sqrt(a)
        psi = psi * (math.pi ** -0.25)
        psi_f = torch.fft.fft(psi, n=L)
        conv_f = Xf * torch.conj(psi_f)
        cwt = torch.fft.ifft(conv_f, n=L)
        outputs.append(torch.abs(cwt))
    return torch.stack(outputs, dim=1)
