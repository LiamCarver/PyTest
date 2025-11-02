import torch
import torch_directml

dml = torch_directml.device()   # this selects your GPU

x = torch.randn(3, 3).to(dml)
print("✅ DirectML GPU is active:", x * 2)