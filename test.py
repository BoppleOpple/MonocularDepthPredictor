import torch

a = torch.randn(3)

b = torch.tensor((10, 100, 1000))

print(torch.mul(a, b))