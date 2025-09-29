import torch

x = torch.tensor([[1, 2, 3, 4], [1, 2, 10, 4]])

a, b = torch.max(x, 1)

print(a)
print(b)


