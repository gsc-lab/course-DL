import torch

x = torch.tensor(2.0, requires_grad=True)

y = x * 3
y.retain_grad()

z = y ** 2
z.retain_grad()

z.backward()

print(x.grad)
print(y.grad)
print(z.grad)