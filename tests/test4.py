import torch


with torch.no_grad():
    x = torch.tensor(2.0, requires_grad=True)
    y = torch.tensor(3.0, requires_grad=True)

    z = (x + y + 5.0)

z.backward()

print(f"z.grad: {z.grad}")
print(f"y.grad: {y.grad}")
print(f"x.grad: {x.grad}")