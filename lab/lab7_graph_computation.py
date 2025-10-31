import torch

a = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(3.0, requires_grad=True)

c = a ** 2
d = b * 2
e = c + d
f = e ** 2

f.backward()

print(f"f: {f.item():.2f}")
print(f"grad of a: {a.grad.item():.2f}")
print(f"grad of b: {b.grad.item():.2f}")

# f.backward() Error 발생! 이유는?

