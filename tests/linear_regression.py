import torch

# h(x) = wx + b
# loss = (wx + b - y)^2

true_w = 2.4
true_b = 0.5

x = torch.linspace(1, 100, 100)
y = true_w * x + true_b

w = torch.tensor(-2.0, requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)
lr = 0.0001

for epoch in range(50000):
    predict_y = w * x + b

    loss = ((predict_y - y)**2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        
        w.grad.zero_()
        b.grad.zero_()
    
print(f"W : {w.item()}, b : {b.item()}")
