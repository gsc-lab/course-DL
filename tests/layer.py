import torch
from torch import nn


raw_x = torch.arange(1, 100, dtype=torch.float32).reshape(-1, 3)
raw_x = (raw_x - raw_x.mean(dim=0, keepdim=True)) / raw_x.std(dim=0, keepdim=True)

raw_y = raw_x @ torch.tensor([1.0, 2.0, 3.0]).reshape(3, -1) + 0.5


layer = nn.Linear(3, 1)

sgd = torch.optim.SGD(layer.parameters(), lr=0.001)

for epoch in range(3000):
    output  = layer(raw_x)
    param = layer.parameters()

    loss = torch.mean((output - raw_y)**2)
    
    if torch.isnan(loss) or torch.isinf(loss):
        print(f"💥 step {epoch}: loss is {loss}. stop.")
        break
    
    layer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(layer.parameters(), max_norm=100.0)

    sgd.step()
        
    if epoch % 100 == 0:
        print(f"{epoch + 1}th epoch")
        print(f"loss: {loss.item():.6f}")


    
print(f"grad for weight: {layer.weight}")
print(f"grad for bias: {layer.bias}")