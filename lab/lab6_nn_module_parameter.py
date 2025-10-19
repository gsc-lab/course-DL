import torch

from torch import Tensor, nn, optim


raw_features: Tensor = torch.arange(1, 13, dtype=torch.float64).reshape(-1, 3) # 3 X 3
label_weight: Tensor = torch.randn(3, dtype=torch.float64).reshape(-1, 1) # 3 X 1
label_bias: Tensor = torch.rand(1, dtype=torch.float64)
label_y: Tensor = raw_features @ label_weight + label_bias
label_y.reshape(-1, 1)




class MyLayer(nn.Module):
    # 3 X 1 
    def __init__(self, input: int, output: int)->None:
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input, output,dtype=torch.float64))
        self.bias = nn.Parameter(torch.randn(output, dtype=torch.float64))
                
    def forward(self, x: Tensor)->Tensor:
        return x @ self.weights + self.bias
    


layer: MyLayer = MyLayer(3, 1)

prediction:Tensor = layer(raw_features)
    
criterion = nn.MSELoss()

loss:Tensor = criterion(prediction, label_y)
print(f"loss: {loss}")

loss.backward()

print(f"weights: {layer.weights}")
print(f"grad of the weights: {layer.weights.grad}")

print("-"*20)
optimizer = optim.SGD(layer.parameters(), lr=0.1)
optimizer.step()


print(f"weights: {layer.weights}")

