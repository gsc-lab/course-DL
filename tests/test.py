import torch
from torch import nn

x = torch.arange(1, 10)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1:nn.Linear = nn.Linear(3, 2)
        self.fc2:nn.Linear = nn.Linear(2,1)
        
    def forward(self, x: torch.Tensor):
        x = nn.ReLU(self.fc1(x))
        x = self.fc2(2)
        

model = MyModel()



raw_x:torch.Tensor = torch.arange(1, 22, dtype=torch.float32).reshape(-1, 3)
w = torch.tensor([0.5, 1.0, 2.0])
b = torch.tensor(0.5)
raw_y:torch.Tensor = (raw_x @ w + b).reshape(-1, 1)

class MyLayer(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(input, output))
        self.bias = nn.Parameter(torch.randn(output))
        
    def forward(self, x):
        return x @ self.weights + self.bias
    
my_layer = MyLayer(3, 1)
criterion:nn.MSELoss = nn.MSELoss()
optimizer = torch.optim.SGD(my_layer.parameters(), lr=0.001)

for epoch in range(10000):
    prediction:torch.Tensor = my_layer(raw_x)
    loss:torch.Tensor = criterion(prediction, raw_y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 1000 == 0:
        print(epoch, loss.item())
    
print(my_layer.weights.T)
print(my_layer.bias)

print(my_layer(raw_x))
print(raw_y)



