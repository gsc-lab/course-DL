import torch
from torch import Tensor, nn, optim

# 재현성
torch.manual_seed(10)
torch.set_default_dtype(torch.float64)

# ----- 데이터 생성 (4 x 3 특징, 1 타깃) -----
raw_features: Tensor = torch.arange(1, 13, dtype=torch.float64).reshape(-1, 3)      # (4,3)
true_weight:  Tensor = torch.randn(3, 1)                       # (3,1)
true_bias:    Tensor = torch.rand(1,)                          # (1,)
label_y:      Tensor = raw_features @ true_weight + true_bias  # (4,1)

# ----- 모델 -----
class MLR(nn.Module):
    def __init__(self, input: int, output: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input, output))  # (3,1)
        self.bias   = nn.Parameter(torch.randn(output))         # (1,)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight + self.bias

model = MLR(3, 1)

# ----- 손실/옵티마 -----
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)  

# ----- 학습 루프 -----
for epoch in range(10000):
    pred: Tensor = model(raw_features)
    loss: Tensor = criterion(pred, label_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"{epoch:4d} | loss = {loss.item():.6e}")

# ----- 결과 -----
print("\n[Learned]")
print("weight:\n", model.weight.T)
print("bias  :", model.bias)

print("\n[Check]")
print("label:\n", label_y.T)
print("pred :\n", model(raw_features).T)

