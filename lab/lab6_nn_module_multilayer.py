import torch
from torch import nn, Tensor, optim

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)  # 재현성 확보 (항상 동일한 초기 weight 사용)

# -------------------------------------------------------
# [Sub-Module ①] Linear 연산을 수행하는 사용자 정의 Layer
# nn.Module을 상속받아 내부에 학습 가능한 Parameter(weight, bias) 등록
# -------------------------------------------------------
class MyLayer(nn.Module):
    def __init__(self, input: int, output: int) -> None:
        super().__init__()
        # nn.Parameter()로 선언하면 model.parameters()에 자동 등록됨
        self.weight = nn.Parameter(torch.randn(input, output))
        self.bias   = nn.Parameter(torch.randn(output))
    def forward(self, x: Tensor) -> Tensor:
        # 선형 변환: y = xW + b
        return x @ self.weight + self.bias

# -------------------------------------------------------
# [Sub-Module ②] 활성화 함수 모듈
# -------------------------------------------------------
class SigmoidFunction(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        # 직접 구현한 sigmoid 함수 (torch.sigmoid()와 동일)
        return 1.0 / (1.0 + torch.exp(-x))

# -------------------------------------------------------
# [상위 Module] XOR 문제 해결용 모델 (2 → 2 → 1 MLP)
# 내부에 Linear Layer와 Activation Function을 Sub-Module로 포함
# -------------------------------------------------------
class MyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # Sub-Modules 등록
        self.fc1 = MyLayer(2, 2)          # Sub-Module (Layer 1)
        self.act = SigmoidFunction()      # Sub-Module (Activation)
        self.fc2 = MyLayer(2, 1)          # Sub-Module (Layer 2)

    def forward(self, x: Tensor) -> Tensor:
        # 입력 → fc1 → act → fc2 → act 순서로 계층적 forward 호출
        # 각 단계에서 Sub-Module의 forward()가 내부적으로 실행됨
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x

# -------------------------------------------------------
# 학습 데이터: XOR 진리표
# -------------------------------------------------------
x = torch.tensor([[0., 0.],
                  [0., 1.],
                  [1., 0.],
                  [1., 1.]])
y = torch.tensor([[0.],
                  [1.],
                  [1.],
                  [0.]])

# -------------------------------------------------------
# 손실 함수 및 옵티마이저 정의
# -------------------------------------------------------
model = MyModel()
criterion = nn.BCELoss()                   # 출력이 sigmoid일 때 사용
optimizer = optim.SGD(model.parameters(), lr=0.1)

# -------------------------------------------------------
# Sub-Module 트리 구조 출력
# model.named_modules()는 모든 계층(Module, Sub-Module)을 트리 형태로 반환
# -------------------------------------------------------
print("Sub-Modules:")
for name, m in model.named_modules():
    print(" ", name, "->", type(m).__name__)

# -------------------------------------------------------
# 학습 루프: forward → loss 계산 → backward → optimizer.step()
# -------------------------------------------------------
for epoch in range(5000):
    pred = model(x)          # forward() 호출 시 내부 Sub-Module들도 자동 호출
    loss = criterion(pred, y)

    optimizer.zero_grad()    # 기울기 초기화
    loss.backward()          # 역전파(backpropagation)
    optimizer.step()         # 파라미터(weight, bias) 갱신

    if epoch % 500 == 0:
        print(f"{epoch:4d} | loss: {loss.item():.6f}")

# -------------------------------------------------------
# 학습 결과 출력 (0.5 기준으로 0/1 분류)
# -------------------------------------------------------
with torch.no_grad():
    pred = model(x)
    yhat = (pred >= 0.5).to(y.dtype)
    print("\nPredictions:\n", pred)
    print("Rounded:\n", yhat)
    print("Target:\n", y)
