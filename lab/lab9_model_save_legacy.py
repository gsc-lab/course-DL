import torch
from torch import nn

# 간단한 신경망 모델 정의
model = nn.Sequential(
            nn.Linear(2, 3),
            nn.ReLU(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )

# ------------------------------
# 1) 모델 전체 직렬화 저장 (비권장)
# ------------------------------
# 이 방식은 모델 객체 전체를 그대로 저장한다
# 즉 클래스 구조, forward 방식 등 모든 정보가 포함됨
# → 코드 구조가 바뀌면 로딩이 불가능해지는 문제가 
#   있으므로 실무에서는 거의 사용하지 않음.
torch.save(model, "my_model.pth")


# ------------------------------
# 2) 모델 전체 로딩
# ------------------------------
# torch.load(..., weights_only=False)
# - weights_only=False는 “모델 전체를 로드한다”는 의미(필수)
# - 즉 state_dict(가중치)만 로드하는 것이 아니라,
#   모델 구조(Sequential, Linear, ReLU 클래스)까지 그대로 복원함.
reload_model: nn.Module = torch.load(
                        "my_model.pth", weights_only=False)

# ------------------------------
# 3) 로딩된 모델의 모듈 구조 출력
# ------------------------------
for module in reload_model.modules():
    print(module)
    
