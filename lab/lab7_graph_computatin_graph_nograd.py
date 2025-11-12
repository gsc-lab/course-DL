import torch
from torch import Tensor

# (1) 추론(inference) 단계: autograd 비활성화
# 모델 평가나 예측 시에는 미분이 필요 없으므로
# torch.no_grad() 블록을 사용하면 메모리 사용량과 연산 속도가 개선됨
with torch.no_grad():
    a = torch.tensor(2.0, requires_grad=True)
    b = a**2
    c = b**3
    print(f"[inference] c.grad_fn: {c.grad_fn}")  # None (그래프 생성 안 됨)
    # c.backward()  # Error: autograd가 비활성화되어 있음

# (2) 학습(training) 단계: autograd 활성화
a = torch.tensor(2.0, requires_grad=True)
b = a**2
c = b**3
print(f"[training] c.grad_fn: {c.grad_fn}")  # <PowBackward0>
c.backward()
print(f"a.grad: {a.grad:.2f}")

