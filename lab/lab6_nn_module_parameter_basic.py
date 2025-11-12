import torch
from torch import nn, Tensor

class MyLayer(nn.Module):
    """단일 모듈: 직접 Parameter(weight, bias) 등록 예제"""
    def __init__(self, input: int, output: int) -> None:
        super().__init__()
        torch.manual_seed(0)  # 결과 재현용 (선택)
        
        # 학습 가능한 파라미터(weight, bias)를 직접 등록 (requires_grad=True 자동 설정)
        # weight는 입력 수(input)에 따라 3개, bias는 출력 수(output)에 따라 1개 생성
        # 초기값은 정규분포를 따르는 난수로 설정
        self.weights = nn.Parameter(torch.randn(input, output, dtype=torch.float32))
        self.bias = nn.Parameter(torch.randn(output, dtype=torch.float32))

        # 모든 신경망 연산은 float형 텐서를 기반으로 수행해야 함
        # 정수형(int) 텐서는 미분 불가능하므로 requires_grad=True 설정 불가
                
    def forward(self, x: Tensor) -> Tensor:
        # (N, input) @ (input, output) + (output,) 형태의 행렬 곱 연산 수행
        # → 수식 h(x) = xW + b 에 해당
        return x @ self.weights + self.bias

# -------------------------------
# 실습 테스트
# -------------------------------
layer = MyLayer(3, 1)

print("등록된 파라미터:")
for name, param in layer.named_parameters():
    print(f"- {name}: shape={tuple(param.shape)}, requires_grad={param.requires_grad}")
    print(f"  values:\n{param.data}\n")

x = torch.tensor([[1.0, 2.0, 3.0]])
output = layer(x)
print("입력:", x)
print("출력:", output)


