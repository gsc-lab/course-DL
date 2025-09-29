import torch

x16  = torch.tensor(1.0, dtype=torch.float16)
xb16 = torch.tensor(2.0, dtype=torch.bfloat16)
x32  = torch.tensor(3.0, dtype=torch.float32)
x64  = torch.tensor(4.0, dtype=torch.float64)

# PyTorch의 암묵적 형 변환(타입 프로모션) 규칙 요약
#   - 같은 '부동소수' 계열끼리는 더 "넓은/안전한" 쪽으로 승격
#   - 서로 다른 16비트 부동소수(float16 vs bfloat16)를 
#     섞으면 공통 안전 타입인 float32로 승격
#   - float16 + float32 → float32
#   - float32 + float64 → float64

# float16 + bfloat16  → float32 로 승격
result_1 = x16 + xb16

# float16 + float32   → float32 로 승격
result_2 = x16 + x32

# float32 + float64   →  float64 로 승격
result_3 = x32 + x64

print(f"result_1 : {result_1.dtype}")  # torch.float32
print(f"result_2 : {result_2.dtype}")  # torch.float32
print(f"result_3 : {result_3.dtype}")  # torch.float64

