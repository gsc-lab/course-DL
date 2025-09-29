import torch

# 2x2x1 텐서 생성
# - 3차원 텐서 (앞차원=2, 행=2, 열=1)
x = torch.tensor([[[1], [2]], [[3], [4]]])

# Storage 확인
# - 실제 값은 항상 1차원 배열 형태로 메모리에 저장됨
print("Raw 데이터(Storage):", x.storage())

# Tensor 출력
# - shape 정보를 이용해 다차원 배열처럼 표시됨
print("Tensor 데이터:", x)

# 차원 수 (ndim)
# - 현재 텐서는 3차원
print("차원 수:", x.ndim)       # 3

# 모양 (shape)
# - (2, 2, 1) → 2개의 블록, 각 블록은 2행×1열
print("모양(shape):", x.shape)  # torch.Size([2, 2, 1])

# 자료형 (dtype)
print("자료형(dtype):", x.dtype) # torch.int64 (정수형 기본)

