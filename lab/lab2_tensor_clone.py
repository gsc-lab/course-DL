import torch

# 원본 1D 텐서 (float32 기본)
x = torch.Tensor([1, 2, 3, 4])

# view: 메모리 복사 없이 shape만 바꾼 텐서 (같은 storage 공유)
y = x.view(2, -1)   # shape: (2, 2)

# clone: 데이터를 새 storage로 복사한 '독립 텐서'
z = x.clone()

# === 복사/공유 판별 1: _base 속성 ===
# - view 텐서는 자신이 파생된 '원본 텐서'를 _base로 가짐
# - 독립 텐서는 _base가 None
print(x._base)      # None (원본 자체는 base가 없음)
print(y._base)      # tensor([1., 2., 3., 4.]) → y는 x의 view
print(z._base)      # None → z는 복사본 (독립 storage)

# y가 x의 view인지 직관적 확인
print(y._base is x) # True


