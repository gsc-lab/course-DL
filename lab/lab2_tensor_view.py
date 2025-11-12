import torch

# 원본 1D 텐서 (기본 View). 
origin = torch.Tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

# Storage: 실제 값이 1차원으로 일렬 저장된 버퍼
print(origin.storage())        # [1., 2., 3., 4., 5., 6.]

# 기본 메타데이터: 모양/차원수/자료형
print(origin.shape)            # torch.Size([6])
print(origin.ndim)             # 1
print(origin.dtype)            # torch.float32

# === 같은 storage를 공유하는 여러 view 만들기 ===
m2_x = origin.view(2, -1)      # 2 x 3
print(m2_x)
print("m2_x shape:", m2_x.shape)
print("m2_x ndim:", m2_x.ndim)

m3_x = origin.view(3, -1)      # 3 x 2
print(m3_x)
print("m3_x shape:", m3_x.shape)
print("m3_x ndim:", m3_x.ndim)

m1_2_x = origin.view(2, 1, -1) # 2 x 1 x 3
print(m1_2_x)
print("m1_2_x shape:", m1_2_x.shape)
print("m1_2_x ndim:", m1_2_x.ndim)

# === storage 공유 확인 (모두 같은 버퍼를 가리킴) ===
print("ptr(origin) :", origin.storage().data_ptr())
print("ptr(m2_x)   :", m2_x.storage().data_ptr())
print("ptr(m3_x)   :", m3_x.storage().data_ptr())
print("ptr(m1_2_x) :", m1_2_x.storage().data_ptr())
# 위 포인터가 모두 같으면 'view = zero-copy'가 직관적으로 증명됨

# === view에서 값 변경 → 모든 view/원본에 반영 ===
m1_2_x[0, 0, 0] = 100   
print("after edit, m2_x:\n", m2_x)
print("after edit, m3_x:\n", m3_x)
print("after edit, origin:\n", origin)

# Tip:
# - transpose/permute 후에는 텐서가 non-contiguous일 수 있음
# - 그 상태에서 view() 사용 시 RuntimeError → .contiguous().view(...) 사용
