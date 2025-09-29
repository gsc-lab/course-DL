import torch

# ------------------------------------------------------------
# 0) 환경/버전/디바이스 정보
# ------------------------------------------------------------
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    cur = torch.cuda.current_device()
    print(f"Current device ID: {cur}, name: {torch.cuda.get_device_name(cur)}")
else:
    print("-> GPU 미사용: CPU로 실행합니다.")

# 재현성(데모용): 완전 재현은 CUDNN 설정까지 필요
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# ------------------------------------------------------------
# 1) 텐서의 핵심 속성
#    - 구조(차원): Scalar, Vector, Matrix, N-dimensional Tensor
#    - 자료형(dtype): 텐서 내 모든 원소 동일
#    - 메모리 레이아웃: 정방 격자(regular n-D array)
# ------------------------------------------------------------
x = torch.tensor(((1, 2), (3, 4)))   # 기본 int64 텐서 (정수 리터럴이므로)
print("\n[1] 기본 텐서 속성")
print("x =", x)
print("shape:", x.shape)     # torch.Size([2,2])
print("ndim :", x.ndim)      # 2
print("device:", x.device)   # cpu 또는 cuda:0
print("dtype :", x.dtype)    # torch.int64 (기본)
print("numel :", x.numel())  # 총 원소 수

# float로 만들고 싶으면 dtype 지정 or 부동소수 리터럴 사용
xf = torch.tensor(((1.0, 2.0), (3.0, 4.0)), dtype=torch.float32)
print("xf dtype:", xf.dtype)  # torch.float32

# ------------------------------------------------------------
# 2) 생성 함수 모음
#    - 직접/난수/범위 생성
# ------------------------------------------------------------
print("\n[2] 생성 함수")
y0 = torch.zeros(10)                 # [10], float32 기본
y1 = torch.ones(10)                  # [10]
yf = torch.full((2, 5), 7)           # [2,5], 값 7 (int64)
ru = torch.rand(3, 4)                # [3,4], U(0,1)
rn = torch.randn(2, 3)               # [2,3], N(0,1)
ri = torch.randint(0, 10, (2, 3, 3)) # [2,3,3], 0~9

print("zeros :", y0)
print("ones  :", y1)
print("full  :", yf)
print("rand  :\n", ru)
print("randn :\n", rn)
print("randint:\n", ri)

# ------------------------------------------------------------
# 3) 크기/속성 따라하기: *_like 계열
#    - shape/dtype/device를 그대로 따르고 값만 초기화
# ------------------------------------------------------------
print("\n[3] *_like 계열")
origin = torch.tensor((2, 3), dtype=torch.float32, device="cpu")
zlike  = torch.zeros_like(origin)
olike  = torch.ones_like(origin)
rlike  = torch.rand_like(origin)
rnlike = torch.randn_like(origin)

print(f"zeros_like : {zlike},  dtype: {zlike.dtype},  shape: {zlike.shape}, device: {zlike.device}")
print(f"ones_like  : {olike},  dtype: {olike.dtype},  shape: {olike.shape}, device: {olike.device}")
print(f"rand_like  : {rlike},  dtype: {rlike.dtype},  shape: {rlike.shape}, device: {rlike.device}")
print(f"randn_like : {rnlike}, dtype: {rnlike.dtype}, shape: {rnlike.shape}, device: {rnlike.device}")

# ------------------------------------------------------------
# 4) 인덱싱/슬라이싱/마스킹
# ------------------------------------------------------------
print("\n[4] 인덱싱/슬라이싱/마스킹")
a = torch.arange(1, 13).reshape(3, 4)   # [[1..4],[5..8],[9..12]]
print("a:\n", a)
print("a[0]      :", a[0])       # 첫 행
print("a[:, 1:3] :\n", a[:, 1:3])# 모든 행, 열 1~2
mask = a % 2 == 0
print("mask(짝수):\n", mask)
print("a[mask]   :", a[mask])    # 불리언 인덱싱(벡터로 반환)

# ------------------------------------------------------------
# 5) 브로드캐스팅
#    - 뒤에서부터 차원 비교, 1 또는 동일 크기이면 확장 가능
#    - expand(뷰) vs repeat(복제)
# ------------------------------------------------------------
print("\n[5] 브로드캐스팅")
b = torch.arange(3).reshape(3, 1)    # [3,1]
c = torch.arange(4).reshape(1, 4)    # [1,4]
bc = b + c                           # [3,4]
print("b:\n", b)
print("c:\n", c)
print("b + c:\n", bc)

# ------------------------------------------------------------
# 6) 형태 변환(메모리 포함)
#    - view/reshape: 원소 재배치 없이 모양만 변경 (view는 contiguous 필요)
#    - permute/transpose: 차원 순서 변경( stride 변경 )
#    - squeeze/unsqueeze: 크기 1 차원 제거/추가
#    - cat/stack: 결합(기존 축/새 축)
# ------------------------------------------------------------
print("\n[6] 형태 변환")
t = torch.arange(24).reshape(2, 3, 4)           # [2,3,4]
print("t.shape:", t.shape)
print("view to [6,4]:\n", t.view(6, 4))
print("permute to [3,2,4]: shape", t.permute(1, 0, 2).shape)
u = torch.randn(1, 3, 1, 5)
print("u.shape:", u.shape, "-> squeeze:", u.squeeze().shape, "-> unsqueeze(0):", u.squeeze().unsqueeze(0).shape)
a1 = torch.ones(2, 3)
a2 = torch.zeros(2, 3)
print("cat dim=0 shape:", torch.cat([a1, a2], dim=0).shape)  # [4,3]
print("stack dim=0 shape:", torch.stack([a1, a2], dim=0).shape) # [2,2,3]

# ------------------------------------------------------------
# 7) 연산·축소(reduction)
#    - sum/mean/max/min, dim/keepdim 파라미터
# ------------------------------------------------------------
print("\n[7] 연산/축소")
m = torch.arange(1, 7).reshape(2, 3)  # [[1,2,3],[4,5,6]]
print("m:\n", m)
print("sum all:", m.sum())
print("mean dim=0:", m.mean(dim=0))
print("max dim=1:", m.max(dim=1))

# ------------------------------------------------------------
# 8) dtype 규칙 & 프로모션
#    - 기본 float32, 정수는 int64
#    - 승격: bool < int < float < complex
#    - FP16+BF16 → FP32, FP32+FP64 → FP64
# ------------------------------------------------------------
print("\n[8] dtype 프로모션")
x16  = torch.tensor(1.0, dtype=torch.float16)
xb16 = torch.tensor(2.0, dtype=torch.bfloat16)
x32  = torch.tensor(3.0, dtype=torch.float32)
x64  = torch.tensor(4.0, dtype=torch.float64)
print("x16+xb16 dtype:", (x16 + xb16).dtype)  # → float32
print("x16+x32  dtype:", (x16 + x32).dtype)   # → float32
print("x32+x64  dtype:", (x32 + x64).dtype)   # → float64

# ------------------------------------------------------------
# 9) Autograd 핵심
#    - leaf tensor: 사용자가 직접 만든 requires_grad=True 텐서
#    - loss.backward() → .grad에 누적
#    - in-place 연산 주의(_ 접미사)
#    - no_grad()/detach()로 그래프 추적 차단
# ------------------------------------------------------------
print("\n[9] Autograd")
w = torch.randn(3, requires_grad=True)  # leaf
y = (w ** 2).sum()
y.backward()                            # dy/dw = 2w
print("w.grad:", w.grad)

# 파라미터 업데이트 시에는 no_grad로 추적 중단
lr = 0.1
with torch.no_grad():
    w -= lr * w.grad
print("w updated (no_grad) OK, requires_grad:", w.requires_grad)

# detach: 데이터 공유, 그래프 분리
z = w.detach()
print("detach shares storage:", z.data_ptr() == w.data_ptr())

