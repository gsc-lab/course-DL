import os
import torch
from torch import nn, optim, Tensor
from torch.utils import data

# =========================
# 1. 설정 값 정의
# =========================
torch.manual_seed(0)                 # 재현성 확보를 위한 Seed 고정
CHECKPOINT_DIR = "models"            # 체크포인트 저장 디렉터리
CHECKPOINT_INTERVAL = 200            # 몇 epoch마다 체크포인트를 저장할지
EPOCHS = 1000                        # 총 학습 epoch 수
LR = 0.1                             # 학습률

# 체크포인트 저장 폴더 생성 (이미 있으면 무시)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# =========================
# 2. 데이터셋 및 DataLoader
# =========================
train_dataset = data.TensorDataset(
    torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),  # 입력 (XOR)
    torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)               # 정답 (0 또는 1)
)

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=1,       # 샘플 1개 단위로 업데이트 (SGD)
    shuffle=True        # 매 epoch마다 샘플 순서 섞기
)

# =========================
# 3. 모델 정의
# =========================
class MyModel(nn.Module):
    """XOR 문제를 풀기 위한 간단한 MLP 모델"""
    def __init__(self) -> None:
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(2, 3),   # 입력 2 → 은닉층 3
            nn.Tanh(),         # 비선형 활성화
            nn.Linear(3, 1),   # 은닉층 3 → 출력 1
            nn.Sigmoid()       # 출력 값을 [0, 1] 범위의 확률로 변환
        )
        
    def forward(self, x: Tensor) -> Tensor:
        # x: (batch_size, 2) 형태의 입력
        return self.fcs(x)

# 모델, 손실 함수, 옵티마이저 초기화
model: MyModel = MyModel()
criterion = nn.BCELoss()                          # 이진 분류용 Cross Entropy (Sigmoid 출력에 사용)
optimizer = optim.SGD(model.parameters(), lr=LR)  # 확률적 경사하강법

# =========================
# 4. 학습 루프 + 체크포인트 저장
# =========================
global_step = 0  # 전체 학습 스텝 수 (나중에 이어서 학습할 때 사용 가능)

for epoch in range(1, EPOCHS + 1):
    epoch_loss = 0.0
    
    for step, (X, y) in enumerate(train_dataloader):
        # -------------------------
        # (1) 순전파(forward)
        # -------------------------
        probs: Tensor = model(X)           # 출력: Sigmoid를 거친 확률 값
        loss: Tensor = criterion(probs, y) # BCELoss: 예측 확률 vs 정답(0/1)
        
        # -------------------------
        # (2) 역전파(backward)
        # -------------------------
        optimizer.zero_grad() # 이전 step에서 계산된 gradient 초기화
        loss.backward()       # 현재 loss 기준으로 gradient 계산
        optimizer.step()      # gradient를 이용해 파라미터 업데이트
        
        # -------------------------
        # (3) 통계값 누적
        # -------------------------
        epoch_loss += loss.item() * probs.size(0)  # 배치 평균 loss × 배치 크기
        global_step += 1                           # 전체 step 카운트

    # epoch 단위 평균 loss 계산
    avg_epoch_loss = epoch_loss / len(train_dataset)
    
    # 지정한 주기마다 학습 상황 출력 + 체크포인트 저장
    if epoch % CHECKPOINT_INTERVAL == 0:
        print(f"{epoch} th epoch - Loss: {avg_epoch_loss:.6f}")
        
        checkpoints = {
            'model_state':     model.state_dict(),      # 모델 가중치/편향
            'optimizer_state': optimizer.state_dict(),  # 옵티마이저 상태 (모멘텀 등)
            'epoch':           epoch,                   # 현재 epoch 번호
            'global_step':     global_step,             # 전체 step 수
            'avg_epoch_loss':  avg_epoch_loss           # 이 epoch의 평균 loss
        }
        
        # ex) models/xor_epoch_200.pth
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"xor_epoch_{epoch}.pth")
        torch.save(checkpoints, ckpt_path)

# =========================
# 5. 학습 후 결과 확인
# =========================
with torch.no_grad():
    X = torch.tensor([[0., 0.],
                      [0., 1.],
                      [1., 0.],
                      [1., 1.]])
    probs = model(X)                # 각 입력에 대한 예측 확률
    preds = (probs >= 0.5).int()    # 0.5 기준으로 0/1로 변환
    print("probs:\n", probs)
    print("preds:\n", preds)
