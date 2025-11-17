import os
import torch
from torch import nn, optim, Tensor
from torch.utils import data

# =========================
# 1. 설정 값 정의
# =========================
torch.manual_seed(0)                 # 재현성 (필수는 아니지만 관례)
CHECKPOINT_DIR = "models"            # 체크포인트 저장 디렉터리
CHECKPOINT_PATH = os.path.join(      # 불러올 체크포인트 파일
    CHECKPOINT_DIR,
    "xor_epoch_200.pth"              # 예: 200 epoch까지 학습된 파일
)
CHECKPOINT_INTERVAL = 200            # 이어서 학습할 때도 동일하게 사용
EPOCHS = 1000                        # 최종 목표 epoch (예: 1000까지 채우기)
LR = 0.1                             # 학습률 (기존과 동일하게)

# =========================
# 2. 데이터셋 및 DataLoader
# =========================
train_dataset = data.TensorDataset(
    torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32),
    torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
)

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=1,
    shuffle=True
)

# =========================
# 3. 모델 정의 (이전과 구조가 완전히 같아야 함)
# =========================
class MyModel(nn.Module):
    """XOR 문제를 풀기 위한 간단한 MLP 모델 (이전 코드와 동일 구조)"""
    def __init__(self) -> None:
        super().__init__()
        self.fcs = nn.Sequential(
            nn.Linear(2, 3),
            nn.Tanh(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: Tensor) -> Tensor:
        return self.fcs(x)

model: MyModel = MyModel()
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=LR)

# =========================
# 4. 체크포인트 로드
# =========================
if not os.path.exists(CHECKPOINT_PATH):
    raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT_PATH}")

# map_location은 CPU/GPU 환경이 바뀌었을 때 안전하게 로드하기 위함 (여기서는 CPU 기준)
checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")

# 저장된 가중치/옵티마이저/메타데이터 복원
model.load_state_dict(checkpoint["model_state"])
optimizer.load_state_dict(checkpoint["optimizer_state"])

start_epoch = checkpoint["epoch"] + 1      # 이어서 시작할 epoch 번호
global_step = checkpoint["global_step"]    # 이전까지의 전체 step 수 (통계용)
print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
print(f"Resuming from epoch {start_epoch}, global_step {global_step}")

# =========================
# 5. 이어서 학습
# =========================
for epoch in range(start_epoch, EPOCHS + 1):
    epoch_loss = 0.0
    
    for step, (X, y) in enumerate(train_dataloader):
        probs: Tensor = model(X)
        loss: Tensor = criterion(probs, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * probs.size(0)
        global_step += 1

    avg_epoch_loss = epoch_loss / len(train_dataset)
    
    if epoch % CHECKPOINT_INTERVAL == 0 or epoch == EPOCHS:
        print(f"{epoch} th epoch - Loss: {avg_epoch_loss:.6f}")
        
        checkpoints = {
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch":           epoch,
            "global_step":     global_step,
            "avg_epoch_loss":  avg_epoch_loss,
        }
        
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"xor_epoch_{epoch}.pth")
        torch.save(checkpoints, ckpt_path)

# =========================
# 6. 학습 후 결과 확인
# =========================
with torch.no_grad():
    X = torch.tensor([[0., 0.],
                      [0., 1.],
                      [1., 0.],
                      [1., 1.]])
    probs = model(X)
    preds = (probs >= 0.5).int()
    print("probs:\n", probs)
    print("preds:\n", preds)
