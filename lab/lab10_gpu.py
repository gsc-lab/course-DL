import torch
from torch import nn, optim, Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time

# reproducibility
torch.manual_seed(42)

# ------------------------------------------------------------
# device 설정
# GPU(CUDA)가 있으면 GPU 사용, 없으면 CPU fallback
# GPU 사용 시: 모델, 입력 텐서 모두 GPU 메모리로 이동해야 계산 가능
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# transform
transform = transforms.Compose([
    transforms.ToTensor(),
])

# dataset (MNIST는 28x28 작은 이미지 → GPU 이득이 매우 작음)
train_dataset = datasets.MNIST(
    root="data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="data", train=False, download=True, transform=transform
)

# ------------------------------------------------------------
# DataLoader
# (작은 batch_size에서는 GPU 이득이 거의 없음)
# ------------------------------------------------------------
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128)

# ------------------------------------------------------------
# 모델 정의
# 이 MLP는 파라미터 개수가 상대적으로 적어 GPU 이득이 제한적임
# ------------------------------------------------------------
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(28 * 28, 30),
    nn.ReLU(),
    nn.Linear(30, 120),
    nn.ReLU(),
    nn.Linear(120, 120),
    nn.ReLU(),
    nn.Linear(120, 120),
    nn.ReLU(),
    nn.Linear(120, 120),
    nn.ReLU(),
    nn.Linear(120, 120),
    nn.ReLU(),
    nn.Linear(120, 10),
).to(device=device)     # ★ 모델 파라미터를 GPU로 이동

# loss & optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# ------------------------------------------------------------
# 학습 루프
# 주의: epoch 1은 GPU warm-up 구간 → CUDA kernel 초기화로 느릴 수 있음
# 실제 시간 비교는 epoch 2부터 측정
# ------------------------------------------------------------
epochs = 10
for epoch in range(1, epochs + 1):

    # GPU 연산 정확한 시간 측정을 위해 warm-up 이후 시간 측정
    if epoch == 2:
        # GPU 연산은 비동기 → 정확한 시간 측정을 위해 synchronize 필요
        if device.type == "cuda":
            torch.cuda.synchronize()   # ★ GPU 연산 대기
        start = time.perf_counter()
        
    train_loss = 0.0
    for inputs, targets in train_loader:
        # ★ 입력/정답 텐서를 GPU 메모리로 이동해야 GPU 계산 가능
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        # forward: GPU에서 행렬 연산 수행
        logits: Tensor = model(inputs)

        # loss 계산 (GPU 텐서 기반 연산)
        loss: Tensor = criterion(logits, targets)

        optimizer.zero_grad()

        # backward: GPU에서 gradient 계산 (행렬 미분 연산)
        loss.backward()

        # optimizer: 파라미터 업데이트 (파라미터가 GPU에 있으므로 GPU 연산)
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    avg_loss = train_loss / len(train_dataset)
    print(f"Epoch {epoch:2d} | loss: {avg_loss:.6f}")

# GPU 비동기성으로 인해 종료 시점에도 synchronize 필요
if device.type == "cuda":
    torch.cuda.synchronize()
end = time.perf_counter()
print(f"train time: {(end - start):.3f}")

# 현재 GPU 메모리 사용량 확인
#print(torch.cuda.memory_summary())

# ------------------------------------------------------------
# 추론(inference)
# GPU 추론도 비동기 → 정확한 시간 측정 시 synchronize 필요
# ------------------------------------------------------------
start = time.perf_counter()
with torch.no_grad():
    correct = 0
    for images, targets in test_loader:
        images = images.to(device)
        targets = targets.to(device)
        
        logits = model(images)         # GPU에서 forward 연산
        preds = torch.argmax(logits, dim=1)
        correct += (preds == targets).sum().item()

    accuracy = correct / len(test_dataset)
    print(f"Test accuracy: {accuracy:.4f}")

# inference 종료 시점도 synchronize 필요
if device.type == "cuda":
    torch.cuda.synchronize()
end = time.perf_counter()
print(f"test time: {(end - start):.3f}")
