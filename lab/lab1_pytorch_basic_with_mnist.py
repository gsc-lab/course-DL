# mnist_mlp.py
# PyTorch 기본 구조를 보여주는 MNIST 분류 예제

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# -----------------------------
# 1. 데이터셋 & 데이터로더
# -----------------------------
transform = transforms.ToTensor()  
# - PIL/ndarray 이미지를 torch.FloatTensor로 변환하고 [0,1] 범위로 스케일
# - 출력 텐서 형식은 항상 채널-우선 [C, H, W]
#   * RGB 등 컬러: (H, W, C) -> (C, H, W)
#   * 그레이스케일(MNIST): (H, W) -> (1, H, W)  # 채널 차원 추가

train_dataset = datasets.MNIST(
    root="data",        # 데이터 저장(또는 존재) 경로의 루트 폴더
    train=True,         # 학습용(60,000장). 훈련용으로만 사용
    download=True,      # 데이터가 없으면 자동 다운로드
    transform=transform # 입력 전처리(이미지→텐서)
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,        # 테스트용(10,000장). 최종 성능 평가에만 사용(학습/튜닝 금지)
    download=True,
    transform=transform
)

# Dataset 주요 속성(참고):
# - .data : 이미지 텐서 (uint8, [N,28,28])
# - .targets : 정답 라벨 텐서 (int64, [N]) — 각 값은 0~9 클래스 인덱스
# - .transform / .target_transform : 입력/라벨 변환 함수(옵션)

# 예시:
# data, target = train_dataset[0]
# print(data.shape, target)  # data: torch.Size([1,28,28]), target: 5

train_loader = DataLoader(
    train_dataset,
    batch_size=64,   # 미니배치 크기(한 step에서 처리할 샘플 수)
    shuffle=True     # 각 epoch마다 학습 데이터 순서를 섞음
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1000, # 평가를 빠르게 하기 위해 큰 배치 사용 가능
    shuffle=False    # 평가 단계에서는 보통 셔플 불필요
)
# DataLoader 동작 요약:
# - 1 step(iteration) = 배치 1개 처리
# - 1 epoch = 전체 데이터 1회 처리
# 사용 예:
# images, labels = next(iter(train_loader))

# -----------------------------
# 2. 모델 정의
# -----------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 128)  # 입력 784 → 은닉 128
        self.fc2 = nn.Linear(128, 10)     # 은닉 128 → 출력 10(클래스 수)

    def forward(self, x):
        x = x.view(-1, 28*28)             # (N,1,28,28) → (N,784)
        x = torch.relu(self.fc1(x))       # 활성화 함수(ReLU)
        x = self.fc2(x)                   # 로짓(logits) 출력(softmax 미적용)
        return x

model = MLP()

# -----------------------------
# 3. 손실 함수 & Optimizer
# -----------------------------
criterion = nn.CrossEntropyLoss()  
# - 멀티클래스 분류용 손실: 내부적으로 log_softmax + NLLLoss 결합
# - 입력은 softmax 전의 로짓(logits)이어야 함(softmax를 따로 적용 X)

optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,          # 학습률(업데이트 스텝 크기)
    momentum=0.9      # 관성항(지그재그 완화/수렴 가속)
)

#-----------------------------
# 4. 학습 루프
#-----------------------------
for epoch in range(1, 6):  # 총 5 에폭 학습
    model.train()  # 학습 모드(예: Dropout on, BatchNorm은 배치 통계 사용)
    for images, labels in train_loader:
        # 순전파
        outputs = model(images)           # 예측값 계산
        loss = criterion(outputs, labels) # 손실 계산(정답 라벨은 클래스 인덱스형)

        # 역전파
        optimizer.zero_grad()  # 이전 step에서 누적된 모든 파라미터의 .grad를 0으로 초기화
        loss.backward()        # 현재 배치에 대한 기울기(.grad) 계산
        optimizer.step()       # 설정된 최적화 규칙(SGD+momentum)으로 파라미터 업데이트

    print(f"Epoch [{epoch}/5], Loss: {loss.item():.4f}")  # 마지막 배치의 손실값(에폭 평균 아님)

# -----------------------------
# 5. 평가
# -----------------------------
model.eval()              # 평가 모드(Dropout off, BatchNorm은 러닝 통계 사용)
correct, total = 0, 0

# 평가 단계: 기울기/계산 그래프 불필요 → 메모리/속도 절약
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)           # 추론(로짓)
        print(outputs)                    # (디버그용 출력 — 실제 제출/리포트 시 보통 제거)
        _, predicted = torch.max(outputs, 1)  # 클래스 축에서 최대값의 인덱스 = 예측 클래스
        total += labels.size(0)               # 누적 샘플 수
        correct += (predicted == labels).sum().item()  # 누적 정답 수

print(f"Test Accuracy: {100 * correct / total:.2f}%")  # 최종 정확도(%)
