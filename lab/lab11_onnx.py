import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        # --- Feature extractor ---
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)   # 28x28
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 28x28
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # 14x14 → 14x14
        self.bn3   = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)

        # 28x28 --conv1,2--> 28x28 --pool--> 14x14
        # 14x14 --conv3--> 14x14 --pool--> 7x7
        # 채널: 128, 크기: 7x7 → 128 * 7 * 7
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))   # (N, 32, 28, 28)
        x = F.relu(self.bn2(self.conv2(x)))   # (N, 64, 28, 28)
        x = self.pool(x)                      # (N, 64, 14, 14)

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))   # (N, 128, 14, 14)
        x = self.pool(x)                      # (N, 128, 7, 7)

        x = self.classifier(x)                # (N, 10)
        return x



# -----------------------------
# 1. ONNX 변환용 입력 생성
# -----------------------------
# transforms.ToTensor() → MNIST 이미지를 [1, 28, 28] 형태로 변환
img = torch.randn(1, 1, 28, 28, dtype=torch.float32)
                          
# -----------------------------
# 2. 학습된 PyTorch 모델 로드
# -----------------------------
# weights_only=False → 모델 구조와 weight 모두 로드
# ※ ONNX export 시에는 반드시 model.eval() 필요 (BatchNorm/Dropout 안정화)
state = torch.load("models/mnist_cnn.pth")
model = CNNModel()
model.load_state_dict(state['model_state'])
model.eval()                     # ONNX 변환에서 필수: 학습 모드에서 export되면 잘못된 그래프 생성됨

# -----------------------------
# 3. PyTorch → ONNX 변환(핵심)
# -----------------------------
torch.onnx.export(
    model,           # 변환할 PyTorch 모델 (nn.Module)
    img,             # 예시 입력(dummy input)
                     # ONNX는 정적 그래프라, 입력 shape을 알기 위해 실제 forward를 한 번 실행해야 함
                     # img의 dtype/shape을 그대로 ONNX 입력 스펙으로 사용함

    "models/mnist.onnx", # 생성될 ONNX 파일 경로

    # ONNX 그래프에서 입력/출력 텐서 이름 지정
    input_names=["image"],
    output_names=["logits"],

    # 상수 폴딩: 그래프 내 상수 연산 미리 계산하여 inference 효율 증가
    do_constant_folding=True,

    # ONNX 연산자 규격 버전: 17
    opset_version=17,

    # dynamic_axes: 입력/출력 중 특정 축을 "동적 크기"로 선언
    # 아래 설정 → image 텐서의 0번 축(batch_size)은 변할 수 있다
    # 즉 batch=1 이미지도, batch=32 이미지도 같은 ONNX 모델로 추론 가능
    dynamic_axes={
        "image": {0: "batch_size"},
        # 출력도 batch 축이 존재하면 아래처럼 설정 가능
        # "logits": {0: "batch_size"},
    }
)


