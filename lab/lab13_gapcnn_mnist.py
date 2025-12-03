import torch
import torch.nn.functional as F
from torch import nn, Tensor, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(
    root="data",
    train=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
    )

test_dataloader = DataLoader(
    test_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=4
    )

class GAPCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 28→14

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),      # 14→7
        )

        # 마지막 conv를 10채널로 만들어 각 채널을 "클래스 스코어 맵"처럼 사용
        self.head = nn.Conv2d(128, 10, kernel_size=1)  # 1x1 conv

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)        # (N, 128, 7, 7)
        x = self.head(x)            # (N, 10, 7, 7)

        # Global Average Pooling: 공간 평균 → (N, 10)
        x = x.mean(dim=(2, 3))
        return x   # CrossEntropyLoss에 logits로 바로 사용




model = GAPCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 18


for epoch in range(1, num_epochs + 1):
    model.train()
    running_loss = 0.0
    loss:Tensor
    X:Tensor
    labels:Tensor
    
    for X, labels in train_dataloader:
        X = X.to(device)
        labels = labels.to(device)
        
        logits = model(X)
        loss = criterion(logits, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss +=  loss.item() * X.size(0)
        
    
    print(f"{epoch} th loss : {(running_loss / len(train_dataset)):.6f}")
        
        
model.to(torch.device("cpu"))
img = torch.randn(1, 1, 28, 28, dtype=torch.float32)

torch.onnx.export(
    model,           # 변환할 PyTorch 모델 (nn.Module)
    img,             # 예시 입력(dummy input)
                     # ONNX는 정적 그래프라, 입력 shape을 알기 위해 실제 forward를 한 번 실행해야 함
                     # img의 dtype/shape을 그대로 ONNX 입력 스펙으로 사용함

    "models/mnist_gap.onnx", # 생성될 ONNX 파일 경로

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
