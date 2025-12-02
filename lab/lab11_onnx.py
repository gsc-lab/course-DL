import torch
from torch import Tensor
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# -----------------------------
# 1. ONNX 변환용 입력 생성
# -----------------------------
# transforms.ToTensor() → MNIST 이미지를 [1, 28, 28] 형태로 변환
transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.MNIST(
    root="data",
    train=False,
    transform=transform
)

img: Tensor
img, _ = dataset[0]              # img shape: [1, 28, 28]
img = img.unsqueeze(0)           # ONNX 모델 입력 형태 생성을 위해 batch dimension 추가
                                 # 최종 shape: [1, 1, 28, 28]
                                 # ONNX에서는 보통 [N, C, H, W] 형식을 사용하므로 꼭 필요
                                 
# -----------------------------
# 2. 학습된 PyTorch 모델 로드
# -----------------------------
# weights_only=False → 모델 구조와 weight 모두 로드
# ※ ONNX export 시에는 반드시 model.eval() 필요 (BatchNorm/Dropout 안정화)
model = torch.load("models/mnist.pth", weights_only=False)
model.eval()                     # ONNX 변환에서 필수: 학습 모드에서 export되면 잘못된 그래프 생성됨

img = torch.randn(1, 1, 28, 28, dtype=torch.float32)

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

    # ONNX 연산자 규격 버전: TensorRT/ORT과의 호환성이 좋은 버전 opset=11
    opset_version=11,

    # dynamic_axes: 입력/출력 중 특정 축을 "동적 크기"로 선언
    # 아래 설정 → image 텐서의 0번 축(batch_size)은 변할 수 있다
    # 즉 batch=1 이미지도, batch=32 이미지도 같은 ONNX 모델로 추론 가능
    dynamic_axes={
        "image": {0: "batch_size"},
        # 출력도 batch 축이 존재하면 아래처럼 설정 가능
        # "logits": {0: "batch_size"},
    }
)


