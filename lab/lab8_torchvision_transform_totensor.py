from torch import Tensor
from torchvision import datasets, transforms
import numpy as np

# -----------------------------------------------------------
#   ToTensor()의 역할:
#   PIL.Image 또는 ndarray 이미지를 Tensor로 변환하고,
#   픽셀값 범위를 [0, 255] → [0.0, 1.0] 실수(float32)로 정규화함
# -----------------------------------------------------------

# 변환 정의 (PIL → Tensor)
to_tensor = transforms.Compose([
    transforms.ToTensor()
])

# 변환이 적용된 데이터셋 (Tensor 형태로 변환)
mnist_tensor = datasets.MNIST( root="data", train=True, download=True,
    transform=to_tensor # PIL → Tensor 변환
)

# 원본 데이터셋 (PIL.Image 형태)
mnist_raw = datasets.MNIST( root="data", train=True, download=True)

# 첫 번째 샘플 비교
img_tensor: Tensor
img_tensor, label_tensor = mnist_tensor[0]   # 변환 후 (Tensor)
img_pil, label_raw = mnist_raw[0]            # 변환 전 (PIL.Image)

# Tensor 형태 출력 (정규화된 실수값)
print("Tensor 변환 후 (0~1 범위):")
print(img_tensor.reshape(-1, ))  # Tensor는 실수(float32), 범위는 0~1

# PIL.Image → numpy 배열로 변환 후 출력 (정수값)
print("\n원본 PIL.Image (0~255 범위):")
print(np.array(img_pil))  # uint8, 픽셀값은 0~255

