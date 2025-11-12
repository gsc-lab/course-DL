import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image

# === 이미지 전처리(transform) 정의 ===
# - Resize: 모든 이미지를 (100, 50) 크기로 변환
# - Grayscale: 흑백으로 변환 (채널 수 1)
# - ToTensor: [0,255] 픽셀값을 [0,1] 범위로 변환 + Tensor로 변환
transform = transforms.Compose([
    transforms.Resize((100, 50)),
    transforms.Grayscale(),
    transforms.ToTensor()
])


# === 사용자 정의 데이터셋 클래스 ===
class CarDataset(Dataset):

    # 초기화 메소드
    # dir : 이미지가 저장된 디렉토리 경로
    # transform : 사전에 정의한 전처리(transform)
    # img_type : 확장자 (png, jpg 등)
    def __init__(self, dir="", transform=None, img_type="png"):
        self.dir_path = dir
        self.transform = transform
        self.img_type = img_type
        
        # 디렉토리 내 모든 이미지 파일 리스트 생성
        self.img_list = [f for f in os.listdir(self.dir_path) if f.endswith(f".{self.img_type}")]

        # 파일명에서 각도(angle) 레이블 추출
        # 예: "car_30.png" → label = 30
        self.angle = [int(f.rsplit(".")[0].split("_")[1]) for f in self.img_list]
        
    # 전체 데이터셋 크기 반환
    def __len__(self):
        return len(self.img_list)

    # 특정 인덱스의 (이미지, 라벨) 반환
    def __getitem__(self, index):

        # 파일 경로 조합
        file = os.path.join(self.dir_path, self.img_list[index])
        
        # 이미지 로드 및 RGB 변환
        img = Image.open(file).convert("RGB")

        # 전처리(transform)가 정의되어 있다면 적용
        if self.transform:
            img = self.transform(img)

        # (이미지 Tensor, 각도 레이블) 반환
        return img, torch.tensor(self.angle[index], dtype=torch.long)


# === 데이터셋 객체 생성 ===
obj = CarDataset("./data/autodrive", transform)

# 첫 번째 샘플 확인
img, label = obj[0]

print(img)    # 전처리된 이미지 Tensor
print(label)  # 각도 레이블 (정수)

# img = Image.open("./data/autodrive/1_30.png").convert("RGB")


# print(img.format)
# print(img.size)
# print(img.mode)

# x = transform(img)

# print(x.shape)

