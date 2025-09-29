from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, paths, labels, transform=None, target_transform=None):
        self.paths = paths              # 원본은 파일 경로, PIL, NumPy 등 자유
        self.labels = labels            # 정수/문자열 라벨 등
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # 1) 원본 로드
        img = Image.open(self.paths[idx]).convert("RGB")
        y = self.labels[idx]

        # 2) 입력 전처리 → Tensor
        if self.transform:
            x = self.transform(img)
        else:
            # 예: NumPy로 변환 후 텐서화 (HWC→CHW)
            x = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0

        # 3) 라벨 전처리 → Tensor(또는 모델이 기대하는 형식)
        if self.target_transform:
            y = self.target_transform(y)
        else:
            y = torch.tensor(y) if not torch.is_tensor(y) else y

        return x, y
