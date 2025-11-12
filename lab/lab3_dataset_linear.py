import numpy as np
import torch
from torch.utils.data import Dataset

np.random.seed(0)

# 커스텀 Dataset 클래스 정의
# - torch.utils.data.Dataset을 상속받아 작성
# - 반드시 __len__(), __getitem__() 메서드를 구현해야 함
class MyData(Dataset):

    def __init__(self, features, labels):
        # numpy 배열을 torch.Tensor로 변환
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        # 전체 샘플 개수 반환
        return len(self.features)
    
    def __getitem__(self, index):
        # 주어진 index에 해당하는 (feature, label) 반환
        return self.features[index], self.labels[index]    

# === 학습용 예제 데이터 ===
train_x = np.random.randint(1, 10, (2, 3))  # 입력 데이터 (2개 샘플, 각 3차원 벡터)
train_y = np.random.randint(10, 20, (2,))   # 레이블 데이터 (2개 정답)

print("numpy 원본 데이터")
print(train_x, train_y)

# Dataset 객체 생성
dataset = MyData(train_x, train_y)

# Dataset은 반복 가능(iterable)
# → 각 샘플 (feature, label)을 순서대로 꺼낼 수 있음
print("\nDataset 순회 결과")
for feature, label in dataset:
    print(feature, label)
