import numpy as np
import torch
from torch.utils.data import Dataset

# ----------------------------
# 예시 데이터 생성 (선형 모델: h(x) = 2*x1 + 5*x2 + 0.5)
# ----------------------------
weight_true = np.array([2.0, 5.0])
bias_true = 0.5

raw_x = np.random.randint(1, 100, size=(4, 2))       # 입력 데이터 (샘플 4개, 각 2차원 특징)
raw_y = raw_x @ weight_true + bias_true              # y = Xw + b
raw_y = raw_y.reshape(-1, 1)                         # (4, 1) 형태로 변환 (회귀 안정성 ↑)


# ----------------------------
# Dataset 클래스 상속 구현
# ----------------------------
class MyDataset(Dataset):  # torch.utils.data.Dataset 상속
    def __init__(self, data, label):
        # __init__ :
        # - Dataset 객체가 생성될 때 실행
        # - 원본 numpy 배열을 torch.Tensor(float32)로 변환
        # - 이렇게 해야 모델 학습에 바로 입력 가능
        self.data = torch.from_numpy(data).float()    # 입력 데이터 (N, 2)
        self.label = torch.from_numpy(label).float()  # 라벨 데이터 (N, 1)

    def __len__(self):
        # __len__ :
        # - Dataset의 전체 샘플 수를 반환
        # - DataLoader는 이 값을 이용해
        #   1 epoch에 포함될 step(배치) 개수를 계산한다.
        #   예) 전체 샘플 1000개, batch_size=100
        #       → 1 epoch = 1000 / 100 = 10 step
        return self.data.shape[0]

    def __getitem__(self, index):
        # __getitem__ :
        # - index번째 (입력, 라벨) 샘플을 반환
        # - DataLoader는 내부적으로 여러 index를 선택해
        #   __getitem__을 반복 호출 → 하나의 batch를 구성
        #   예) batch_size=4라면
        #       __getitem__(0), __getitem__(1),
        #       __getitem__(2), __getitem__(3)
        #       → (X_batch, Y_batch) 형태로 묶어서 모델에 전달
        x = self.data[index]     # index번째 입력 데이터 (특징 벡터)
        y = self.label[index]    # index번째 라벨
        return x, y

# ----------------------------
# Dataset 객체 생성 및 확인
# ----------------------------
obj = MyDataset(raw_x, raw_y)

# Dataset은 인덱스로 샘플 접근 가능
print(obj[0])  # 첫 번째 샘플 (x0, y0)
print(obj[1])  # 두 번째 샘플 (x1, y1)

# Dataset 크기 확인
print(obj.data.shape, obj.label.shape)  
# torch.Size([4, 2]) torch.Size([4, 1])
