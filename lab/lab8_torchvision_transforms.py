from torchvision import transforms
# transform: 이미지 데이터를 신경망에 입력하기 전에
# 일관된 크기와 분포로 변환하고, 데이터 다양성을 높이기 위한 전처리 과정 정의

transform = transforms.Compose([
    transforms.Resize((224, 224)),          # 모든 이미지를 224x224 크기로 맞춤 (모델 입력 크기 통일)
    transforms.RandomHorizontalFlip(),      # 50% 확률로 이미지를 좌우 반전 (데이터 증강, 과적합 방지)
    transforms.ToTensor(),                  # 이미지를 Tensor로 변환 (픽셀값 0~255 → 0~1 실수 범위)
    transforms.Normalize(                   # 정규화: 평균과 표준편차 기준으로 픽셀값 분포 조정
        mean=[0.485, 0.456, 0.406],         # 채널별 평균값 (RGB 기준, ImageNet 통계값)
        std=[0.229, 0.224, 0.225]           # 채널별 표준편차 (RGB 기준)
    )
])
