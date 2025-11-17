import torch
from torch import nn

# 1) 모델 정의
model = nn.Linear(2, 3)

# 모델 외에 함께 저장하고 싶은 추가 정보
# - epoch, step, loss, hyperparameters 등 원하는 모든 값 가능
additional_info = "everything you want"

# 2) 체크포인트 딕셔너리 구성
# state_dict()는 모델의 weight, bias 등 학습된 파라미터만 포함
# ※ 모델 구조(클래스 정의)는 포함되지 않는다
checkpoint = {
    'model_state': model.state_dict(),   # 모델 파라미터 저장 (핵심)
    'additional_info': additional_info   # 기타 정보도 함께 저장 가능
}

# 3) 저장 (권장 방식)
# state_dict 기반 저장 방식이 실무에서 권장되는 이유:
#   - 모델 구조 변경에도 비교적 유연
torch.save(checkpoint, "model_dict.pth")

# 4) 로딩
# 모델 로딩 시에는 동일한 구조의 모델 인스턴스를 먼저 생성해야 한다
# (state_dict는 파라미터만 저장하기 때문에 모델 클래스 코드는 필요함)
model_new = nn.Linear(2, 3)
# 파일에서 checkpoint 딕셔너리를 읽어온다
state_dict = torch.load("model_dict.pth")
# 저장된 파라미터를 새로운 모델에 로딩
model_new.load_state_dict(state_dict['model_state'])
# 추가로 저장했던 정보도 그대로 복원 가능
print(state_dict['additional_info'])

