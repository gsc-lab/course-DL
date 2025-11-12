import torch
from torch import nn


# -------------------------------
# (1) Forward Pre Hook 정의
# -------------------------------
# forward() 함수가 실행되기 "직전"에 호출됨
# module: 현재 실행 중인 모듈 객체
# input : forward에 전달될 입력값 (tuple 형태)
def pre_hook(module, input):
    print(f"invoked pre-hook: {type(module)}, input: {input}")


# -------------------------------
# (2) Forward Post Hook 정의
# -------------------------------
# forward() 함수가 실행된 "직후"에 호출됨
# module: 현재 실행 중인 모듈 객체
# input : forward()에 전달된 입력값
# output: forward()가 반환한 결과
def post_hook(module, input, output):
    print(f"invoked post-hook: {type(module)}")
    print(f"\tinput: {input}\n\toutput: {output}")


# -------------------------------
# (3) 사용자 정의 Module 클래스
# -------------------------------
# nn.Module을 상속받아 forward 로직을 직접 정의
class MyModule(nn.Module):
    def __init__(self):
        super().__init__()   # 부모 클래스 초기화 필수
    
    def forward(self, x): # 각 모듈별 순전파 구현 내용 오버라이딩
        print("Forward is invoked")   
        return x                      

# -------------------------------
# (4) 모듈 객체 생성 및 Hook 등록
# -------------------------------
module = MyModule()

# forward() 실행 전후에 hook 함수 연결
module.register_forward_pre_hook(pre_hook)   # pre_hook 등록
module.register_forward_hook(post_hook)      # post_hook 등록

# -------------------------------
# (5) 입력 텐서 준비 및 실행
# -------------------------------
input = torch.arange(1, 11)   # tensor([1, 2, 3, ..., 10])

module(input) # 매직메서드 __call__ 호출

