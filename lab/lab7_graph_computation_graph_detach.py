import torch

a = torch.tensor(2.0, requires_grad=True)

# detach(): a ** 2의 계산 결과를 'autograd 그래프에서 분리'하여 새 Tensor 반환
# 내부 데이터는 복사되지 않고 동일한 메모리를 참조 (shared storage)
# 단, 반환된 Tensor는 autograd 추적 대상이 아님 (requires_grad=False)
# 즉, 이후 연산은 그래프에 포함되지 않음
b = (a ** 2).detach()

c = b ** 2

# c.backward()  #  Error: c는 autograd 그래프에 연결되지 않았기 때문

print("a.requires_grad:", a.requires_grad) # true
print("b.requires_grad:", b.requires_grad) # false
print("c.requires_grad:", c.requires_grad) # false


from torch import nn

criterion = nn.MSELoss()
output, target = 0, 0
running_loss = torch.tensor()

# 사용 예
# loss.item() 같은 값을 로그에 쓰거나,
# 통계 계산에 이용할 때 불필요한 그래프가 생성
# 즉 detach()로 그래프 추적을 끊어서 메모리 낭비, 그래프 확장 방지.
loss = criterion(output, target)
running_loss += loss.detach()   # loss.item() 도 동일한 의미

