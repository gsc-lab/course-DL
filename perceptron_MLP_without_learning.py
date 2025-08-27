import numpy as np


# 계단 함수 (활성화 함수)
def step(z):
    return (z >= 0).astype(int)


# --------------------------------------
# 1. 신경망 파라미터 (고정, 학습 없음)
# 은닉층 뉴런 2개 (h1, h2)
# h1 = step(x1 + x2 - 0.5)
# h2 = step(x1 + x2 - 1.5)
W_hidden = np.array(
    [[1.0, 1.0], [1.0, 1.0]]  # h1의 가중치 (x1, x2)
)  # h2의 가중치 (x1, x2)
b_hidden = np.array([-0.5, -1.5])  # h1, h2의 편향

# 출력층 뉴런 1개 (y)
# y = step(1*h1 + (-1)*h2 - 0.5)
W_out = np.array([1.0, -1.0])  # (h1, h2) → y 가중치
b_out = -0.5  # 출력층 편향

# --------------------------------------
# 2. XOR 입력 (진리표)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

# --------------------------------------
# 3. 순전파 계산 (Forward Pass)
# 은닉층 계산: (입력 X @ 은닉 가중치.T) + 편향
hidden_input = X @ W_hidden.T + b_hidden
h = step(hidden_input)

# 출력층 계산: (은닉층 h @ 출력 가중치) + 편향
output_input = h @ W_out + b_out
y_hat = step(output_input)

# --------------------------------------
# 4. 결과 출력
print("입력 X:\n", X)
print("은닉층 출력 h:\n", h)
print("최종 출력 y_hat (XOR):\n", y_hat)
