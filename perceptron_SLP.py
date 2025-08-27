import numpy as np


# ----------------------------------
# 계단 함수 (Step Function)
# ----------------------------------
def step(z):
    """입력값이 0 이상이면 1, 아니면 0을 출력"""
    return 1 if z >= 0 else 0


# ----------------------------------
# 퍼셉트론 학습 함수
# ----------------------------------
def train(X, y, lr=1.0, epochs=10):
    """
    X : 입력 데이터 (2차원 배열)
    y : 정답 (1차원 배열)
    lr : 학습률
    epochs : 전체 데이터 반복 학습 횟수
    """

    w = np.zeros(X.shape[1])  # 가중치 초기화
    b = 0.0  # 편향 초기화

    for _ in range(epochs):
        for x, y_true in zip(X, y):
            # 1. 가중합 (net input)
            z = x @ w + b
            # 2. 예측값
            y_hat = step(z)
            # 3. 오차 계산 (정답 - 예측)
            error = y_true - y_hat
            # 4. 오차가 있으면 가중치와 편향 업데이트
            w += lr * error * x
            b += lr * error
    return w, b


# ----------------------------------
# 예측 함수
# ----------------------------------
def predict(X, w, b):
    """학습된 w, b로 입력 X에 대한 예측 반환"""
    return [step(np.dot(x, w) + b) for x in X]


# ----------------------------------
# 사용 예시 (XOR 연산 학습)
# ----------------------------------
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # 입력
y = np.array([0, 1, 1, 0])  # 정답 (XOR 결과)

# 퍼셉트론 학습
w, b = train(X, y, lr=1.0, epochs=10)

# 결과 출력
print("학습된 가중치:", w, "편향:", b)
print("예측 결과:", predict(X, w, b))
