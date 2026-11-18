# Transformer의 구조와 LLM이 답변을 생성하는 원리
# ① Tokenization → ② Embedding → ③ Attention → ④ Transformer → ⑤ Next Token Prediction
# 
# 간단한 숫자 벡터를 이용해 Attention 계산 과정을 구현하겠습니다.
# NumPy를 이용해 Q와 K의 유사도를 계산하고, Softmax를 적용하고, 최종 Attention 결과까지 출력하는 실습입니다.
# 그 후에 실제 Gemma 3 모델이 이 계산을 어떻게 활용하는지 연결해서 설명하겠습니다.
# 오늘의 첫 번째 목표는 'LLM이 문맥을 이해하는 데 Attention이 어떤 역할을 하는지'를 Python 코드로 확인하는 것입니다.

# 학습 목표: Q, K, V → 유사도 계산 → Softmax → Attention 결과
# 요소	이름	역할
# Q	Query	어떤 정보를 찾을 것인가
# K	Key	어떤 정보와 관련성이 있는가
# V	Value	실제 전달할 정보는 무엇인가
# 이번에는 실행 결과를 이해하기 쉽도록 3개 토큰과 2차원 벡터만 사용합니다.
import numpy as np

# -------------------------------------------------
# 1. 입력 토큰과 임베딩 벡터
# -------------------------------------------------
tokens = ["나는", "사과를", "먹었다"]

# 학습용 가상 (입력) 임베딩 (실제 Gemma의 값이 아님)
X = np.array([
    [1.0, 0.0],  # 나는
    [0.0, 1.0],  # 사과를
    [1.0, 1.0]   # 먹었다
])

# -------------------------------------------------
# 2. Q, K, V 생성
# -------------------------------------------------
# 실제 모델에서는 학습된 가중치 행렬을 사용
# 여기서는 계산 원리를 보기 위해 단위행렬 사용
W_Q = np.eye(2)
W_K = np.eye(2)
W_V = np.eye(2)

# @는 Python의 행렬 곱셈 연산자
Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# -------------------------------------------------
# 3. Attention Score 계산
# -------------------------------------------------
d_k = K.shape[1]

# K.T는 행렬을 전치(Transpose)하는 연산입니다.
# Q @ K.T를 계산하면 각 토큰의 Query와 모든 토큰의 Key 사이의 내적이 구해집니다.
# 이 값을 \(\sqrt{2}\)로 나눔 : 벡터 차원이 커질 때 내적 값이 지나치게 커지는 현상을 완화하기 위해서입니다.
scores = (Q @ K.T) / np.sqrt(d_k)

# -------------------------------------------------
# 4. Softmax 함수 : Softmax는 각 행의 점수를 0~1 사이의 가중치로 바꾸며, 각 행의 가중치 합계는 1이 됩니다.
# -------------------------------------------------
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

weights = softmax(scores)

# -------------------------------------------------
# 5. Attention 결과 계산
# -------------------------------------------------
output = weights @ V

# -------------------------------------------------
# 6. 결과 출력
# -------------------------------------------------
np.set_printoptions(precision=3, suppress=True)

print("========== 입력 임베딩 ==========")
print(X)

print("\n========== Q ==========")
print(Q)

print("\n========== K ==========")
print(K)

print("\n========== V ==========")
print(V)

print("\n========== Attention Score ==========")
print(scores)

print("\n========== Attention Weights ==========")
print(weights)

print("\n========== Attention Output ==========")
print(output)

print("\n========== 토큰별 Attention 비중 ==========")

for i, token in enumerate(tokens):
    print(f"\n[{token}]이 참고하는 토큰")
    for j, target in enumerate(tokens):
        print(f"  {target:<6}: {weights[i, j]:.2%}")

