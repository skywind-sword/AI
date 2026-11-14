import json
import math
from urllib.request import Request, urlopen

# URL = "http://127.0.0.1:8081/v1/embeddings"     # 갤폴드6 llama.cpp에서 EmbeddingGemma 호출
URL = "http://127.0.0.1:11434/api/embed"        # Windows ollama에서 EmbeddingGemma 호출

sentences = [
    "이번 달 카드 결제금액이 얼마인가요?",
    "이번 달 신용카드 청구액을 알려주세요.",
    "신용카드를 분실했어요.",
    "카드 분실 신고는 어떻게 하나요?",
    "오늘 서울 날씨가 어떤가요?"
]

# 문장 유사도 비교용 접두어
prefix = "task: sentence similarity | query: "

payload = {
    # "model": "embeddinggemma",        # 갤폴드6 llama.cpp에서 embeddinggemma 사용
    "model": "embeddinggemma-local",    # Windows ollama에서 embeddinggemaa tkdyd
    "input": [prefix + s for s in sentences]    # Windows(ollama)에서 embeddinggemma 사용
}

req = Request(
    URL,
    data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

with urlopen(req, timeout=180) as response:
    result = json.load(response)

# 입력 문장 순서대로 벡터 정렬
# 겔폴드6 llama.cpp 버전
# data = sorted(result["data"], key=lambda x: x["index"])
# vectors = [item["embedding"] for item in data]

# ollama 버전
vectors = result["embeddings"]

def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

print("벡터 개수:", len(vectors))
print("벡터 차원:", len(vectors[0]))
print("\n기준 문장:", sentences[0])

for i in range(1, len(sentences)):
    score = cosine_similarity(vectors[0], vectors[i])
    print(f"{score:.4f} : {sentences[i]}")
