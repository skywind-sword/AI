import json
import math
from urllib.request import Request, urlopen

URL = "http://127.0.0.1:11434/api/embed"
MODEL = "embeddinggemma-local"

# 가상의 FDS 업무 문서
documents = [
    {
        "title": "해외 반복 결제 이상거래",
        "text": (
            "해외 결제가 짧은 시간에 반복 발생하면 "
            "이전 거래 이력, 거래 국가, 결제 패턴을 "
            "확인한다. 이상 징후가 있으면 담당자가 "
            "추가 검토한다."
        )
    },
    {
        "title": "카드 분실 신고 절차",
        "text": (
            "카드를 분실한 고객은 카드사에 신고한다. "
            "본인 확인 후 카드 사용정지 및 재발급 "
            "절차를 진행한다."
        )
    },
    {
        "title": "신용카드 청구금액 조회",
        "text": (
            "고객은 이번 달 카드 이용금액, 결제예정금액, "
            "청구금액을 카드사 앱에서 확인할 수 있다."
        )
    },
    {
        "title": "FDS 오탐 처리 절차",
        "text": (
            "이상거래로 탐지되었으나 정상 거래로 "
            "확인된 경우 오탐으로 분류하고 "
            "탐지 규칙의 적절성을 검토한다."
        )
    },
    {
        "title": "이상거래 사고 사례",
        "text": (
            "해외에서 여러 차례 결제가 발생한 후 "
            "고객이 본인 거래가 아니라고 신고한 "
            "사례가 있다. 거래 내역과 고객 확인 "
            "결과를 분석했다."
        )
    }
]

def get_embeddings(texts):
    payload = {
        "model": MODEL,
        "input": texts
    }

    req = Request(
        URL,
        data=json.dumps(
            payload, ensure_ascii=False
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    with urlopen(req, timeout=180) as response:
        result = json.load(response)

    return result["embeddings"]


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)


# 1. 문서 Embedding (색인 생성)
doc_texts = [
    f"title: {doc['title']} | text: {doc['text']}"
    for doc in documents
]

doc_vectors = get_embeddings(doc_texts)

print("문서 개수:", len(doc_vectors))
print("벡터 차원:", len(doc_vectors[0]))

# 2. 사용자 질문 Embedding
question = (
    "해외에서 여러 건의 결제가 연속으로 "
    "발생하면 무엇을 확인해야 하나요?"
)

query_text = f"task: search result | query: {question}"
query_vector = get_embeddings([query_text])[0]

# 3. 모든 문서와 코사인 유사도 비교
results = []

for doc, vector in zip(documents, doc_vectors):
    score = cosine_similarity(query_vector, vector)
    results.append((score, doc))

# 4. 유사도 내림차순 정렬
results.sort(key=lambda item: item[0], reverse=True)

# 5. 가장 유사한 문서 3개 출력
print("\n질문:", question)
print("\n검색 결과 TOP 3")

for rank, (score, doc) in enumerate(results[:3], 1):
    print(f"\n{rank}위 / 유사도: {score:.4f}")
    print("제목:", doc["title"])
    print("내용:", doc["text"])
