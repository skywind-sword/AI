# ============================================================
# 프로그램명 : 07.rag_test.py
#
# [학습 목적]
# 1. RAG(Retrieval-Augmented Generation) 전체 흐름 구현
# 2. EmbeddingGemma로 관련 업무 문서 검색
# 3. 검색 결과를 Gemma 3 LLM의 문맥(Context)에 제공
# 4. LLM이 검색 문서를 근거로 답변하는 과정 확인
#
# [RAG 핵심 원리]
# 문서를 LLM에 Fine-tuning하는 것이 아니라
# 질문할 때 필요한 문서를 검색하여 Prompt에 추가한다.
#
# 문서 → Embedding → 검색 → Prompt 구성 → LLM 답변
#
# [성능 개선]
# 문서 Embedding은 최초 한 번만 수행하고
# 결과를 JSON 파일에 저장하여 재사용한다.
#
# 본 예제는 RAG의 최소 구현이며 실제 업무용
# 정확도 검증, 권한 관리 등은 포함하지 않는다.
# ============================================================

import json
import math
import hashlib
from pathlib import Path
from urllib.request import Request, urlopen


# ============================================================
# 1. 실행 환경 설정
# ============================================================

BACKEND = "llama"  # "ollama" 또는 "llama"

if BACKEND == "ollama":
    EMBED_URL = "http://127.0.0.1:11434/api/embed"
    LLM_URL = "http://127.0.0.1:11434/api/generate"
    EMBED_MODEL = "embeddinggemma-local"
    LLM_MODEL = "gemma3-4b-local"

elif BACKEND == "llama":
    EMBED_URL = "http://127.0.0.1:8081/v1/embeddings"
    LLM_URL = "http://127.0.0.1:8080/v1/chat/completions"
    EMBED_MODEL = "embeddinggemma"
    LLM_MODEL = None  # llama.cpp 서버의 모델 ID 조회

else:
    raise ValueError("지원하지 않는 BACKEND입니다.")

# 문서 벡터 저장 파일: 재실행 시 Embedding 생략
CACHE_FILE = Path.home() / ".rag07_vectors.json"

# 검색 결과 중 LLM에 전달할 문서 개수
TOP_K = 2


# ============================================================
# 2. 가상의 FDS 업무 문서
#  - Dictionay의 리스트 (JSON이 아님!!!) : 문법은 JSON과 거의 동일 
# ============================================================
DOCUMENTS = [
    {
        "title": "해외 반복 결제 이상거래",
        "text": (
            "해외 결제가 짧은 시간에 반복 발생하면 "
            "이전 거래 이력, 거래 국가, 결제 패턴을 "
            "확인한다. 이상 징후가 있으면 담당자가 "
            "추가 검토한다."
        ),
    },
    {
        "title": "카드 분실 신고 절차",
        "text": (
            "카드를 분실한 고객은 카드사에 신고한다. "
            "본인 확인 후 카드 사용정지 및 재발급 "
            "절차를 진행한다."
        ),
    },
    {
        "title": "이상거래 사고 사례",
        "text": (
            "해외에서 여러 차례 결제가 발생한 후 "
            "고객이 본인 거래가 아니라고 신고한 "
            "사례가 있다. 거래 내역과 고객 확인 "
            "결과를 분석했다."
        ),
    },
]


# ============================================================
# 3. 공통 HTTP JSON 호출
# ============================================================


def call_api(url, payload=None):
    """REST API를 호출하고 JSON 응답을 반환한다."""

    data = None
    headers = {}

    if payload is not None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = Request(url, data=data, headers=headers)

    with urlopen(req, timeout=300) as response:
        return json.load(response)


# ============================================================
# 4. Embedding 생성
# ============================================================


def get_embeddings(texts):
    """
    [목적]
    텍스트 목록을 768차원 Embedding 벡터로 변환한다.

    문서와 질문 모두 같은 Embedding 모델을 사용한다.
    단, 검색 목적에 맞게 입력 Prefix는 구분한다.
    """

    result = call_api(EMBED_URL, {"model": EMBED_MODEL, "input": texts})

    if BACKEND == "ollama":
        return result["embeddings"]

    data = sorted(result["data"], key=lambda item: item["index"])
    return [item["embedding"] for item in data]


# ============================================================
# 5. 문서 Embedding 캐시
# ============================================================


def get_document_vectors():
    """
    [목적] 캐시파일에서 전체 문서의 임베딩 vector를 읽어옴 (없으면 EmbeddingGemma를 호출해 생성 & JSON 파일 저장)
    """

    # DOCUMENT - 딕셔너리의 리스트(list[dict[str,str]]) 를 
    # ["title: ... | text: ...", "title: ... | text: ...", "title: ... | text: ..."]  형태의 문자열 리스트로 변환
    # python의 List Comprehension 문법 사용
    doc_texts = [f"title: {d['title']} | text: {d['text']}" for d in DOCUMENTS]


    # 1. DOCUMENTS 전체의 캐시 식별값(cache_key) 생성
    #   - [BACKEND, EMBED_MODEL, doc_texts[ → JSON 직렬화(dump) → UTF-8 변환 → SHA-256 해시 생성
    #   [참고] 현재는 DOCUMENTS 전체에 대해 단일 cache_key 사용 (문서 하나만 변경되어도 전체 문서를 다시 Embedding한다)
    #         실제에서는 Chunk 단위 증분 색인이 효율적
    cache_key = hashlib.sha256(
        json.dumps([BACKEND, EMBED_MODEL, doc_texts], ensure_ascii=False).encode(
            "utf-8"
        )
    ).hexdigest()

    # 2. 캐시 파일 존재 여부 및 해시값 일치 확인
    #    - CACHE_FILE.exists()로 파일 존재 여부 확인
    #    - cache["key"]와 cache_key 비교
    #    - 캐시가 유효하면 기존 Vector 반환 (JSON 파일의 "vectors" 값을 반환)
    if CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
  
            if cache.get("key") == cache_key:
                print("저장된 문서 벡터 재사용")
                return cache["vectors"]

        except (OSError, ValueError, KeyError):
            pass  # 캐시 손상 시 새로 생성

    # 3. 캐시가 없거나 유효하지 않으면 새로 생성
    #   1) get_embeddings(doc_texts) : EmbeddingGemma를 호출해 문서별 Vector 생성
    #   2) JSON 파일에 cache_key와 vectors 저장
    #   3) 생성된 vectors 반환
    print("문서 Embedding 생성 중...")
    vectors = get_embeddings(doc_texts)

    CACHE_FILE.write_text(
        json.dumps({"key": cache_key, "vectors": vectors}), encoding="utf-8"
    )

    return vectors


# ============================================================
# 6. Vector Search
# ============================================================


def cosine_similarity(a, b):
    """두 벡터의 방향 유사도를 계산한다."""

    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


def search_documents(question, doc_vectors):
    """
    [목적] 질문의 벡터(임베딩) 값을 구해, 전체 문서내 각 문서의 벡터값과 비교(유사도 계산) -> 정렬 -> TOP_K 만큼의 결과만 리턴
    """

    # 1. 질문의 벡터값(query_vector)를 구함
    # [참고] Emedding Gemma 권장 Prompt Format
    #  - 문서 검색(질문)용 Emedding 벡터 생성 : "task: search result | query: 질문"
    #  - 문서 색인용 Emedding 벡터 생성      : "title: 제목 | text: 본문"
    #  - 유사도 비교용 Emedding 벡터 생성    :  "task: sentense similarity | query: 문장" 
    # [query_text] : 문자열 하나를 리스트에 담아 전달, 마지막 [0]은 반환된 벡터 목록에서 첫번째 벡터를 꺼냄
    query_text = f"task: search result | query: {question}"
    query_vector = get_embeddings([query_text])[0]  

    results = []

    # 2. 루프를 돌면서 위 1.에서 구해진 query_vector와 DOCUMENT의 각 vector 값간의 유사도를 저장
    #   - results에 (유사도, 원본문서) 튜플 append : 타입 : list[tuple[float,dict]]
    #     예) results = [
    #            (0.92, {"title": "해외 반복 결제 이상거래", "text": "해외 결제가 .... "},
    #            (0.45, {"title": "카드 분실 신고 절차", "text": " .... "},
    #            (0.81, {"title": "이상거래 사고 사례", "text": " .... "},
    #        ]
    #   - results.sort(key=lambda x: x[0], reverse=True)
    #      . key= : 정렬 기준 지정, . "lambda x: x[0]"" : 각 튜플의 첫번째 값(유사도)를 반환
    #      . sort() : 기존 리스트(파라미터) 자체를 변경하고, None을 반환. 정렬된 새 리스트를 return 값으로 받고 싶으면 sorted() 사용
    #   - result[:TOP_K] : 0 ~ (TOP_K-1) 까지 반환 (파이썬 슬라이싱 문법)
    #      . 유사도 내림차순 상위 TOP_K개 반환 (최소 유사도와 관계없이 최대 TOP_K개 선택
    for doc, vector in zip(DOCUMENTS, doc_vectors):  # zip() : 원본 문서와 해당 문서의 Emedding 벡터를 순서대로 묶음 (길이가 다르면 짧은쪽까지만 반복)
        score = cosine_similarity(query_vector, vector)
        results.append((score, doc))

    results.sort(key=lambda x: x[0], reverse=True)

    return results[:TOP_K]


# ============================================================
# 7. 검색 문서를 이용한 답변 생성
# ============================================================


def generate_answer(question, search_results):
    """
    [목적]
    검색된 문서를 Prompt에 추가하여 LLM에 전달한다.

    [RAG의 Generation 단계]
    생성형 LLM은 외부 문서를 학습하는 것이 아니라
    현재 입력된 Context로 참고하여 답변을 생성한다.

    검색 문서에 없는 정보를 임의로 만들지 않도록
    답변 지침을 명시하지만, 실제 준수 여부는
    별도로 검증해야 한다.
    """

    # 검색 문서를 번호와 함께 Context로 구성한다.
    context = "\n\n".join(
        f"[{i}] {doc['title']}\n{doc['text']}"
        for i, (_, doc) in enumerate(search_results, 1)
    )

    prompt = f"""
너는 금융 FDS 업무를 지원하는 AI 어시스턴트다.

다음 참고 문서의 내용만 근거로 답변하라.
문서에 없는 정보는 추측하지 말고 모른다고 답하라.
가능하면 참고한 문서 번호를 표시하라.

[참고 문서]
{context}

[사용자 질문]
{question}

[답변]
"""

    if BACKEND == "ollama":
        result = call_api(
            LLM_URL,
            {
                "model": LLM_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0, "num_predict": 200},
            },
        )
        return result["response"]

    # llama.cpp 서버에서 현재 모델 ID 확인
    models = call_api("http://127.0.0.1:8080/v1/models")
    model_id = models["data"][0]["id"]

    result = call_api(
        LLM_URL,
        {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 200,
            "stream": False,
        },
    )

    return result["choices"][0]["message"]["content"]


# ============================================================
# 8. 메인 프로그램
# ============================================================


def main():
    """
    [전체 실행 흐름]
    1. 문서 Embedding(또는 저장된 벡터 사용)
    2. 사용자 질문 Embedding
    3. 유사한 문서 TOP 2 검색
    4. 검색 문서를 LLM에 전달
    5. RAG 최종 답변 출력
    """

    # question = (
    #     "해외에서 여러 건의 결제가 연속 발생하면 "
    #     "어떤 사항을 확인해야 하나요?"
    # )
    question = "카드를 잃어 버렸는데 어떻게 처리해야하나요?"

    # STEP 1. 전체 문서의 임베딩 벡터를 읽어옴
    doc_vectors = get_document_vectors()

    # STEP 2. 관련 문서 검색
    search_results = search_documents(question, doc_vectors)

    print("\n========== 검색 결과 ==========")

    for i, (score, doc) in enumerate(search_results, 1):
        print(f"{i}위: {doc['title']} ({score:.4f})")

    # STEP 3. 검색 결과를 근거로 답변 생성
    answer = generate_answer(question, search_results)

    print("\n========== RAG 최종 답변 ==========")
    print("질문:", question)
    print("\n답변:", answer)


# ============================================================
# 9. 프로그램 시작점
# ============================================================

if __name__ == "__main__":
    main()
