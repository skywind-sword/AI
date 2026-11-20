
# ============================================================
# 프로그램명 : 04.sampling_test.py
#
# [학습 목적]
# 1. LLM의 다음 토큰 예측(Next Token Prediction) 이해
# 2. Temperature와 Top-p에 따른 답변 다양성 비교
# 3. 동일한 질문을 반복하여 Sampling 특성 관찰
# 4. 토큰 생성속도 및 전체 응답시간 측정
#
# [LLM의 토큰 생성 원리]
# LLM은 지금까지의 입력 문맥을 바탕으로 다음 토큰의
# 확률 분포를 계산하고, Sampling 설정에 따라
# 다음 토큰을 선택한다.
#
# 선택한 토큰을 문맥에 추가하고 같은 과정을 반복하여
# 최종 답변을 생성한다.
#
# [Temperature]
# - 낮은 값 : 높은 확률의 토큰을 선택하는 경향 강화
# - 높은 값 : 확률 분포가 평탄해져 다양한 선택 가능
# - 0       : 일반적으로 가장 확률 높은 토큰 선택
#
# Temperature가 높다고 답변 품질이 좋아지는 것은 아니다.
# 정확한 규정 안내에는 낮은 값이 유리할 수 있지만,
# 낮은 Temperature도 할루시네이션을 방지하지는 못한다.
#
# [Top-p / Nucleus Sampling]
# 확률이 높은 토큰부터 누적하여 p에 도달하는 후보
# 집합을 만들고, 그 안에서 다음 토큰을 선택한다.
#
# 예: A=40%, B=30%, C=20%, D=10%
#     Top-p=0.7 -> A, B가 샘플링 후보
#     Top-p=0.9 -> A, B, C가 샘플링 후보
#
# Top-p=1.0은 이 제한을 사실상 해제하지만,
# Top-k, Min-p 등 다른 설정은 여전히 영향을 준다.
#
# [Seed]
# 난수 생성의 초기값으로, 동일한 Seed를 사용하면
# 결과의 재현성을 높일 수 있다.
#
# 이번에는 답변 다양성을 관찰하기 위해 Seed를
# 지정하지 않는다.
#
# [실험 조건]
# - Temperature : 0.0 / 0.5 / 1.0 / 1.5
# - Top-p       : 1.0으로 고정
# - 반복 횟수   : 각 설정마다 3회
# - 총 실행     : 12회
#
# 본 실험은 생성 결과의 변화를 관찰하는 실습이며,
# 통계적으로 유의미한 품질 평가를 수행하지는 않는다.
# ============================================================


# ============================================================
# 1. 라이브러리 import
# ============================================================

import requests
import time


# ============================================================
# 2. 실행 환경 설정
# ============================================================

# "ollama" : Windows PC의 Ollama
# "llama"  : 갤폴드6 Termux의 llama.cpp
BACKEND = "llama"

if BACKEND == "ollama":
    URL = "http://127.0.0.1:11434/api/generate"
    MODEL = "gemma3-4b-local"

elif BACKEND == "llama":
    URL = "http://127.0.0.1:8080/v1/chat/completions"
    MODEL = None  # 서버에서 모델명을 자동 조회

else:
    raise ValueError("BACKEND는 ollama 또는 llama여야 합니다.")


# 실험에 사용할 질문
PROMPT = (
    "은행 콜센터에 적용할 수 있는 "
    "새로운 AI 서비스 아이디어를 하나 제안해줘."
)

# 동일한 Sampling 설정으로 반복 실행할 횟수
REPEAT_COUNT = 3

# 최대 출력 토큰 수
NUM_PREDICT = 400


# ============================================================
# 3. Sampling 실험 설정
# ============================================================

# Temperature의 영향을 관찰하기 위해
# Top-p는 모두 동일한 1.0으로 설정한다.
#
# 변수 하나의 영향을 비교할 때는 다른 조건을
# 가능한 한 동일하게 유지하는 것이 중요하다.
#
# 단, Ollama와 llama.cpp의 기본 Sampling 옵션은
# 서로 다를 수 있으므로 플랫폼 간 결과 차이를
# Temperature만의 영향으로 해석해서는 안 된다.

SETTINGS = [
    {"temperature": 0.0, "top_p": 1.0},
    {"temperature": 0.5, "top_p": 1.0},
    {"temperature": 1.0, "top_p": 1.0},
    {"temperature": 1.5, "top_p": 1.0},
]


# ============================================================
# 4. 현재 LLM 모델명 확인
# ============================================================

def get_model_name():
    """
    [목적]
    실행 환경에서 사용할 모델명을 확인한다.

    Ollama는 등록된 모델명을 사용하고,
    llama.cpp는 실행 중인 서버에서 모델명을 조회한다.

    [반환값]
    str : API 요청에 사용할 모델명
    """

    if BACKEND == "ollama":
        return MODEL

    # llama.cpp 서버의 모델 목록 API 호출
    response = requests.get(
        "http://127.0.0.1:8080/v1/models",
        timeout=10
    )
    response.raise_for_status()

    models = response.json()["data"]

    if not models:
        raise RuntimeError("llama.cpp에 로딩된 모델이 없습니다.")

    # 현재는 Gemma 3 하나만 로딩된 서버를 가정한다.
    return models[0]["id"]


# ============================================================
# 5. LLM 답변 생성 함수
# ============================================================

def generate_answer(options, model_name):
    """
    [목적]
    지정한 Sampling 옵션으로 LLM 답변을 생성하고,
    응답시간과 토큰 생성 정보를 반환한다.

    [매개변수]
    options    : temperature, top_p 설정 Dictionary
    model_name : 사용할 LLM 모델명

    [반환값]
    dict
        answer      : 생성된 답변
        tokens      : 출력 토큰 수
        seconds     : 토큰 생성 평가시간(초)
        elapsed     : 전체 API 응답시간(초)
        done_reason : 생성 종료 사유

    [참고]
    출력 토큰 생성시간과 전체 API 응답시간은 다르다.

    전체 시간에는 프롬프트 처리, 모델 로딩,
    토큰 생성, HTTP 통신 등의 시간이 포함될 수 있다.
    """

    # --------------------------------------------------------
    # STEP 1. 실행 환경별 API 요청 데이터 구성
    # --------------------------------------------------------

    if BACKEND == "ollama":

        # Ollama는 prompt를 전달하고,
        # 생성 옵션은 options 객체 안에 지정한다.
        payload = {
            "model": model_name,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                **options,
                "num_predict": NUM_PREDICT
            }
        }

    else:

        # llama.cpp는 OpenAI 호환 Chat Completions API를
        # 사용하므로 messages 형식으로 질문을 전달한다.
        #
        # Temperature와 Top-p는 최상위 필드에 지정한다.
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT
                }
            ],
            "stream": False,
            "max_tokens": NUM_PREDICT,
            **options
        }

    # **options는 Dictionary Unpacking 문법이다.
    # temperature, top_p 항목을 현재 Dictionary에 합친다.

    # --------------------------------------------------------
    # STEP 2. REST API 호출 및 전체 응답시간 측정
    # --------------------------------------------------------

    # perf_counter()는 경과시간 측정용 고해상도 타이머
    start = time.perf_counter()

    response = requests.post(
        URL,
        json=payload,
        timeout=300
    )
    response.raise_for_status()

    data = response.json()

    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # STEP 3. 실행 환경별 생성 결과 추출
    # --------------------------------------------------------

    if BACKEND == "ollama":

        # Ollama의 응답 데이터
        answer = data.get("response", "")
        tokens = data.get("eval_count", 0)
        done_reason = data.get("done_reason")

        # Ollama는 토큰 평가시간을 나노초(ns)로 반환
        eval_duration = data.get("eval_duration", 0)

        seconds = (
            eval_duration / 1e9
            if eval_duration > 0 else None
        )

    else:

        # llama.cpp Chat Completions 응답
        #
        # choices[0].message.content : 생성된 답변
        # usage.completion_tokens    : 출력 토큰 수
        # choices[0].finish_reason   : 종료 사유
        choice = data["choices"][0]

        answer = choice["message"].get("content") or ""

        usage = data.get("usage") or {}
        timings = data.get("timings") or {}

        tokens = usage.get(
            "completion_tokens",
            timings.get("predicted_n", 0)
        )

        done_reason = choice.get("finish_reason")

        # llama.cpp의 timings.predicted_ms는
        # 출력 토큰 생성에 소요된 시간(밀리초)이다.
        #
        # 서버 버전에 따라 timings 정보가 없을 수 있다.
        predicted_ms = timings.get("predicted_ms")

        seconds = (
            predicted_ms / 1000
            if predicted_ms is not None and predicted_ms > 0
            else None
        )

    # --------------------------------------------------------
    # STEP 4. 공통 형식으로 결과 반환
    # --------------------------------------------------------

    # 플랫폼마다 응답 JSON 형식이 다르더라도
    # 이후 main()에서는 동일한 방식으로 처리할 수 있다.
    return {
        "answer": answer,
        "tokens": tokens,
        "seconds": seconds,
        "elapsed": elapsed,
        "done_reason": done_reason
    }


# ============================================================
# 6. 실험 결과 요약 출력 함수
# ============================================================

def print_summary(results):
    """
    [목적]
    Temperature별 토큰 수, 생성속도,
    전체 응답시간을 비교하여 출력한다.

    [성능 지표]
    - Tokens/sec = 출력 토큰 수 / 토큰 생성 평가시간
    - Elapsed    = API 요청부터 응답 수신까지의 전체 시간

    Tokens/sec는 모델의 생성 성능을 나타내는 지표이고,
    Elapsed는 사용자가 기다린 전체 시간에 가깝다.
    """

    print("\n========== 실험 결과 요약 ==========")

    print(
        f"{'TEMP':>6} "
        f"{'TOP-P':>6} "
        f"{'TOKENS':>7} "
        f"{'TOK/S':>9} "
        f"{'TOTAL(S)':>10} "
        f"{'REASON':>8}"
    )

    for r in results:

        # 서버가 토큰 평가시간을 반환한 경우에만
        # 순수 토큰 생성속도를 계산한다.
        #
        # 측정값이 없을 때 전체 응답시간으로 대체하면
        # 서로 다른 성능 지표가 섞이므로 "-"로 표시한다.
        if r["seconds"]:
            speed = r["tokens"] / r["seconds"]
            speed_text = f"{speed:.2f}"
        else:
            speed_text = "-"

        print(
            f"{r['temperature']:>6.1f} "
            f"{r['top_p']:>6.1f} "
            f"{r['tokens']:>7} "
            f"{speed_text:>9} "
            f"{r['elapsed']:>10.2f} "
            f"{str(r['done_reason']):>8}"
        )


# ============================================================
# 7. 메인 프로그램
# ============================================================

def main():
    """
    [전체 실행 흐름]

    1. 실행 중인 LLM 모델명 확인
    2. Temperature 설정을 순서대로 선택
    3. 동일 조건으로 3회씩 답변 생성
    4. 답변과 실행시간을 출력 및 저장
    5. 실험 결과를 표 형태로 요약

    이번 실험의 핵심은 답변의 다양성을 관찰하는 것이다.

    출력 토큰 수, 응답시간은 답변 길이와
    모델 로딩 상태에 따라서도 달라질 수 있다.
    """

    model_name = get_model_name()

    print("실행 환경:", BACKEND)
    print("실행 모델:", model_name)
    print("실험 횟수:", len(SETTINGS) * REPEAT_COUNT)

    # 실험 결과를 Dictionary 목록으로 저장
    results = []

    # --------------------------------------------------------
    # STEP 1. Temperature별 실험
    # --------------------------------------------------------

    for options in SETTINGS:

        # 동일한 설정을 3회 반복하여
        # Seed 미지정 상태의 답변 다양성을 관찰한다.
        for repeat in range(1, REPEAT_COUNT + 1):

            print(f"\n{'=' * 65}")
            print(
                f"Temperature={options['temperature']}, "
                f"Top-p={options['top_p']}, "
                f"반복={repeat}"
            )
            print("=" * 65)

            # ------------------------------------------------
            # STEP 2. LLM 답변 생성
            # ------------------------------------------------

            result = generate_answer(options, model_name)

            # ------------------------------------------------
            # STEP 3. 답변 및 성능 정보 출력
            # ------------------------------------------------

            print(result["answer"])

            print(f"\n출력 토큰: {result['tokens']}")
            print(f"전체 소요시간: {result['elapsed']:.2f}초")

            if result["seconds"] is not None:
                print(
                    f"토큰 평가시간: {result['seconds']:.2f}초"
                )
            else:
                print("토큰 평가시간: 측정값 없음")

            print(f"종료 사유: {result['done_reason']}")

            # ------------------------------------------------
            # STEP 4. 실험 결과 누적
            # ------------------------------------------------

            # 생성 설정과 결과를 하나의 Dictionary로 결합
            results.append({
                **options,
                **result
            })

    # --------------------------------------------------------
    # STEP 5. 전체 실험 결과 요약
    # --------------------------------------------------------

    print_summary(results)


# ============================================================
# 8. 프로그램 시작점
# ============================================================

# 직접 실행할 때만 main()을 호출한다.
# 다른 Python 프로그램에서 import하면 실행되지 않는다.

if __name__ == "__main__":
    main()
