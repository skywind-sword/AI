# ============================================================
# 프로그램명 : 02.compare_ollama_models.py
#
# [학습 목적]
# 1. 여러 LLM의 추론(Inference) 성능 비교
# 2. 모델 파라미터 수와 처리속도의 관계 이해
# 3. 모델 로딩, 입력 처리, 출력 생성시간 구분
# 4. Tokens/sec를 계산하여 토큰 생성속도 측정
# 5. 모델 선정 시 성능과 품질의 차이 이해
#
# [LLM의 기본 추론 과정]
#
# 사용자 Prompt 입력
#       ↓
# ① Model Loading
#    모델 가중치를 메모리에 적재
#       ↓
# ② Prompt Processing (Prefill)
#    입력 토큰을 처리하고 문맥 정보 계산
#       ↓
# ③ Token Generation (Decode)
#    다음 토큰을 예측하여 순차적으로 답변 생성
#       ↓
# ④ 최종 답변 반환
#
# [주요 성능 지표]
#
# LOAD   : 모델 로딩시간(초)
# INPUT  : 입력 Prompt 처리시간(초)
# OUTPUT : 출력 토큰 생성시간(초)
# TOTAL  : 전체 추론 처리시간(초)
# TOK/S  : 초당 출력 토큰 수
#
# 주의:
# INPUT, OUTPUT은 토큰 개수가 아닌 처리시간이다.
# TOTAL에는 기타 내부 처리시간도 포함될 수 있으므로
# LOAD + INPUT + OUTPUT과 정확히 같지는 않을 수 있다.
#
# [모델 크기와 성능]
#
# 3B, 4B, 8B 등은 대략적인 파라미터 수를 의미한다.
# B = Billion = 10억
#
# 일반적으로 모델이 커지면 메모리와 연산량이
# 증가하여 추론속도가 느려질 수 있다.
#
# 반면 복잡한 추론 및 언어 이해에서는
# 더 큰 모델이 유리할 가능성이 있다.
#
# 그러나 모델 파라미터 수가 2배라고 해서
# 정확도가 2배 높아지는 것은 아니다.
#
# 실제 성능은 모델 구조, 양자화 방식,
# CPU/GPU, 메모리 대역폭, 컨텍스트 길이,
# 추론 엔진 등에 따라 달라진다.
#
# [이번 실습의 한계]
#
# 동일한 질문으로 처리속도와 답변을 비교한다.
# 하지만 질문 하나만으로 모델의 정확도나
# 전체적인 지능 수준을 판단할 수는 없다.
#
# 또한 모델별 Tokenizer가 다르므로
# 출력 토큰 수와 TOK/S를 비교할 때 주의해야 한다.
# ============================================================


# ============================================================
# 1. 라이브러리 import
# ============================================================

import requests
import time


# ============================================================
# 2. 실행 환경 설정
# ============================================================

# "ollama" : Windows PC
# "llama"  : 갤폴드6 Termux + llama.cpp
BACKEND = "ollama"

if BACKEND == "ollama":

    URL = "http://127.0.0.1:11434/api/generate"

    # Ollama에 등록된 비교 대상 모델
    MODELS = [
        "gemma3-4b-local",
        "llama3.2-3b-local",
        "qwen3-4b-local",
        "qwen3-8b-local"
    ]

elif BACKEND == "llama":

    URL = "http://127.0.0.1:8080/v1/chat/completions"

    # llama.cpp는 현재 기동된 모델을 대상으로 측정한다.
    # 모델명은 /v1/models API에서 조회한다.
    MODELS = []

else:
    raise ValueError("BACKEND는 ollama 또는 llama여야 합니다.")


# 모든 모델에 동일하게 전달할 질문
PROMPT = (
    "Python의 리스트와 튜플의 차이점을 "
    "예제와 함께 설명해줘."
)

# 답변 생성 옵션
TEMPERATURE = 0
NUM_PREDICT = 300

# HTTP 응답 대기 제한시간
TIMEOUT = 600


# ============================================================
# 3. 모델명 조회 함수
# ============================================================

def get_models():
    """
    [목적]
    비교할 모델 목록을 반환한다.

    Ollama:
        사전에 지정한 여러 모델을 비교한다.

    llama.cpp:
        현재 서버에 로딩된 모델 목록을 조회한다.

    [반환값]
    list[str] : 모델명 목록
    """

    if BACKEND == "ollama":
        return MODELS

    # llama.cpp의 OpenAI 호환 모델 조회 API
    response = requests.get(
        "http://127.0.0.1:8080/v1/models",
        timeout=10
    )

    response.raise_for_status()

    data = response.json().get("data", [])

    models = [
        item["id"]
        for item in data
    ]

    if not models:
        raise RuntimeError("실행 중인 LLM 모델이 없습니다.")

    return models


# ============================================================
# 4. 모델 추론 함수
# ============================================================

def generate_answer(model):
    """
    [목적]
    지정한 모델에 동일한 질문을 전달하고
    생성된 답변과 성능 측정값을 반환한다.

    [매개변수]
    model : 실행할 LLM 모델명

    [반환값]
    dict:
        model           : 모델명
        answer          : 생성된 답변
        input_tokens    : 입력 토큰 수
        output_tokens   : 출력 토큰 수
        load            : 모델 로딩시간
        input_duration  : 입력 처리시간
        output_duration : 출력 생성시간
        total           : 서버 내부 전체 처리시간
        elapsed         : 클라이언트 전체 경과시간
        speed           : 초당 출력 토큰 수
        done_reason     : 생성 종료 사유

    [주요 원리]

    Prompt Processing:
        입력 토큰을 처리하여 다음 토큰 생성에
        필요한 내부 상태를 준비하는 단계

    Token Generation:
        생성된 토큰을 문맥에 추가하면서
        다음 토큰을 순차적으로 예측하는 단계

    두 단계는 연산 특성이 달라 처리속도도 다르다.
    """

    # --------------------------------------------------------
    # STEP 1. 실행 환경에 따른 요청 데이터 구성
    # --------------------------------------------------------

    if BACKEND == "ollama":

        payload = {
            "model": model,
            "prompt": PROMPT,
            "stream": False,

            # 추론/사고 모드를 별도로 지원하는 모델에서는
            # 해당 기능을 비활성화하도록 요청한다.
            # 지원 여부와 동작은 모델에 따라 다르다.
            "think": False,

            "options": {
                "temperature": TEMPERATURE,
                "num_predict": NUM_PREDICT
            }
        }

    else:

        # llama.cpp OpenAI 호환 Chat API
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": PROMPT
                }
            ],
            "temperature": TEMPERATURE,
            "max_tokens": NUM_PREDICT,
            "stream": False
        }

    # Temperature=0은 일반적으로
    # 가장 확률이 높은 토큰을 선택하는 설정이다.
    #
    # 모델 답변의 다양성을 줄이기 위한 설정이지만
    # 실행 결과가 항상 동일함을 보장하지는 않는다.
    #
    # NUM_PREDICT는 생성 가능한 최대 출력 토큰 수이다.

    # --------------------------------------------------------
    # STEP 2. API 호출 및 전체 경과시간 측정
    # --------------------------------------------------------

    start = time.perf_counter()

    response = requests.post(
        URL,
        json=payload,
        timeout=TIMEOUT
    )

    response.raise_for_status()

    data = response.json()

    # 실제 클라이언트에서 측정한 전체 소요시간
    elapsed = time.perf_counter() - start

    # --------------------------------------------------------
    # STEP 3. 추론 결과 및 성능 정보 추출
    # --------------------------------------------------------

    if BACKEND == "ollama":

        answer = data.get("response", "")

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)

        # Ollama의 시간 정보는 나노초(ns) 단위이다.
        # 1초 = 10억 나노초
        load = data.get("load_duration", 0) / 1e9

        input_duration = (
            data.get("prompt_eval_duration", 0) / 1e9
        )

        output_duration = (
            data.get("eval_duration", 0) / 1e9
        )

        total = data.get("total_duration", 0) / 1e9

        done_reason = data.get("done_reason", "")

    else:

        # llama.cpp의 OpenAI 호환 응답 구조
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        timings = data.get("timings") or {}

        answer = choice["message"].get("content") or ""

        input_tokens = usage.get(
            "prompt_tokens", 0
        )

        output_tokens = usage.get(
            "completion_tokens", 0
        )

        # llama.cpp의 timings 정보는 밀리초(ms) 단위
        # 서버 버전에 따라 일부 항목이 없을 수 있다.
        #
        # 모델 로딩시간은 일반적인 요청 응답에서
        # 별도로 제공되지 않으므로 None으로 처리한다.
        load = None

        prompt_ms = timings.get("prompt_ms")
        predicted_ms = timings.get("predicted_ms")

        input_duration = (
            prompt_ms / 1000
            if prompt_ms is not None else None
        )

        output_duration = (
            predicted_ms / 1000
            if predicted_ms is not None else None
        )

        # llama.cpp의 요청 처리시간과
        # Ollama의 total_duration은 정의가 다를 수 있다.
        # 이번에는 클라이언트 측 전체 시간만 사용한다.
        total = None

        done_reason = choice.get("finish_reason", "")

    # --------------------------------------------------------
    # STEP 4. 초당 출력 토큰 수 계산
    # --------------------------------------------------------

    # Tokens/sec = 출력 토큰 수 / 출력 토큰 생성시간
    #
    # 예:
    # 300토큰 생성에 25초가 소요되었다면
    # TOK/S = 300 / 25 = 12
    #
    # 즉, 초당 약 12개 토큰을 생성한 것이다.
    #
    # 입력 처리시간과 모델 로딩시간은 제외한다.
    if output_duration and output_duration > 0:
        speed = output_tokens / output_duration
    else:
        speed = None

    # --------------------------------------------------------
    # STEP 5. 측정 결과 반환
    # --------------------------------------------------------

    return {
        "model": model,
        "answer": answer,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "load": load,
        "input_duration": input_duration,
        "output_duration": output_duration,
        "total": total,
        "elapsed": elapsed,
        "speed": speed,
        "done_reason": done_reason
    }


# ============================================================
# 5. 성능 비교 결과 출력 함수
# ============================================================

def print_summary(results):
    """
    [목적]
    모델별 성능 측정 결과를 표 형태로 출력한다.

    [측정 항목]

    LOAD:
        모델 가중치를 메모리에 적재한 시간

    INPUT:
        입력 Prompt를 처리한 시간

    OUTPUT:
        출력 토큰을 생성한 시간

    TOTAL:
        서버가 보고한 전체 처리시간

    TOK/S:
        초당 출력 토큰 생성속도

    ELAPSED:
        Python에서 측정한 전체 경과시간

    [주의]
    측정값이 없는 항목은 "-"로 표시한다.
    서로 다른 의미의 시간 지표를 임의로 대체하지 않는다.
    """

    print("\n========== 모델 성능 비교 ==========")

    print(
        f"{'MODEL':<23}"
        f"{'LOAD':>8}"
        f"{'INPUT':>8}"
        f"{'OUTPUT':>8}"
        f"{'TOTAL':>8}"
        f"{'TOK/S':>9}"
        f"{'ELAPSED':>9}"
        f"{'REASON':>9}"
    )

    for r in results:

        # 측정값이 없는 경우 "-"로 출력
        load = f"{r['load']:.2f}" if r["load"] is not None else "-"
        input_sec = (
            f"{r['input_duration']:.2f}"
            if r["input_duration"] is not None else "-"
        )
        output_sec = (
            f"{r['output_duration']:.2f}"
            if r["output_duration"] is not None else "-"
        )
        total = (
            f"{r['total']:.2f}"
            if r["total"] is not None else "-"
        )
        speed = (
            f"{r['speed']:.2f}"
            if r["speed"] is not None else "-"
        )

        print(
            f"{r['model']:<23}"
            f"{load:>8}"
            f"{input_sec:>8}"
            f"{output_sec:>8}"
            f"{total:>8}"
            f"{speed:>9}"
            f"{r['elapsed']:>9.2f}"
            f"{str(r['done_reason']):>9}"
        )


# ============================================================
# 6. 메인 프로그램
# ============================================================

def main():
    """
    [전체 실행 흐름]

    1. 비교할 모델 목록 확인
    2. 모든 모델에 동일한 Prompt 전달
    3. 응답 내용 및 성능 측정
    4. 결과를 리스트에 누적
    5. 모델별 성능 비교표 출력

    모델별 품질 비교를 위해서는 응답속도 외에도
    실제 정답 정확도, 근거 충실도 등의 평가가 필요하다.
    """

    models = get_models()
    results = []

    print("실행 환경:", BACKEND)
    print("비교 모델 수:", len(models))

    # --------------------------------------------------------
    # STEP 1. 모델별 추론 실행
    # --------------------------------------------------------

    for model in models:

        print(f"\n{'=' * 60}")
        print(f"모델 실행: {model}")
        print("=" * 60)

        try:

            # 동일한 Prompt와 생성 옵션으로 답변 생성
            result = generate_answer(model)

            print("\n[LLM 응답]")
            print(result["answer"])

            print("\n[성능 정보]")
            print(f"입력 토큰: {result['input_tokens']}")
            print(f"출력 토큰: {result['output_tokens']}")
            print(f"전체 경과시간: {result['elapsed']:.2f}초")
            print(f"종료 사유: {result['done_reason']}")

            # 각 모델의 측정 결과 저장
            results.append(result)

        except requests.RequestException as e:

            # 특정 모델에서 API 오류가 발생하더라도
            # 다른 모델에 대한 실험은 계속 진행한다.
            print(f"모델 실행 오류: {e}")

    # --------------------------------------------------------
    # STEP 2. 최종 결과 출력
    # --------------------------------------------------------

    print_summary(results)


# ============================================================
# 7. 프로그램 시작점
# ============================================================

# 직접 실행한 경우에만 main() 호출
if __name__ == "__main__":
    main()
