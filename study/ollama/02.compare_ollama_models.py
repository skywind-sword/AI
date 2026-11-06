# ========== 모델 성능 비교 ==========
# MODEL                       LOAD    INPUT   OUTPUT    TOTAL    TOK/S    REASON
# gemma3-4b-local             3.87     0.55    24.75    29.19    12.12    length
# llama3.2-3b-local           3.14     0.84    18.61    22.61    16.12    length
# qwen3-4b-local              4.37     0.60    23.26    28.25    12.89    length
# qwen3-8b-local             14.25     1.27    42.47    58.03     7.06    length
#
# ① 파라미터가 많으면 무조건 성능이 좋을까?
#   Qwen3 4B와 Qwen3 8B를 비교해 보면 됩니다.
#   일반적으로 큰 모델은 복잡한 문제를 해결하는 데 유리할 수 있지만, 모델 크기가 두 배라고 해서 정확도가 두 배 높아지는 것은 아닙니다.
# ② 왜 모델마다 토큰 생성 속도가 다를까?
#   모델 구조, 파라미터 수, 양자화 방식, CPU/GPU 사용 여부 등에 영향을 받기 때문입니다.
# ③ 왜 같은 질문인데 출력 토큰 수가 다를까?
#   모델마다 사용하는 토크나이저가 다르고, 생성하는 답변의 길이도 다르기 때문입니다.
# ④ Temperature를 변경하면 어떻게 될까?
#   예를 들어 다음 두 설정을 비교해 보세요.
#   "temperature": 0, "temperature": 1.0
#   Temperature는 다음 토큰을 선택할 때 확률 분포를 조절합니다.
#   낮은 값은 일반적으로 일관된 답변에, 높은 값은 다양한 답변에 유리합니다. 다만 temperature=0이라고 해서 모든 실행 결과가 반드시 동일한 것은 아닙니다.
import requests

# Ollama REST API
URL = "http://127.0.0.1:11434/api/generate"

# 비교 대상 모델
MODELS = [
    "gemma3-4b-local",
    "llama3.2-3b-local",
    "qwen3-4b-local",
    "qwen3-8b-local"
]

# 동일한 질문
PROMPT = "Python의 리스트와 튜플의 차이점을 예제와 함께 설명해줘."

# 결과 저장
results = []

for model in MODELS:

    print(f"\n{'=' * 60}")
    print(f"모델 실행: {model}")
    print("=" * 60)

    payload = {
        "model": model,
        "prompt": PROMPT,
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0,
            "num_predict": 300
        }
    }

    try:
        response = requests.post(
            URL,
            json=payload,
            timeout=600
        )

        response.raise_for_status()

        data = response.json()

        # 응답 내용
        print("\n[LLM 응답]")
        print(data.get("response", ""))

        # 성능 정보
        total = data.get("total_duration", 0) / 1e9
        load = data.get("load_duration", 0) / 1e9
        eval_time = data.get("eval_duration", 0) / 1e9

        input_tokens = data.get("prompt_eval_count", 0)
        output_tokens = data.get("eval_count", 0)
        
        model_loading_duration = round(data.get("load_duration", 0) / 1e9, 2)
        input_processing_duration = round(data.get("prompt_eval_duration", 0) / 1e9, 2)
        output_generation_duration = round(data.get("eval_duration", 0) / 1e9, 2)       
        done_reason = data.get("done_reason", "")
        
        speed = (
            output_tokens / eval_time
            if eval_time > 0 else 0
        )

        results.append({
            "model": model,
            "input": input_tokens,
            "output": output_tokens,
            "total": total,
            "load": load,
            "input_duration": input_processing_duration,
            "output_duration": output_generation_duration,
            "speed": speed,
            "done_reason": done_reason
        })

    except requests.RequestException as e:
        print(f"오류 발생: {e}")

# 최종 비교 결과
print("\n========== 모델 성능 비교 ==========")

print(
    f"{'MODEL':<23}"
    f"{'LOAD':>9}"
    f"{'INPUT':>9}"
    f"{'OUTPUT':>9}"
    f"{'TOTAL':>9}"
    f"{'TOK/S':>9}"
    f"{'REASON':>10}"
)

for r in results:
    print(
        f"{r['model']:<23}"
        f"{r['load']:>9.2f}"
        f"{r['input_duration']:>9.2f}"
        f"{r['output_duration']:>9.2f}"
        f"{r['total']:>9.2f}"
        f"{r['speed']:>9.2f}"
        f"{r['done_reason']:>10}"
    )