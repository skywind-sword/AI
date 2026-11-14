import requests
import time

URL = "http://127.0.0.1:11434/api/generate"
MODEL = "gemma3-4b-local"

PROMPT = "은행 콜센터에 적용할 수 있는 새로운 AI 서비스 아이디어를 하나 제안해줘."

# 실험할 생성 옵션 (temperature 에 따른 답변의 다양성 관찰을 위해 일단 top_p도 1.0으로 통일)
settings = [
    {"temperature": 0,   "top_p": 1.0},     # "top_p": 1.0},
    {"temperature": 0.5, "top_p": 1.0},     # "top_p": 0.9},
    {"temperature": 1.0, "top_p": 1.0},     # "top_p": 0.9},
    {"temperature": 1.5, "top_p": 1.0},     # "top_p": 0.9},
]

results = []

for options in settings:
    for repeat in range(1, 4):  # temperature 에 따른 답변의 다양성 관찰을 동일 질문을 반복 수행
        print(f"\n{'=' * 65}")
        print(
            f"Temperature={options['temperature']}, "
            f"Top-p={options['top_p']}, "
            f"반복={repeat}"
        )
        print("=" * 65)

        payload = {
            "model": MODEL,
            "prompt": PROMPT,
            "stream": False,
            "options": {
                **options,
                "num_predict": 400,
                # "seed": 42    # seed=42는 무작위 샘플링의 재현성을 높이기 위한 설정
                                # seed를 고정하면 동일한 조건에서 결과를 재현하기 쉬워집니다. 실행할 때마다 생성되는 답변의 다양성을 관찰하기 위해 제거
            }
        }

        start = time.perf_counter()

        response = requests.post(
            URL,
            json=payload,
            timeout=300
        )
        response.raise_for_status()

        data = response.json()
        elapsed = time.perf_counter() - start

        answer = data.get("response", "")
        output_tokens = data.get("eval_count", 0)
        generation_sec = data.get("eval_duration", 0) / 1e9

        print(answer)
        print(f"\n출력 토큰: {output_tokens}")
        print(f"소요 시간: {elapsed:.2f}초")
        print(f"종료 사유: {data.get('done_reason')}")

        results.append({
            "temperature": options["temperature"],
            "top_p": options["top_p"],
            "tokens": output_tokens,
            "seconds": generation_sec,
            "answer": answer,
            "done_reason": data.get("done_reason")
        })

print("\n========== 실험 결과 요약 ==========")
print(f"{'TEMP':>6} {'TOP-P':>7} {'TOKENS':>8} {'TOK/S':>9} {'REASON':>10}")

for r in results:
    speed = r["tokens"] / r["seconds"] if r["seconds"] else 0
    print(
        f"{r['temperature']:>6.1f}"
        f"{r['top_p']:>7.1f}"
        f"{r['tokens']:>8}"
        f"{speed:>9.2f}"
        f"{str(r['done_reason']):>10}"
X    )
