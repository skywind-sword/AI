# Python (requests) → Ollama (localhost:11434) → Gemma 3 (로컬 LLM)
# 내용 : HTTP 요청, JSON 응답, 모델 파라미터, 토큰 사용량 확인
# Ollama가 단순한 채팅 프로그램이 아니라 로컬에서 실행되는 API 서버 역할도 한다。


import requests

url = "http://localhost:11434/api/generate"

models = [
    "gemma3-4b-local:latest",
    # "llama3.2-3b-local:latest",
    # "qwen3-4b-local:latest",
    # "qwen3-8b-local:latest"
]

prompt = "Java와 Python 각 언어의 장단점과 AI를 공부할 때 어떤 언어로 하면 좋을지 알려줘"

for model in models:
    print(f"=================== 사용모델 : {model} ================")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": 300,
            "temperature": 0
        }
    }

    response = requests.post(url, json=payload, timeout=180)
    response.raise_for_status()

    result = response.json()

    print("=== LLM 응답 ===")
    print(result["response"])

    print("\n=== 실행 정보 ===")
    print("입력 토큰:", result.get("prompt_eval_count"))
    print("출력 토큰:", result.get("eval_count"))
    print("전체 처리 시간(초):", round(result["total_duration"] / 1e9, 2))

    print("\n=== 성능 분석 ===")


    # 토큰 처리속도 측정
    total_duration = result.get("total_duration", 0)
    load_duration = result.get("load_duration", 0)
    eval_duration = result.get("eval_duration", 0)

    # 출력 토큰 수
    eval_count = result.get("eval_count", 0)

    print("전체 처리 시간:", round(total_duration / 1e9, 2), "초")
    print("모델 로딩 시간:", round(load_duration / 1e9, 2), "초")
    print("토큰 생성 시간:", round(eval_duration / 1e9, 2), "초")

    if eval_duration > 0:
        tokens_per_sec = eval_count / (eval_duration / 1e9)
        print("토큰 생성 속도:", round(tokens_per_sec, 2), "tokens/sec")    
    
    