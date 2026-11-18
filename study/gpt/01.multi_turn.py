import requests

url = "http://localhost:11434/api/chat"

messages = [
    {
        "role": "system",
        "content": "당신은 친절한 Python 강사입니다."
    }
]

while True:
    question = input("질문: ")

    if question.lower() == "exit":
        break

    messages.append({
        "role": "user",
        "content": question
    })

    response = requests.post(
        url,
        json={
            "model": "gemma3-4b-local",
            "messages": messages,
            "stream": False
        },
        timeout=180
    )
    response.raise_for_status()

    answer = response.json()["message"]["content"]
    print("AI:", answer)

    messages.append({
        "role": "assistant",
        "content": answer
    })