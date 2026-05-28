from llm import get_llm

llm = get_llm()

messages = [
    (
        "system",
        "당신은 사용자가 한 말을 영어로 번역하는 유능한 번역가입니다.",
    ),
    ("human", "안녕하세요"),
]

ai_msg = llm.invoke(messages)
print(ai_msg)