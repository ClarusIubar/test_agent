from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(model="gpt-4o")