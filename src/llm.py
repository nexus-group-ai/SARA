from langchain_openai import ChatOpenAI
import os

def get_llm_client(**kwargs):
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs
    )
