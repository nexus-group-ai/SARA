from langchain_openai import ChatOpenAI
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os

def get_llm_client(**kwargs):
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        **kwargs
    )

def get_gemini_embeddings_client():
    return GoogleGenerativeAIEmbeddings(
        google_api_key=os.getenv("OPENAI_API_KEY"),
        model="models/text-embedding-004",
    )
