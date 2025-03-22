import sys
import os
import pandas as pd
import json
from dotenv import load_dotenv
import pickle

from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.rate_limiters import InMemoryRateLimiter

from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.llm import get_azure_embeddings_client, get_llm_client, get_gemini_llm_client

if not load_dotenv():
    raise Exception('Error loading .env file. Make sure to place valid keys in the .env file.')


ARTICLES_CLEAN_DIR = os.path.join("..", "data", "articles_clean")
FILTERED_METADATA_PATH = os.path.join("..", "data", "filtered_metadata.csv")
DB_PATH = os.path.join("ai_topic.db")

if not os.path.exists(DB_PATH):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)


filtered_metadata = pd.read_csv(FILTERED_METADATA_PATH)

def get_documents_from_path(filenames: list[str]) -> [Document]:
    documents = []
    
    for file_name in filenames:
        file_path = os.path.join(ARTICLES_CLEAN_DIR, file_name)
        with open(file_path, "r", encoding="utf-8") as file:
            file = json.load(file)

        text = file.get("text", "")
        documents.append(Document(page_content=text, metadata={
            "title": file.get("title", ""),
            "author": file.get("author", ""),
            "published_at": file.get("published_at", ""),
            "id": file.get("id", ""),
        }))

    return documents

documents = get_documents_from_path(filtered_metadata["filename"])


# Create database
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, separators=["\n\n", "\n"])

# Split documents and create vector database
texts = text_splitter.split_documents(documents)


embeddings = get_azure_embeddings_client(
    chunk_size=512, # number of documents' chunks processed in parallel, decrease if you hit rate limits
    show_progress_bar=True,
)

db = FAISS.from_documents(texts, embeddings)

with open(DB_PATH, "wb") as f:
    pickle.dump(db.serialize_to_bytes(), f)