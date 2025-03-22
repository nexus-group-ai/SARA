import sys
import os
import pandas as pd
import json
from dotenv import load_dotenv
import tiktoken
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
DB_PATH = os.path.join("rag", "ai_topic.db")


if not os.path.exists(DB_PATH):
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)






embeddings = get_azure_embeddings_client(
    chunk_size=512, # number of documents' chunks processed in parallel, decrease if you hit rate limits
    show_progress_bar=True,
)

# CHECKPOINT: Load vector DB
with open(DB_PATH, "rb") as f:
    serialized_data = pickle.load(f)

# Reconstruct the FAISS database
db = FAISS.deserialize_from_bytes(serialized_data, embeddings, allow_dangerous_deserialization=True)




# FYI: free tier Gemini LLM 
# rate_limiter = InMemoryRateLimiter(
#     requests_per_second=0.5,  # <-- Gemini Free Tier
#     check_every_n_seconds=0.1,
# )

# llm = get_gemini_llm_client(
#     max_tokens=1024,
#     temperature=0.2,
#     rate_limiter=rate_limiter,
# )

# Default go-to Openrouter LLM - check README for other available models
llm = get_llm_client(
    # Configurable parameters
    max_tokens=1024,
    temperature=0.2,
)



system_prompt = """
You are an expert assistant to find sentences or articles that are about similar topics to that of the inputed sentence. Use only the following retrieved context to answer the question accurately and concisely. 
If you find absolutely no sentences or articles relating to this topic, say "I don't know".
Context: {context}
Question: {question}
"""

prompt_template = PromptTemplate(
    input_variables=["context", "question"], 
    template=system_prompt
)

retrieval_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt_template}
)


def ask_question(query):
    if query == 'sample1':
        query = 'Künstliche Intelligenz bietet die Freiheit, viele Dinge gleichzeitig und schnell zu erledigen.'
    if query == 'sample2':
        query = "Es geht um das 'Smart Home', das 'intelligente Heim' und wie man in Zukunft wohnen wollen soll"
    if query == 'sample3':
        query = 'Pünktlich zur Internationalen Orchideen- und Tillandsienschau der Blumengärten Hirschstetten wartet der Botanische Garten der Universität Wien mit einer kleinen Sensation auf'
    
    response = retrieval_chain.invoke({"query": query})
    
    # Print for debugging if needed but also return full response
    print(f"Question: Could you find articles that are related to this quote and summarize them: {query}?\nAnswer: {response['result']}")
    print("\nSources: \n")
    for source in response["source_documents"]:
        print(source.metadata)
    
    return response