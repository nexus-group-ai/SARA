"""
SARA Functional Contradiction Checker Demo
This demo uses the actual vector database and LLM to find contradictions.
Includes detailed debug information to show every step of the process.
"""

import sys
import os
import json
import pickle
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt_tab')

print(f"[DEBUG] Starting SARA Contradiction Checker Demo")
print(f"[DEBUG] Current working directory: {os.getcwd()}")
print(f"[DEBUG] Python version: {sys.version}")
print(f"[DEBUG] Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

try:
    print("[DEBUG] Checking for NLTK punkt tokenizer...")
    nltk.data.find('tokenizers/punkt')
    print("[DEBUG] NLTK punkt tokenizer already installed")
except LookupError:
    print("[DEBUG] NLTK punkt tokenizer not found, downloading...")
    nltk.download('punkt', quiet=True)
    print("[DEBUG] NLTK punkt tokenizer installation complete")

# Add the parent directory to the path to access the src module
module_path = os.path.abspath(os.path.join('.'))
if module_path not in sys.path:
    sys.path.append(module_path)

from src.llm import get_azure_embeddings_client, get_llm_client
from langchain.vectorstores import FAISS
from langchain_core.documents import Document

# Load environment variables and check for API keys
if not load_dotenv():
    raise Exception('Error loading .env file. Make sure to place valid keys in the .env file.')

# Constants
ARTICLES_CLEAN_DIR = os.path.join("data", "articles_clean")
DB_PATH = os.path.join("rag", "ai_topic.db")

def extract_sentences(article_text, max_sentences=3):
    """Extract a sample of sentences from the article text."""
    # Tokenize the article into sentences
    print("\n[DEBUG] Tokenizing article text into sentences...")
    all_sentences = sent_tokenize(article_text)
    print(f"[DEBUG] Found {len(all_sentences)} total sentences in the article")
    
    # Filter out very short sentences (likely incomplete or titles)
    print("[DEBUG] Filtering out very short sentences...")
    valid_sentences = [s for s in all_sentences if len(s.split()) > 5]
    print(f"[DEBUG] After filtering: {len(valid_sentences)} valid sentences")
    
    # If we have fewer sentences than the max, return all of them
    if len(valid_sentences) <= max_sentences:
        print(f"[DEBUG] Returning all {len(valid_sentences)} valid sentences (fewer than requested max)")
        return valid_sentences
    
    # Otherwise, take sentences from different parts of the article
    print(f"[DEBUG] Selecting {max_sentences} representative sentences from different parts of the article...")
    selected = []
    selected.append(valid_sentences[0])  # First sentence
    print(f"[DEBUG] Selected first sentence: '{valid_sentences[0][:50]}...'")
    
    if max_sentences > 2:
        selected.append(valid_sentences[-1])  # Last sentence
        print(f"[DEBUG] Selected last sentence: '{valid_sentences[-1][:50]}...'")
    
    if max_sentences > 3:
        middle_idx = len(valid_sentences) // 2
        selected.append(valid_sentences[middle_idx])  # Middle sentence
        print(f"[DEBUG] Selected middle sentence: '{valid_sentences[middle_idx][:50]}...'")
    
    # Fill the rest with sentences at equal intervals
    remaining = max_sentences - len(selected)
    if remaining > 0 and len(valid_sentences) > max_sentences:
        step = len(valid_sentences) // (remaining + 1)
        print(f"[DEBUG] Selecting {remaining} additional sentences at intervals of {step}")
        for i in range(1, remaining + 1):
            idx = i * step
            if idx < len(valid_sentences) and valid_sentences[idx] not in selected:
                selected.append(valid_sentences[idx])
                print(f"[DEBUG] Selected additional sentence {i}: '{valid_sentences[idx][:50]}...'")
    
    print(f"[DEBUG] Final selection: {len(selected)} sentences")
    return selected

def find_related_sentences(db, sentence, current_article_id=None, k=3):
    """Find related sentences from other articles using RAG."""
    print(f"\n[DEBUG] Finding related sentences for: '{sentence[:70]}...'")
    print(f"[DEBUG] Current article ID: {current_article_id}")
    print(f"[DEBUG] Looking for up to {k} related sentences...")
    
    try:
        # Perform the similarity search
        print("[DEBUG] Performing vector similarity search...")
        search_k = k+5  # Get a few extra in case we need to filter
        print(f"[DEBUG] Requesting {search_k} results to handle filtered items")
        docs = db.similarity_search_with_score(sentence, k=search_k)
        print(f"[DEBUG] Received {len(docs)} initial results from vector search")
        
        # Filter out sentences from the current article if an ID is provided
        print("[DEBUG] Filtering results to exclude current article and exact matches...")
        filtered_docs = []
        for i, (doc, score) in enumerate(docs):
            doc_id = doc.metadata.get("id", "")
            doc_title = doc.metadata.get("title", "Unknown")
            
            print(f"[DEBUG] Examining result {i+1}: '{doc.page_content[:50]}...'")
            print(f"[DEBUG]   From article: {doc_title} (ID: {doc_id})")
            print(f"[DEBUG]   Similarity score: {score}")
            
            # Skip if from the same article
            if current_article_id and doc_id == current_article_id:
                print(f"[DEBUG]   SKIPPING - Same article as source")
                continue
            
            # Skip if the content contains the exact query (likely the same sentence)
            if sentence in doc.page_content:
                print(f"[DEBUG]   SKIPPING - Contains exact source sentence")
                continue
            
            print(f"[DEBUG]   KEEPING - Different article and not exact match")
            filtered_docs.append({
                "content": doc.page_content, 
                "metadata": doc.metadata, 
                "score": score
            })
            
            # Stop once we have enough documents
            if len(filtered_docs) >= k:
                print(f"[DEBUG]   Reached target of {k} filtered results, stopping search")
                break
        
        print(f"[DEBUG] After filtering: found {len(filtered_docs)} related sentences from other articles")
        for i, doc in enumerate(filtered_docs):
            print(f"[DEBUG] Related sentence {i+1}: '{doc['content'][:70]}...'")
            print(f"[DEBUG]   From: {doc['metadata'].get('title', 'Unknown')} (Author: {doc['metadata'].get('author', 'Unknown')})")
            print(f"[DEBUG]   Score: {doc['score']}")
        
        return filtered_docs
    except Exception as e:
        print(f"[ERROR] Error finding related sentences: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return []

def check_contradiction(llm, source_sentence, related_sentence):
    """Check if the related sentence contradicts the source sentence using the LLM."""
    print(f"\n[DEBUG] Checking for contradiction between:")
    print(f"[DEBUG] Sentence 1: '{source_sentence[:100]}...'")
    print(f"[DEBUG] Sentence 2: '{related_sentence[:100]}...'")
    
    prompt = f"""Analyze these two sentences and determine if they contradict each other:
    
    Sentence 1: "{source_sentence}"
    Sentence 2: "{related_sentence}"
    
    Respond with a JSON object with these fields:
    - "contradicts": true if the sentences contradict each other, false otherwise
    - "explanation": a brief explanation of why or why not
    
    Consider:
    - Direct factual contradiction (e.g., "The market increased" vs "The market decreased")
    - Temporal contradiction (statements about different time periods aren't contradictory)
    - Opinion vs fact (opinions don't contradict facts)
    - Different aspects of the same topic aren't contradictory
    
    Output only valid JSON, with no additional text.
    """
    
    print(f"[DEBUG] Sending prompt to LLM (length: {len(prompt)} characters)")
    try:
        print("[DEBUG] Waiting for LLM response...")
        response = llm.invoke(prompt)
        print(f"[DEBUG] Received response (length: {len(response)} characters)")
        print(f"[DEBUG] Raw response: '{response[:200]}...'")
        
        # Extract the JSON part
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        
        if json_start >= 0 and json_end > json_start:
            print(f"[DEBUG] Found JSON from position {json_start} to {json_end}")
            json_str = response[json_start:json_end]
            print(f"[DEBUG] Extracted JSON: {json_str}")
            
            result = json.loads(json_str)
            contradicts = result.get("contradicts", False)
            explanation = result.get("explanation", "No explanation provided")
            
            print(f"[DEBUG] Contradiction found: {contradicts}")
            print(f"[DEBUG] Explanation: {explanation}")
            
            return result
        else:
            print("[ERROR] Could not extract valid JSON from the model response.")
            print(f"[ERROR] JSON markers not found. json_start={json_start}, json_end={json_end}")
            return {"contradicts": False, "explanation": "Error processing response - no valid JSON found"}
    except Exception as e:
        print(f"[ERROR] Error checking contradiction: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return {"contradicts": False, "explanation": f"Error: {str(e)}"}

def load_article(article_id):
    """Load the test article from test.json regardless of article_id."""
    test_path = "test.json"
    print(f"\n[DEBUG] Loading test article from: {test_path}")
    
    try:
        with open(test_path, "r", encoding="utf-8") as f:
            article_data = json.load(f)
            print(f"[DEBUG] Successfully loaded test article")
            print(f"[DEBUG] Article ID: {article_data.get('id', 'No ID')}")
            print(f"[DEBUG] Title: {article_data.get('title', 'No title')}")
            print(f"[DEBUG] Word count: {len(article_data.get('text', '').split())}")
            return article_data
    except FileNotFoundError:
        print(f"[ERROR] Test file not found at {test_path}")
        print(f"[INFO] Please create a test.json file with article content")
        return None
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON in test file")
        return None
    except Exception as e:
        print(f"[ERROR] Error loading test article: {e}")
        return None

def run_contradiction_checker(article_id, max_sentences=3):
    """
    Run the contradiction checker on an article.
    
    Args:
        article_id (str): The ID of the article to analyze
        max_sentences (int): Maximum number of sentences to analyze from the article
    """
    print("\n" + "=" * 100)
    print(f"SARA CONTRADICTION CHECKER - DETAILED DEBUG MODE".center(100))
    print("=" * 100)
    print(f"\nStarting analysis for article ID: {article_id}")
    print(f"Maximum sentences to analyze: {max_sentences}")
    print(f"Current time: {pd.Timestamp.now()}")
    print(f"Python version: {sys.version}")
    print("\n" + "-" * 100)
    
    # Load the article
    print("\n[STEP 0] LOADING ARTICLE DATA")
    print("-" * 50)
    
    start_time = pd.Timestamp.now()
    article_data = load_article(article_id)
    if not article_data:
        print(f"[ERROR] Article with ID {article_id} not found.")
        return
    
    print(f"\n[INFO] Article successfully loaded:")
    print(f"  Title: {article_data.get('title')}")
    print(f"  Published: {article_data.get('published_at')}")
    print(f"  Author: {article_data.get('author')}")
    print(f"  Section: {article_data.get('section')}")
    print(f"  Category: {article_data.get('category')}")
    print(f"  Word count: {len(article_data.get('text', '').split())}")
    print(f"  Time taken: {pd.Timestamp.now() - start_time}")
    
    print("\n" + "-" * 100)
    print("[STEP 1] INITIALIZING CLIENTS")
    print("-" * 50)
    
    # Get the embeddings client
    print("\n[DEBUG] Initializing embeddings client...")
    start_time = pd.Timestamp.now()
    try:
        embeddings = get_azure_embeddings_client(
            chunk_size=512,
            show_progress_bar=True,
        )
        print(f"[DEBUG] Embeddings client initialized successfully")
        print(f"[DEBUG] Embeddings model: {getattr(embeddings, 'model', 'Unknown')}")
        print(f"[DEBUG] Time taken: {pd.Timestamp.now() - start_time}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize embeddings client: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return
    
    # Load the vector database
    print("\n[DEBUG] Loading vector database...")
    start_time = pd.Timestamp.now()
    try:
        print(f"[DEBUG] Database path: {DB_PATH}")
        with open(DB_PATH, "rb") as f:
            serialized_data = pickle.load(f)
        
        print(f"[DEBUG] Serialized data size: {len(serialized_data) / (1024*1024):.2f} MB")
        
        # Reconstruct the FAISS database
        print("[DEBUG] Deserializing FAISS database...")
        db_start_time = pd.Timestamp.now()
        db = FAISS.deserialize_from_bytes(serialized_data, embeddings, allow_dangerous_deserialization=True)
        print(f"[DEBUG] FAISS database deserialized in {pd.Timestamp.now() - db_start_time}")
        print(f"[INFO] Vector database loaded successfully")
        print(f"[DEBUG] Total time to load database: {pd.Timestamp.now() - start_time}")
    except Exception as e:
        print(f"[ERROR] Error loading vector database: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return
    
    # Get the LLM client
    print("\n[DEBUG] Initializing LLM client...")
    start_time = pd.Timestamp.now()
    try:
        llm = get_llm_client(
            max_tokens=1024,
            temperature=0.2,
        )
        print(f"[DEBUG] LLM client initialized successfully")
        print(f"[DEBUG] LLM model: {getattr(llm, 'model', 'Unknown')}")
        print(f"[DEBUG] Temperature: 0.2, Max tokens: 1024")
        print(f"[DEBUG] Time taken: {pd.Timestamp.now() - start_time}")
    except Exception as e:
        print(f"[ERROR] Failed to initialize LLM client: {e}")
        import traceback
        print(f"[ERROR] Traceback: {traceback.format_exc()}")
        return
    
    print("\n" + "-" * 100)
    print("[STEP 2] EXTRACTING SENTENCES FROM ARTICLE")
    print("-" * 50)
    
    # Extract sentences from the article
    start_time = pd.Timestamp.now()
    article_text = article_data.get("text", "")
    print(f"[DEBUG] Article text length: {len(article_text)} characters")
    sentences = extract_sentences(article_text, max_sentences)
    
    print(f"\n[INFO] Extracted {len(sentences)} sentences in {pd.Timestamp.now() - start_time}:")
    for i, sentence in enumerate(sentences):
        print(f"  {i+1}. \"{sentence}\"")
    
    print("\n" + "-" * 100)
    print("[STEP 3] FINDING RELATED SENTENCES AND CHECKING CONTRADICTIONS")
    print("-" * 50)
    
    # Find contradictions
    all_contradictions = {}
    contradiction_count = 0
    total_checks = 0
    
    for i, sentence in enumerate(sentences):
        sentence_start_time = pd.Timestamp.now()
        print(f"\n[INFO] Processing sentence {i+1}/{len(sentences)}:")
        print(f"  \"{sentence}\"")
        
        # Find related sentences
        print(f"\n[DEBUG] Finding related sentences...")
        find_start_time = pd.Timestamp.now()
        related_docs = find_related_sentences(db, sentence, current_article_id=article_id)
        
        if not related_docs:
            print("[INFO] No related sentences found for this sentence.")
            print(f"[DEBUG] Time taken: {pd.Timestamp.now() - find_start_time}")
            continue
        
        print(f"[INFO] Found {len(related_docs)} potentially related sentences in {pd.Timestamp.now() - find_start_time}")
        
        # Check for contradictions
        print(f"\n[DEBUG] Checking for contradictions...")
        check_start_time = pd.Timestamp.now()
        contradicting_items = []
        
        for j, doc in enumerate(related_docs):
            check_item_start_time = pd.Timestamp.now()
            print(f"\n[INFO] Checking related sentence {j+1}/{len(related_docs)}:")
            
            # Skip if too short for meaningful comparison
            content_words = len(doc["content"].split())
            if content_words < 5:
                print(f"[DEBUG] Skipping - sentence too short ({content_words} words)")
                continue
            
            # Check for contradiction
            total_checks += 1
            contradiction_start_time = pd.Timestamp.now()
            contradiction_result = check_contradiction(llm, sentence, doc["content"])
            print(f"[DEBUG] Contradiction check took {pd.Timestamp.now() - contradiction_start_time}")
            
            # If contradiction found, add to results
            if contradiction_result.get("contradicts", False):
                contradiction_count += 1
                print(f"[SUCCESS] CONTRADICTION FOUND!")
                print(f"[INFO] Source: \"{sentence[:100]}...\"")
                print(f"[INFO] Contradicts: \"{doc['content'][:100]}...\"")
                print(f"[INFO] Explanation: {contradiction_result.get('explanation', 'No explanation provided')}")
                print(f"[INFO] From article: {doc['metadata'].get('title')}")
                
                contradicting_items.append({
                    "related_sentence": doc["content"],
                    "source": doc["metadata"],
                    "explanation": contradiction_result.get("explanation", "No explanation provided"),
                    "similarity_score": doc["score"]
                })
            else:
                print(f"[INFO] No contradiction found between these sentences")
            
            print(f"[DEBUG] Processed this pair in {pd.Timestamp.now() - check_item_start_time}")
        
        # Store contradictions for this sentence
        if contradicting_items:
            all_contradictions[sentence] = contradicting_items
            print(f"\n[SUCCESS] Found {len(contradicting_items)} contradicting statements for this sentence")
        else:
            print(f"\n[INFO] No contradictions found for this sentence")
        
        print(f"[DEBUG] Total contradiction checks for this sentence: {len(related_docs)}")
        print(f"[DEBUG] Time taken for checking all related sentences: {pd.Timestamp.now() - check_start_time}")
        print(f"[DEBUG] Total time for this sentence: {pd.Timestamp.now() - sentence_start_time}")
    
    print("\n" + "=" * 100)
    print("RESULTS SUMMARY".center(100))
    print("=" * 100)
    
    # Performance stats
    print("\n[STATISTICS]")
    print(f"  Total sentences analyzed: {len(sentences)}")
    print(f"  Total contradiction checks performed: {total_checks}")
    print(f"  Total contradictions found: {contradiction_count}")
    print(f"  Contradiction rate: {contradiction_count/total_checks*100:.2f}% of checks")
    print(f"  Sentences with at least one contradiction: {len(all_contradictions)}")
    print(f"  Percentage of sentences with contradictions: {len(all_contradictions)/len(sentences)*100:.2f}%")
    
    if all_contradictions:
        print(f"\n[SUCCESS] Found contradictions for {len(all_contradictions)} out of {len(sentences)} sentences.")
        
        for i, (source_sentence, contradicting_items) in enumerate(all_contradictions.items()):
            print(f"\n[INFO] Source Statement {i+1}:")
            print(f"  \"{source_sentence}\"")
            print(f"  Contradicting Statements ({len(contradicting_items)} found):")
            
            for j, item in enumerate(contradicting_items):
                print(f"    {j+1}. \"{item['related_sentence'][:100]}...\"")
                print(f"       Published: {item['source'].get('published_at', 'Unknown date')}")
                print(f"       Author: {item['source'].get('author', 'Unknown author')}")
                print(f"       Explanation: {item['explanation']}")
                print(f"       Relevance Score: {item['similarity_score']:.2f}")
    else:
        print("\n[INFO] No contradictions found in any of the analyzed sentences.")
    
    # Save results to a JSON file
    output_file = f"contradictions_{article_id}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_contradictions, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] Results have been saved to {output_file}")
    print("\n" + "=" * 100)
    print("END OF CONTRADICTION ANALYSIS".center(100))
    print("=" * 100)

if __name__ == "__main__":
    print("\n" + "=" * 100)
    print("SARA CONTRADICTION CHECKER DEMO STARTING".center(100))
    print("=" * 100)
    
    # Use an article ID from the demo list in rag/ids.csv
    # First one: "7415582d-2272-44a8-91d2-10fd9ba380ac"
    # Or specify your own article ID from the dataset
    article_id = "7415582d-2272-44a8-91d2-10fd9ba380ac"
    max_sentences = 3
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        article_id = sys.argv[1]
        print(f"[DEBUG] Using article ID from command line: {article_id}")
    
    if len(sys.argv) > 2:
        try:
            max_sentences = int(sys.argv[2])
            print(f"[DEBUG] Using max_sentences from command line: {max_sentences}")
        except ValueError:
            print(f"[WARNING] Invalid max_sentences value '{sys.argv[2]}', using default: {max_sentences}")
    
    print(f"[INFO] Starting contradiction check for article: {article_id}")
    print(f"[INFO] Will analyze up to {max_sentences} sentences")
    print(f"[INFO] Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = time.time()
    
    try:
        run_contradiction_checker(article_id, max_sentences=max_sentences)
        
        elapsed_time = time.time() - start_time
        print(f"\n[INFO] Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
        
        print("\n" + "=" * 100)
        print("CONTRADICTION CHECKER COMPLETED SUCCESSFULLY".center(100))
        print("=" * 100)
        
    except Exception as e:
        print(f"\n[ERROR] An unexpected error occurred: {e}")
        import traceback
        print(f"[ERROR] Traceback:\n{traceback.format_exc()}")
        
        elapsed_time = time.time() - start_time
        print(f"[INFO] Execution failed after {elapsed_time:.2f} seconds")
        
        print("\n" + "=" * 100)
        print("CONTRADICTION CHECKER FAILED".center(100))
        print("=" * 100)