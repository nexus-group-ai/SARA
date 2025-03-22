# SARA - Summarize. Analyze. Retrieve. Annotate

![SARA Logo](img/logo_full.png)

## Overview

SARA is an intelligent interface that transforms historical news archives into an accessible, searchable knowledge base for both readers and journalists. Built during the AIM x MIL Hackathon "Put News Archives to Life" on March 22, 2025, this platform leverages advanced language models with retrieval-augmented generation to provide deeper context, entity understanding, and content transformation features.

## 🎯 Key Features

- **Text Transformation**: Convert articles to different reading levels and styles
- **Smart Summarization**: Generate summaries of varying lengths and perspectives
- **Entity Extraction**: Identify people, organizations, locations, and events
- **Sentiment Analysis**: Analyze the tone and emotional content of articles
- **Topic Identification**: Extract main topics and themes from articles
- **Related Articles**: Find semantically similar content using RAG technology
- **Interactive Dashboard**: Explore article metadata and trends

## 📊 Project Structure

```
├── data/                    # Data directory (not included in repository)
│   ├── articles_clean/      # Cleaned article JSON files 
│   └── metadata.csv         # Article metadata
├── img/                     # Images and logos
├── notebooks/               # Jupyter notebooks for data exploration
├── pages/                   # Streamlit pages
│   └── 01_Explorer.py       # Data exploration dashboard
├── rag/                     # RAG implementation
│   ├── ask_rag.py           # Question answering with RAG
│   ├── create_rag.py        # Vector database creation
│   └── ids.csv              # Demo article IDs
├── src/                     # Source code
│   └── llm.py               # LLM client implementations
├── .env                     # Environment variables (not in repository)
├── Home.py                  # Main Streamlit application
├── Makefile                 # Build and run commands
└── requirements.txt         # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- API keys for LLM services (OpenAI, OpenRouter)
- The Wiener Zeitung dataset (obtain from organizers)

### Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/sara.git
   cd sara
   ```

2. Install dependencies:

   ```bash
   make setup
   ```

   or manually:

   ```bash
   pip install -r requirements.txt
   ```

3. Place your `.env` file with API keys in the project root:

   ```
   OPENAI_API_KEY=your_key_here
   OPENROUTER_API_KEY=your_key_here
   OPENROUTER_API_BASE=https://openrouter.ai/api/v1
   AZURE_OPENAI_API_KEY=your_key_here
   AZURE_OPENAI_ENDPOINT=your_endpoint_here
   MODEL_NAME=openai/gpt-4o
   ```

4. Prepare the data:
   - Download the dataset from the provided source
   - Unzip it in the project root:

     ```bash
     python -m zipfile -e data.zip .
     ```

### Running the application

Start the Streamlit app:

```bash
make run
```

or manually:

```bash
streamlit run Home.py
```

### Building the RAG index

To enable the Related Articles feature, you need to build the vector database:

```bash
make rag
```

or manually:

```bash
python rag/create_rag.py
```

## 💡 Key Components Explained

### Home.py (Main Application)

This is the main entry point for the Streamlit application. It handles:

- **Article Selection**: Filtering and selecting articles from the dataset
- **Text Transformation**: Converting articles to different styles and reading levels
- **Summarization**: Generating different types of article summaries
- **Entity Analysis**: Extracting and classifying named entities
- **Sentiment Analysis**: Analyzing article tone and emotion
- **Topic Analysis**: Identifying main themes and keywords
- **Related Articles**: Finding semantically similar content

Key functions:

- `get_llm_response()`: Sends prompts to the LLM model and handles responses
- `create_transformation_prompt()`: Creates prompts for text style transformation
- `create_summary_prompt()`: Creates prompts for different summary types
- `handle_article_selection()`: Manages the article filtering and selection UI
- `extract_entities()`: Uses LLM to identify and classify entities in text
- `analyze_sentiment()`: Determines the sentiment and tone of articles
- `find_related_articles()`: Uses RAG to find semantically similar articles

### Explorer Dashboard (pages/01_Explorer.py)

An interactive dashboard for exploring the article dataset with:

- Article distribution by time, section, and category
- Word count analysis
- Topic distribution
- Custom filtering and dataset exploration

### RAG Implementation (rag/)

The Retrieval-Augmented Generation system includes:

- **ask_rag.py**: Query the vector database to find related articles
- **create_rag.py**: Build the FAISS vector database from article content

Key components:

- `FAISS.from_documents()`: Creates a vector index from document embeddings
- `RetrievalQA.from_chain_type()`: Builds a question-answering chain using RAG
- `ask_question()`: Main function to query the RAG system

## 📊 Data Structure

The application works with the Wiener Zeitung dataset, which includes:

- **articles_clean/**: JSON files with article content

  ```json
  {
    "id": "unique-id",
    "title": "Article title",
    "author": "Author name",
    "published_at": "2023-01-01",
    "category": "Category",
    "section": "Section",
    "text": "Full article text..."
  }
  ```

- **metadata.csv**: Information about all articles
  - id, filename, published_at, author, title, category, section, word_count
  - Topic tags (financial_crisis, sustainability, fake_news, ai, etc.)

## 🛠️ Technology Stack

- **LLM Integration**: OpenAI GPT-4o, Claude, Gemini via OpenRouter
- **Vector Database**: FAISS for efficient similarity search
- **Backend**: Python/Streamlit
- **Frontend**: Streamlit with interactive visualizations
- **Data Processing**: Pandas, LangChain
- **Visualization**: Plotly

## 🔄 Workflow

1. **Data Selection**: Filter and browse the article archive
2. **Article Analysis**: Extract insights from selected articles
3. **Content Transformation**: Convert articles to different formats
4. **Related Content**: Discover semantically similar articles
5. **Export**: Download transformed content and summaries

## 🙏 Acknowledgments

- AIM (AI Impact Mission) and Media Innovation Lab / Wiener Zeitung for organizing the hackathon
- Contributors to the Wiener Zeitung dataset
- The open-source community behind the libraries used in this project
