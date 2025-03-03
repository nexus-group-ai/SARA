# AIM Hackathon March 2025 - Template
Repository for the AIM Hackathon together with Media Innovation Lab on 22.03.2025

<br>

## Check out OpenAI pricing 
https://openai.com/api/pricing/

<br>

## Set up OpenAI API Key
Copy your teams API key from the [slack]("TODO") channel description and place it in the `.env_template` file.

Don't forget to replace the filename to `.env` afterwards!

Check out the sample code to see how to load the key.

<br>

## About the data
TODO

<br>

## Jump start
### Fork this repository
Simply fork this repository to start working on your project.

### Set up environment
With [uv](https://docs.astral.sh/uv/getting-started/installation/):
```bash
uv sync
```

With [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/windows.html):
```bash
conda create -n aim_hackathon_oct24 python=3.13
pip install -r requirements.txt
```

### Sample code
There is a super simple RAG implementation to help getting you started in [`sample_code.ipynb`](sample_code.ipynb).

<br>

## Hints
Very simple [RAG pipeline](https://medium.com/@ahmed.mohiuddin.architecture/using-ai-to-chat-with-your-documents-leveraging-langchain-faiss-and-openai-3281acfcc4e9) to start with.

You can [extract openAI API token usage](https://help.openai.com/en/articles/6614209-how-do-i-check-my-token-usage) from the response with `response['usage']`.

You can use [tiktoken](https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken) to manually count tokens of a string:
```bash
import tiktoken
tokenizer = tiktoken.get_encoding("o200k_base")  # for gpt 4o
```

[Structured outputs](https://platform.openai.com/docs/guides/structured-outputs/introduction) force the LLM to output e.g. only integers.

<br>