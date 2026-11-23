# Pinoy Recipe RAG

A retrieval-augmented generation (RAG) system that recommends Filipino recipes from the ingredients a user has. Queries can be written in English, Tagalog, or Taglish.

CMSC 197 mini-project.

## Pipeline

query -> embed -> retrieve top-k recipes -> prompt -> local LLM (Ollama)

- Dataset: [FILIPINO_RECIPES_2K](https://huggingface.co/datasets/joackimagno/FILIPINO_RECIPES_2K)
- Embeddings: `intfloat/multilingual-e5-small` (sentence-transformers)
- LLM: `llama3.2:3b` through Ollama (temperature 0, fixed seed)

## Setup (local)

1. Install [Ollama](https://ollama.com/download) and run `ollama pull llama3.2:3b`.
2. Create and activate a virtual environment:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Install the packages:
   ```
   pip install -r requirements.txt
   ```
4. Open `notebooks/pipeline.ipynb` and run the cells from the top.

## Team

Borces, Del Rosario, Lansoy
