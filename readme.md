# ProdRAG

ProdRAG is an experimental Retrieval-Augmented Generation (RAG) comparison project.

This repository was built to explore multiple RAG frameworks side by side, understand how their pipelines differ, and evaluate tracing and observability workflows. It is not positioned as a production-ready system yet.

## What This Repo Covers

- LangChain-based RAG pipeline
- LangGraph-based agentic RAG flow
- Haystack-based RAG pipeline
- Streamlit UI to run each backend separately
- Pinecone as the vector store
- PDF ingestion and chunking
- LLM and embedding integrations using Groq and Google Gemini
- Tracing and observability setup with Langfuse and LangSmith

## Why This Exists

The goal of this project is to explore the RAG ecosystem in practice:

- compare framework ergonomics
- test end-to-end PDF question answering
- understand retrieval + generation flow differences
- experiment with tracing, monitoring, and observability
- create a base that can later be hardened for production

## Current State

This project is currently a learning and experimentation workspace.

It includes working pieces for:

- document loading from PDF
- chunking and embedding
- vector storage in Pinecone
- retrieval-based answering
- framework-level experimentation
- tracing configuration for observability

It is not yet fully productionized. Some code paths are still exploratory, and a few modules contain hardcoded values or demo-oriented logic.

## Project Structure

```text
prodRAG/
|-- app.py                  # Streamlit comparison UI
|-- core/
|   |-- doc_store.py        # Pinecone index/document store helpers
|   |-- llm_call.py         # LLM setup helpers
|   |-- prompt_template.py  # Shared prompt templates
|-- Langchain/
|   |-- pipeline.py         # LangChain RAG pipeline
|   |-- langchain.ipynb     # Notebook exploration
|-- Langgraph/
|   |-- pipeline.py         # LangGraph workflow-based RAG
|   |-- langgraph.ipynb     # Notebook exploration
|-- Haystack/
|   |-- pipeline.py         # Haystack RAG pipeline
|   |-- haystack.ipynb      # Notebook exploration
|-- pyproject.toml          # Project dependencies
|-- .env                    # Local environment variables
```

## Frameworks Explored

### 1. LangChain

The LangChain pipeline focuses on a more traditional RAG flow:

- load PDF
- split content into chunks
- generate embeddings
- store vectors in Pinecone
- retrieve relevant chunks
- send retrieved context to the LLM

### 2. LangGraph

The LangGraph version explores an agent-style workflow where:

- the graph maintains message state
- a retrieval tool is invoked when needed
- the answer is generated from retrieved context

This is useful for understanding how graph-driven orchestration differs from a standard chain.

### 3. Haystack

The Haystack pipeline explores component-based orchestration:

- PDF conversion
- document splitting
- embedding
- Pinecone retrieval
- prompt building
- generation
- tracing through Langfuse connector integration

## UI

The Streamlit app in [app.py](/d:/ProdRAG/prodRAG/app.py) lets you:

- enter a question
- upload a PDF
- run each backend independently
- compare responses across LangChain, LangGraph, and Haystack

Run it with:

```bash
streamlit run app.py
```

## Setup

### 1. Create environment

This project uses Python 3.11+.

If you are using `uv`:

```bash
uv sync
```

Or with `pip`:

```bash
pip install -e .
```

### 2. Configure environment variables

Create a local `.env` file with the required API keys.

Typical services used in this repo include:

- `GROQ_API_KEY`
- `PINECONE_API_KEY`
- `GEMINI_API_KEY`
- `LANGFUSE_SECRET_KEY`
- `LANGFUSE_PUBLIC_KEY`
- `LANGFUSE_BASE_URL`
- `LANGSMITH_TRACING`
- `LANGSMITH_ENDPOINT`
- `LANGSMITH_API_KEY`
- `LANGSMITH_PROJECT`
- `HAYSTACK_CONTENT_TRACING_ENABLED`

Do not commit real credentials.

## Observability And Tracing

One of the main outcomes of this repo is the observability setup.

The project already experiments with:

- Langfuse for tracing and monitoring
- LangSmith tracing for LangChain and LangGraph workflows
- Haystack tracing integration

This makes the repository useful not only for comparing frameworks, but also for understanding how to inspect:

- prompt execution
- retrieval steps
- model calls
- overall pipeline behavior

## Notes And Limitations

A few things to keep in mind:

- this repo is exploratory, not production-ready
- some pipelines use hardcoded sample questions internally
- indexing happens during execution, which is convenient for demos but not ideal for production
- there is no persistent ingestion workflow yet
- there is no evaluation harness yet
- error handling and config management are still lightweight

## Production Direction

If this project is moved toward production, the next improvements would likely be:

- separate ingestion from query-time retrieval
- remove hardcoded test queries
- centralize configuration
- add structured logging and evaluation
- add caching and index lifecycle management
- improve test coverage
- define a single production backend after comparison

## Key Learning Outcome

This repository is best understood as a comparative RAG lab.

It helped explore how different frameworks approach the same problem while also validating tracing and observability patterns that are important for real-world RAG systems.

## Status

Experimental / exploratory project with observability setup in place.
