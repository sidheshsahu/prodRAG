import os
from haystack import Pipeline
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pathlib import Path
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.components.embedders.google_genai import (
    GoogleGenAIDocumentEmbedder,
    GoogleGenAITextEmbedder,
)
from dotenv import load_dotenv
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

load_dotenv()

converter = PyPDFToDocument()


pdf_path = r"D:\ProdRAG\prodRAG\Blockchain_Course_Proposal.pdf"

docs = converter.run(sources=[pdf_path])["documents"]

splitter = DocumentSplitter(
    split_by="word",
    split_length=10,
    split_overlap=0,
)

docs_split = splitter.run(documents=docs)["documents"]


document_store = PineconeDocumentStore(
    index="practice",
    metric="cosine",
    dimension=3072,
    spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
)


embedder = GoogleGenAIDocumentEmbedder(api="gemini")

documents_with_embeddings = embedder.run(docs_split)["documents"]


document_store.write_documents(documents_with_embeddings)

prompt_template = """
According to the contents:
{% for document in documents %}
{{document.content}}
{% endfor %}
Answer the given question: {{query}}
Answer:
"""

prompt_builder = PromptBuilder(template=prompt_template)


llm = OpenAIGenerator(
    api_key=Secret.from_env_var("GROQ_API_KEY"),
    api_base_url="https://api.groq.com/openai/v1",
    model="llama-3.1-8b-instant",
    generation_kwargs={"max_tokens": 512},
)


query_pipeline = Pipeline()
query_pipeline.add_component("tracer", LangfuseConnector("Basic RAG Pipeline"))
query_pipeline.add_component("prompt", prompt_builder)
query_pipeline.add_component("llm", llm)
query_pipeline.add_component("text_embedder", GoogleGenAITextEmbedder())
query_pipeline.add_component(
    "retriever", PineconeEmbeddingRetriever(document_store=document_store)
)
query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
query_pipeline.connect("retriever.documents", "prompt.documents")
query_pipeline.connect("prompt", "llm")
query = "What are the main topics covered in the document?"
result = query_pipeline.run(
    {"text_embedder": {"text": query}, "prompt": {"query": query}}
)


print(result["llm"]["replies"][0])
