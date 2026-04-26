import os
from haystack import Pipeline
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from core.doc_store import create_index_1, create_index_2
from core.prompt_template import template_1, template_2
from pathlib import Path
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack_integrations.components.embedders.google_genai import (
    GoogleGenAIDocumentEmbedder,
    GoogleGenAITextEmbedder,
)
from haystack.components.embedders import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)
from dotenv import load_dotenv
from haystack_integrations.components.retrievers.pinecone import (
    PineconeEmbeddingRetriever,
)
from core.llm_call import llm_1, llm_2
from haystack.components.generators import OpenAIGenerator
from haystack.utils import Secret
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.connectors.langfuse import LangfuseConnector

load_dotenv()


def rag_pipeline(query, pdf_path):
    converter = PyPDFToDocument()
    docs = converter.run(sources=[pdf_path])["documents"]

    splitter = DocumentSplitter(
        split_by="word",
        split_length=500,
        split_overlap=50,
    )

    docs_split = splitter.run(documents=docs)["documents"]

    # embedder = GoogleGenAIDocumentEmbedder(api="gemini")
    embedder = SentenceTransformersDocumentEmbedder(
        model="sentence-transformers/all-MiniLM-L6-v2"
    )

    documents_with_embeddings = embedder.run(docs_split)["documents"]

    document_store = create_index_2()
    document_store.write_documents(documents_with_embeddings)

    prompt_builder = PromptBuilder(template=template_2())

    query_pipeline = Pipeline()
    query_pipeline.add_component("tracer", LangfuseConnector("Basic RAG Pipeline"))
    query_pipeline.add_component("prompt", prompt_builder)
    query_pipeline.add_component("llm", llm_2())
    query_pipeline.add_component(
        "text_embedder",
        SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        ),
    )
    query_pipeline.add_component(
        "retriever", PineconeEmbeddingRetriever(document_store=document_store)
    )
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt.documents")
    query_pipeline.connect("prompt", "llm")
    query = "Explain the summary of each modules?"
    result = query_pipeline.run(
        {"text_embedder": {"text": query}, "prompt": {"query": query}}
    )
    return result["llm"]["replies"][0]


res = rag_pipeline(
    "Explain the summary of each modules?", r"D:\ProdRAG\prodRAG\example.pdf"
)
print(res)
