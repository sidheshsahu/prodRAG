from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from pinecone import ServerlessSpec, Pinecone
import os


def create_index_1():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "practice-rag"

    if not pc.has_index(index_name):
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            serverless=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    index = pc.Index(index_name)
    return index_name


def create_index_2():
    document_store = PineconeDocumentStore(
        index="random",
        metric="cosine",
        dimension=384,
        spec={"serverless": {"region": "us-east-1", "cloud": "aws"}},
    )
    return document_store
