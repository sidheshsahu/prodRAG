from pinecone import ServerlessSpec, Pinecone
import os


def create_index():
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
    return index
