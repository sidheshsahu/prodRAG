import os
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import ServerlessSpec, Pinecone
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
)
from core.doc_store import create_index
from core.prompt_template import template
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)


def rag_pipeline(query, file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embedder = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", output_dimensionality=384
    )

    vectorstore = PineconeVectorStore.from_documents(
        texts, embedder, index_name=create_index()
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    rag_chain = RunnableParallel(
        {
            "question": RunnablePassthrough(),
            "context": retriever | RunnableLambda(format_docs),
        }
    )

    parser = StrOutputParser()
    output = rag_chain | template() | llm | parser
    result = output.invoke("What is outcome from proposal?")
    return result
