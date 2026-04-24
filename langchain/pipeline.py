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
from core.doc_store import create_index_1, create_index_2
from core.llm_call import llm_1, llm_2
from core.prompt_template import template_1, template_2
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings


load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0.7)


def rag_pipeline(query, file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # embedder = GoogleGenerativeAIEmbeddings(
    #     model="gemini-embedding-2-preview", output_dimensionality=384
    # )
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore.from_documents(
        texts, embedder, index_name=create_index_1()
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
    output = rag_chain | template_1() | llm_1() | parser
    result = output.invoke("What is outcome from proposal?")
    return result


output = rag_pipeline(
    "Waht is blockchain?", "D:\ProdRAG\prodRAG\Blockchain_Course_Proposal.pdf"
)


print(output)
