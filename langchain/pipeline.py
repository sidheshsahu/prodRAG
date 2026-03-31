import os
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone import ServerlessSpec,Pinecone
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableLambda
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)


pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "practice" 

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        serverless=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)



file_path = "D:\ProdRAG\prodRAG\Blockchain_Course_Proposal.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()
print(type(loader))


text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",output_dimensionality=384)


vectorstore = PineconeVectorStore.from_documents(
    texts,
    embedder,
    index_name=index_name
)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
query = "What is the module4?"
results = retriever.invoke(query)



rag_prompt = PromptTemplate.from_template(
   template= """You are a helpful assistant.
Use only the provided context to answer the question.
If the answer is not present in the context, say: "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:"""
)



rag_chain=RunnableParallel({
    "question" :RunnablePassthrough(),
    "context" :retriever| RunnableLambda(format_docs)
}
)

parser=StrOutputParser()


output=rag_chain | rag_prompt | llm | parser

output.invoke("What is 4th module?")
