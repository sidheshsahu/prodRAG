from langgraph.graph import StateGraph,START, END,add_messages
from langchain_core.messages import BaseMessage,HumanMessage
from typing import TypedDict,Annotated,List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec,Pinecone
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import os
from langgraph.prebuilt import ToolNode,tools_condition
from langfuse import get_client
from langfuse.langchain import CallbackHandler

load_dotenv()

langfuse = get_client()
langfuse_handler = CallbackHandler()


file_path = r"D:\ProdRAG\prodRAG\Blockchain_Course_Proposal.pdf"
loader = PyPDFLoader(file_path)
documents = loader.load()


pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = "practice-rag" 

if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        serverless=ServerlessSpec(cloud="aws", region="us-east-1"),
    )

index = pc.Index(index_name)

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

llm = ChatGroq(
    model_name="llama-3.1-8b-instant",
    temperature=0.7
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embedder = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview",output_dimensionality=384)


vectorstore = PineconeVectorStore.from_documents(
    texts,
    embedder,
    index_name=index_name
)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})


@tool
def rag_tool(query: str) -> dict:
    '''A tool that takes a user query, retrieves relevant information from the PDF, and returns an answer based on the retrieved context.
    Use this tool when the user asks factual/conceptual questions that might be answered from the stored documents.'''
    
    # Retrieve similar chunks from the vectorstore
    docs = retriever.invoke(query)
    context = format_docs(docs)
    
    # Generate answer using the context
    rag_chain = RunnableParallel({
        "question": RunnablePassthrough(),
        "context": RunnableLambda(lambda x: context)
    })
    
    output = (rag_chain | rag_prompt | llm | StrOutputParser()).invoke(query)
    
    return {
        "query": query,
        "output": output
    }


tools=[rag_tool]
llm_with_tools=llm.bind_tools(tools)

tool_node=ToolNode(tools)

class RAGState(TypedDict):
    messages:Annotated[List[BaseMessage],add_messages]

def chat_rag(state:RAGState):
    response=llm_with_tools.invoke(state["messages"])
    return {"messages":[response]}


workflow=StateGraph(RAGState)
workflow.add_node("chat_rag",chat_rag)
workflow.add_node("tools", tool_node)
workflow.add_edge(START, "chat_rag")
workflow.add_conditional_edges(
    "chat_rag",
    tools_condition,
    {"tools": "tools",END: END}

)
workflow.add_edge("tools", "chat_rag")      

app=workflow.compile()


result=app.invoke({
"messages":[HumanMessage(content=("What is objectives?"))]
    },
    config={"callbacks": [langfuse_handler]})


print(result["messages"][-1].content)
