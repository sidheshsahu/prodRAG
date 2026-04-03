from langgraph.graph import StateGraph, START, END, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from core.doc_store import create_index_1
from core.llm_call import llm_1

load_dotenv()


# ---------------- STATE ----------------
class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    file_path: str


# ---------------- UTIL ----------------
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# ---------------- RETRIEVER (NO GLOBAL STATE) ----------------
def rag_retriever(query: str, file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embedder = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview", output_dimensionality=384
    )

    vectorstore = PineconeVectorStore.from_documents(
        texts, embedder, index_name=create_index_1
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )

    return retriever


# ---------------- TOOL (DYNAMIC INPUT FIXED) ----------------
@tool
def rag_tool(state: dict) -> str:
    """
    Retrieves relevant context from PDF based on latest user query.
    """
    query = state["query"]
    file_path = state["file_path"]

    retriever = rag_retriever(query, file_path)
    docs = retriever.invoke(query)

    return format_docs(docs)


tools = [rag_tool]
tool_node = ToolNode(tools)

llm = llm_1()
llm_with_tools = llm.bind_tools(tools)


# ---------------- CHAT NODE ----------------
def chat_rag(state: RAGState):
    system_prompt = """
You are a helpful assistant.

Use rag_tool when needed.
Always base answers only on retrieved context.
If context is insufficient, say: "I don't know based on the provided context."
"""

    messages = [SystemMessage(content=system_prompt)] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ---------------- WORKFLOW ----------------
def build_workflow():
    workflow = StateGraph(RAGState)

    workflow.add_node("chat_rag", chat_rag)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "chat_rag")

    workflow.add_conditional_edges(
        "chat_rag", tools_condition, {"tools": "tools", END: END}
    )

    workflow.add_edge("tools", "chat_rag")

    return workflow.compile()


# ---------------- FINAL RUN FUNCTION ----------------
def run_rag(query: str, file_path: str):
    app = build_workflow()

    result = app.invoke(
        {"messages": [HumanMessage(content=query)], "file_path": file_path}
    )

    return result["messages"][-1].content


# ---------------- TEST CALL (YOUR "LAST CALL") ----------------
if __name__ == "__main__":
    answer = run_rag(
        query="What are the main topics covered in the document?",
        file_path="D:\ProdRAG\prodRAG\Blockchain_Course_Proposal.pdf",
    )

    print("\n🔹 FINAL ANSWER:\n", answer)
