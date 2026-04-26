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
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from core.doc_store import create_index_1
from core.llm_call import llm_1
from core.prompt_template import template_1
from langchain_core.messages import ToolMessage


load_dotenv()


class RAGState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    file_path: str


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def rag_retriever(query: str, file_path: str):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = splitter.split_documents(documents)

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = PineconeVectorStore.from_documents(
        texts, embedder, index_name=create_index_1()
    )
    # Adding retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 3}
    )
    return retriever


@tool
def rag_tool(query: str, file_path: str) -> str:
    """Retrieves relevant context from the PDF."""
    retriever = rag_retriever(query, file_path)
    docs = retriever.invoke(query)
    return format_docs(docs)


tools = [rag_tool]


def custom_tool_node(state: RAGState):
    messages = state["messages"]
    file_path = state["file_path"]
    last_message = messages[-1]
    tool_results = []
    for tool_call in last_message.tool_calls:
        tool_call["args"]["file_path"] = file_path

        result = rag_tool.invoke(tool_call["args"])

        tool_results.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

    return {"messages": tool_results}


def chat_rag(state: RAGState):
    llm = llm_1()
    llm_with_tools = llm.bind_tools(tools)

    system_prompt = """
    You are a helpful assistant.
    Use rag_tool when the user asks factual questions.
    Always answer only on retrieved context.
    If context is insufficient, say: "I don't know based on the provided context."
    """
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def build_workflow():
    workflow = StateGraph(RAGState)

    workflow.add_node("chat_rag", chat_rag)
    workflow.add_node("tools", custom_tool_node)

    workflow.add_edge(START, "chat_rag")
    workflow.add_conditional_edges(
        "chat_rag", tools_condition, {"tools": "tools", END: END}
    )
    workflow.add_edge("tools", "chat_rag")

    return workflow.compile()


def run_rag(query: str, file_path: str):
    app = build_workflow()
    result = app.invoke(
        {"messages": [HumanMessage(content=query)], "file_path": file_path}
    )
    print(result)
    return result["messages"][-1].content


if __name__ == "__main__":
    answer = run_rag(
        query="What is future scope of proposal?",
        file_path=r"D:\ProdRAG\prodRAG\Blockchain_Course_Proposal.pdf",
    )
    print("\n FINAL ANSWER:\n", answer)
