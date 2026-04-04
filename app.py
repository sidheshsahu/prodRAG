import tempfile
from pathlib import Path
import streamlit as st
from Haystack.pipeline import rag_pipeline as rag_pipeline_haystack
from Langchain.pipeline import rag_pipeline as rag_pipeline_langchain
from Langgraph.pipeline import run_rag as rag_pipeline_langgraph

BACKENDS = {
    "Langchain": rag_pipeline_langchain,
    "Langgraph": rag_pipeline_langgraph,
    "Haystack": rag_pipeline_haystack,
}


def save_uploaded_pdf(uploaded_file):
    temp_path = Path(tempfile.mkdtemp()) / uploaded_file.name
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return str(temp_path)


def run_pipeline(backend_name: str, query: str, file_path: str):
    pipeline = BACKENDS[backend_name]
    if backend_name == "Langgraph":
        return pipeline(query=query, file_path=file_path)
    return (
        pipeline(query=query, pdf_path=file_path)
        if backend_name == "Haystack"
        else pipeline(query, file_path)
    )


def main():
    st.set_page_config(page_title="ProdRAG Streamlit", page_icon="📄")
    st.title("ProdRAG RAG Query Interface")
    st.markdown(
        "Use this app to ask questions over a PDF document with Langchain, Langgraph, or Haystack."
    )

    backend = st.selectbox("Select backend", list(BACKENDS.keys()))
    query = st.text_input("Query")
    uploaded_file = st.file_uploader(
        "Upload a PDF document", type=["pdf"], accept_multiple_files=False
    )

    if st.button("Run"):
        if not query:
            st.error("Please enter a query.")
            return
        if uploaded_file is None:
            st.error("Please upload a PDF document.")
            return

        with st.spinner(f"Running {backend}... this may take a moment."):
            try:
                file_path = save_uploaded_pdf(uploaded_file)
                answer = run_pipeline(backend, query, file_path)
                st.success("Answer generated")
                st.subheader("Result")
                st.write(answer)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")


if __name__ == "__main__":
    main()
