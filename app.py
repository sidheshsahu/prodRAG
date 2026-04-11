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


def init_results_state():
    if "backend_results" not in st.session_state:
        st.session_state.backend_results = {
            name: {"answer": "", "error": ""} for name in BACKENDS
        }


def main():
    st.set_page_config(page_title="ProdRAG Streamlit", page_icon="📄")
    st.title("ProdRAG RAG Query Interface")
    st.markdown(
        "Enter your query and upload a PDF, then run each backend individually."
    )
    init_results_state()

    query = st.text_input(
        "Query",
        placeholder="Enter your question here",
        key="query_input",
    )
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type=["pdf"],
        key="pdf_uploader",
        help="Choose a PDF from your local machine.",
    )
    resolved_file_path = save_uploaded_pdf(uploaded_file) if uploaded_file else ""
    if uploaded_file:
        st.caption(f"Using uploaded file: `{uploaded_file.name}`")

    cols = st.columns(3)

    for backend_name, col in zip(BACKENDS.keys(), cols):
        with col:
            with st.container(border=True):
                st.subheader(backend_name)
                if st.button(f"Run {backend_name}", key=f"run_{backend_name}"):
                    if not query:
                        st.session_state.backend_results[backend_name] = {
                            "answer": "",
                            "error": "Please enter a query.",
                        }
                    elif not resolved_file_path:
                        st.session_state.backend_results[backend_name] = {
                            "answer": "",
                            "error": "Please upload a PDF.",
                        }
                    else:
                        with st.spinner(
                            f"Running {backend_name}... this may take a moment."
                        ):
                            try:
                                answer = run_pipeline(
                                    backend_name, query, resolved_file_path
                                )
                                st.session_state.backend_results[backend_name] = {
                                    "answer": answer,
                                    "error": "",
                                }
                            except Exception as exc:
                                st.session_state.backend_results[backend_name] = {
                                    "answer": "",
                                    "error": f"Pipeline error: {exc}",
                                }

                result = st.session_state.backend_results[backend_name]
                if result["error"]:
                    st.error(result["error"])
                elif result["answer"]:
                    st.success("Answer generated")
                    st.write(result["answer"])
                else:
                    st.info("No answer yet. Click Run.")


if __name__ == "__main__":
    main()
