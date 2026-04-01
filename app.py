from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import streamlit as st


def _load_local_pipeline(module_label: str, file_path: Path):
    spec = spec_from_file_location(module_label, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {file_path}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_DIR = Path(__file__).resolve().parent
langchain_module = _load_local_pipeline("local_langchain_pipeline", BASE_DIR / "langchain" / "pipeline.py")
haystack_module = _load_local_pipeline("local_haystack_pipeline", BASE_DIR / "haystack" / "pipeline.py")

run_langchain_rag = langchain_module.run_langchain_rag
run_haystack_rag = haystack_module.run_haystack_rag


st.set_page_config(page_title="RAG Playground", page_icon=":books:", layout="wide")
st.title("RAG with LangChain and Haystack")
st.caption("Upload a document, ask a question, and compare responses from both pipelines.")

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "txt", "md", "csv", "json", "doc", "docx"],
)
question = st.text_input("Ask a question about the uploaded document")
run_clicked = st.button("Run RAG")

if run_clicked:
    if uploaded_file is None:
        st.error("Please upload a document first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        file_name = uploaded_file.name
        file_bytes = uploaded_file.getvalue()

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("LangChain Output")
            with st.spinner("Running LangChain pipeline..."):
                try:
                    langchain_answer = run_langchain_rag(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        question=question,
                    )
                    st.write(langchain_answer)
                except Exception as exc:
                    st.error(f"LangChain pipeline failed: {exc}")

        with col2:
            st.subheader("Haystack Output")
            with st.spinner("Running Haystack pipeline..."):
                try:
                    haystack_answer = run_haystack_rag(
                        file_name=file_name,
                        file_bytes=file_bytes,
                        question=question,
                    )
                    st.write(haystack_answer)
                except Exception as exc:
                    st.error(f"Haystack pipeline failed: {exc}")
