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
        "Use the section for the backend you want to query. Each backend has its own query and PDF upload form."
    )

    cols = st.columns(3)

    for backend_name, col in zip(BACKENDS.keys(), cols):
        with col:
            st.subheader(backend_name)
            with st.form(key=f"form_{backend_name}"):
                query = st.text_input("Query", key=f"query_{backend_name}")
                uploaded_file = st.file_uploader(
                    "Upload a PDF document",
                    type=["pdf"],
                    accept_multiple_files=False,
                    key=f"upload_{backend_name}",
                )
                submit = st.form_submit_button("Run")

                if submit:
                    if not query:
                        st.error("Please enter a query.")
                    elif uploaded_file is None:
                        st.error("Please upload a PDF document.")
                    else:
                        with st.spinner(
                            f"Running {backend_name}... this may take a moment."
                        ):
                            try:
                                file_path = save_uploaded_pdf(uploaded_file)
                                answer = run_pipeline(backend_name, query, file_path)
                                st.success("Answer generated")
                                st.write(answer)
                            except Exception as exc:
                                st.error(f"Pipeline error: {exc}")


if __name__ == "__main__":
    main()
