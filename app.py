import sys, os, warnings, shutil
warnings.filterwarnings("ignore")
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from rag.loader import load_pdf
from rag.embedder import (chunk_documents, build_vector_store,
                          load_vector_store, deduplicate_chunks,
                          filter_toc_pages)
from rag.chain import build_qa_chain, ask

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="centered")
st.title("📄 Document Q&A")
st.caption("Upload a PDF and ask questions about it.")

# Session state
if "chain"         not in st.session_state: st.session_state.chain         = None
if "retriever"     not in st.session_state: st.session_state.retriever     = None
if "chat_history"  not in st.session_state: st.session_state.chat_history  = []
if "doc_name"      not in st.session_state: st.session_state.doc_name      = None

import hashlib

def get_collection_name(filename):
    """Each PDF gets its own isolated ChromaDB collection."""
    safe = hashlib.md5(filename.encode()).hexdigest()[:16]
    return f"doc_{safe}"

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file:
        save_path = os.path.join("uploads", uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        if st.session_state.doc_name != uploaded_file.name:
            col_name   = get_collection_name(uploaded_file.name)
            index_flag = os.path.join("chroma_db", f".indexed_{col_name}")
            already_indexed = os.path.exists(index_flag)

            if already_indexed:
                with st.spinner("Loading existing index from disk..."):
                    try:
                        vectordb = load_vector_store(
                            "./chroma_db", collection_name=col_name
                        )
                        chain, retriever = build_qa_chain(vectordb)
                        st.session_state.chain        = chain
                        st.session_state.retriever    = retriever
                        st.session_state.doc_name     = uploaded_file.name
                        st.session_state.chat_history = []
                        st.success(f"✓ Loaded '{uploaded_file.name}'")
                    except Exception as e:
                        st.error(f"Failed to load: {e}")
            else:
                with st.spinner("Reading and indexing your PDF..."):
                    try:
                        pages   = load_pdf(save_path)
                        pages   = filter_toc_pages(pages)
                        chunks  = chunk_documents(pages)
                        chunks  = deduplicate_chunks(chunks)
                        vectordb = build_vector_store(
                            chunks,
                            persist_directory="./chroma_db",
                            collection_name=col_name
                        )
                        chain, retriever = build_qa_chain(vectordb)
                        st.session_state.chain        = chain
                        st.session_state.retriever    = retriever
                        st.session_state.doc_name     = uploaded_file.name
                        st.session_state.chat_history = []

                        with open(index_flag, "w") as f:
                            f.write(uploaded_file.name)

                        st.success(
                            f"✓ Indexed {len(chunks)} chunks "
                            f"from '{uploaded_file.name}'"
                        )
                    except Exception as e:
                        st.error(f"Failed to process PDF: {e}")

    if st.session_state.doc_name:
        st.info(f"Active document:\n{st.session_state.doc_name}")
    else:
        st.warning("No document loaded yet.")

    if st.button("Clear & Start Over"):
        st.session_state.chain        = None
        st.session_state.retriever    = None
        st.session_state.chat_history = []
        st.session_state.doc_name     = None
        st.rerun()

# Main chat area
if st.session_state.chain is None:
    st.info("Upload a PDF in the sidebar to get started.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View source chunks used"):
                for i, doc in enumerate(message["sources"]):
                    st.markdown(
                        f"**Chunk {i+1}** — "
                        f"Page {doc.metadata.get('page', '?') + 1}"
                    )
                    st.caption(doc.page_content)
                    st.divider()

    if question := st.chat_input("Ask a question about your document..."):
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = ask(
                        st.session_state.chain,
                        st.session_state.retriever,
                        question
                    )
                    st.markdown(answer)
                except Exception as e:
                    answer  = f"Error: {e}"
                    sources = []
                    st.error(answer)

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": sources
        })

        if sources:
            with st.expander("View source chunks used"):
                for i, doc in enumerate(sources):
                    st.markdown(
                        f"**Chunk {i+1}** — "
                        f"Page {doc.metadata.get('page', '?') + 1}"
                    )
                    st.caption(doc.page_content)
                    st.divider()