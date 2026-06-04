import streamlit as st
import requests
import json

API_URL = "http://localhost:8001"

st.set_page_config(
    page_title="RAG Document Assistant",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 RAG Document Assistant")
st.markdown(
    "Ask questions about any PDF document using "
    "**Retrieval-Augmented Generation** with a locally hosted AI model."
)

# Sidebar
with st.sidebar:
    st.header("System Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.status_code == 200:
            info = r.json()
            st.success("API Online")
            st.markdown(f"**LLM:** {info['model']}")
            st.markdown(f"**Embeddings:** {info['embedding_model']}")
            st.markdown(f"**Docs indexed:** {info['documents_indexed']}")
        else:
            st.error("API Error")
    except Exception:
        st.error("API Offline")
        st.markdown("Start backend:\n```\npython backend/main.py\n```")

    st.divider()
    st.header("How it works")
    st.markdown("""
1. **Upload** a PDF document
2. **Index** — text is chunked and embedded locally
3. **Ask** a question in natural language
4. **Retrieve** — relevant chunks found via FAISS vector search
5. **Generate** — LLM answers using only retrieved context
6. **Sources** — see exactly which parts of the document were used

All processing is local. No data leaves your machine.
    """)

    st.divider()
    st.header("Architecture")
    st.markdown("""

PDF → PyMuPDF → Chunks
↓
SentenceTransformer
(all-MiniLM-L6-v2)
↓
FAISS Index
↓
Query → Retrieval
↓
Ollama (llama3.2)
↓
Grounded Answer

    """)

# Tabs
tab1, tab2 = st.tabs(["Ask a Question", "Index Document for Repeated Use"])

# Tab 1: Direct query
with tab1:
    st.subheader("Upload a PDF and ask a question")
    st.markdown("The document will be processed each time. Use the Index tab for repeated queries.")

    col1, col2 = st.columns([2, 1])

    with col1:
        uploaded_file = st.file_uploader(
            "Choose a PDF",
            type="pdf",
            key="query_upload"
        )

    with col2:
        question = st.text_area(
            "Your question",
            placeholder="What is the date of the incident?\nWho are the parties involved?\nWhat damages are claimed?",
            height=120
        )

    if uploaded_file and question:
        if st.button("Get Answer", type="primary", use_container_width=True):
            with st.spinner("Chunking, embedding, retrieving, and generating answer..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    data = {"question": question}
                    response = requests.post(
                        f"{API_URL}/query",
                        files=files,
                        data=data,
                        timeout=120
                    )

                    if response.status_code == 200:
                        result = response.json()

                        st.success(f"Answer generated using {result['context_chunks_used']} relevant chunks from {result['total_chunks_in_document']} total.")

                        st.subheader("Answer")
                        st.info(result["answer"])

                        st.subheader("Sources Used")
                        for source in result["sources"]:
                            with st.expander(f"Source {source['chunk_index']} — Relevance score: {source['relevance_score']}"):
                                st.markdown(source["text"])

                        with st.expander("View Full JSON Response"):
                            st.json(result)

                    else:
                        st.error(f"Error: {response.json().get('detail', 'Unknown error')}")

                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Start it first.")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    else:
        if not uploaded_file:
            st.info("Upload a PDF to get started.")
        elif not question:
            st.info("Enter a question about the document.")

# Tab 2: Index then query
with tab2:
    st.subheader("Index a document once, query it many times")

    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Step 1: Index your document**")
        index_file = st.file_uploader("Choose a PDF to index", type="pdf", key="index_upload")

        if index_file:
            if st.button("Index Document", type="primary"):
                with st.spinner("Indexing document..."):
                    try:
                        files = {"file": (index_file.name, index_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{API_URL}/index", files=files, timeout=120)

                        if response.status_code == 200:
                            result = response.json()
                            st.success(
                                f"Indexed successfully. "
                                f"Doc ID: **{result['doc_id']}** | "
                                f"{result['total_chunks']} chunks created."
                            )
                            st.session_state["last_doc_id"] = result["doc_id"]
                        else:
                            st.error(response.json().get("detail", "Indexing failed"))
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    with col4:
        st.markdown("**Step 2: Query the indexed document**")

        try:
            r = requests.get(f"{API_URL}/indexed-documents", timeout=3)
            if r.status_code == 200:
                docs = r.json()["documents"]
                if docs:
                    doc_options = {d["filename"]: d["doc_id"] for d in docs}
                    selected_doc = st.selectbox("Select indexed document", list(doc_options.keys()))
                    doc_id = doc_options[selected_doc]

                    indexed_question = st.text_area(
                        "Your question",
                        placeholder="Ask anything about this document...",
                        height=100,
                        key="indexed_q"
                    )

                    if indexed_question:
                        if st.button("Query Indexed Document", type="primary"):
                            with st.spinner("Retrieving and generating answer..."):
                                try:
                                    data = {"doc_id": doc_id, "question": indexed_question}
                                    response = requests.post(
                                        f"{API_URL}/query-indexed",
                                        data=data,
                                        timeout=60
                                    )

                                    if response.status_code == 200:
                                        result = response.json()
                                        st.subheader("Answer")
                                        st.info(result["answer"])
                                        for source in result["sources"]:
                                            with st.expander(f"Source {source['chunk_index']}"):
                                                st.markdown(source["text"])
                                    else:
                                        st.error(response.json().get("detail", "Query failed"))
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")
                else:
                    st.info("No documents indexed yet. Index one using Step 1.")
        except Exception:
            st.warning("Could not fetch indexed documents. Is the backend running?")