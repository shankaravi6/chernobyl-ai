# ==============================
# app.py  –  Chernobyl RAG Chat (PDF + TXT Support)
# ==============================
# Run with:  streamlit run app.py
# ==============================

import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# ---------- CONFIG ----------
DATA_FILE  = "data/Chernobyl-Facts.pdf"   # Change to .pdf or .txt
DB_DIR     = "./chernobyl_db"                 # Local Chroma folder
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
TOP_K      = 3
# ----------------------------

@st.cache_resource
def get_vectorstore():
    if not os.path.exists(DATA_FILE):
        st.error(f"File `{DATA_FILE}` not found. Please add it to the `data/` folder.")
        st.stop()

    # --- Auto-detect file type ---
    file_ext = os.path.splitext(DATA_FILE)[1].lower()
    if file_ext not in [".txt", ".pdf"]:
        st.error(f"Unsupported file type: `{file_ext}`. Use `.txt` or `.pdf`.")
        st.stop()

    # --- Load document based on type ---
    if file_ext == ".txt":
        loader = TextLoader(DATA_FILE, encoding="utf-8")
    elif file_ext == ".pdf":
        loader = PyPDFLoader(DATA_FILE)

    docs = loader.load()

    # --- Split into chunks ---
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # --- Initialize embeddings ---
    embedder = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # --- Load or create vector DB ---
    if os.path.exists(DB_DIR):
        db = Chroma(persist_directory=DB_DIR, embedding_function=embedder)
        st.success("Loaded existing knowledge base.")
    else:
        with st.spinner("Creating knowledge base from PDF/TXT..."):
            db = Chroma.from_documents(chunks, embedder, persist_directory=DB_DIR)
            db.persist()
        st.success("Created new knowledge base.")

    return db

def build_rag_chain(db):
    retriever = db.as_retriever(search_kwargs={"k": TOP_K})
    llm = OllamaLLM(model=OLLAMA_MODEL, temperature=0.1)

    prompt = """You are an expert on the Chernobyl disaster. Answer **only** using the provided context.  
If the context does not contain the answer, say “I don’t have that information in my source.”

Context:
{context}

Question: {question}
Answer:"""

    PROMPT = PromptTemplate(template=prompt, input_variables=["context", "question"])

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )
    return qa

# ---------- UI ----------
st.set_page_config(page_title="Chernobyl AI", layout="centered")
st.title("Chernobyl Disaster Q&A")
st.caption(f"Answers are grounded **only** in `{os.path.basename(DATA_FILE)}`")

# Load vector DB and RAG chain
vector_db = get_vectorstore()
rag_chain = build_rag_chain(vector_db)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_q := st.chat_input("Ask about Chernobyl…"):
    st.session_state.messages.append({"role": "user", "content": user_q})
    with st.chat_message("user"):
        st.markdown(user_q)

    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            result = rag_chain({"query": user_q})
            answer = result["result"].strip()
            sources = result["source_documents"]

        st.markdown(answer)

        # Show sources in expandable section
        if sources:
            with st.expander("Sources (from document)"):
                for i, doc in enumerate(sources, 1):
                    st.caption(f"Chunk {i}")
                    st.code(doc.page_content.strip(), language=None)

    st.session_state.messages.append({"role": "assistant", "content": answer})