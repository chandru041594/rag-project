import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Load .env for local use. On Streamlit Cloud, use App settings → Secrets.
load_dotenv()

if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if "LANGSMITH_API_KEY" in st.secrets:
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = st.secrets.get(
        "LANGSMITH_PROJECT", "free-rag-demo"
    )

if not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is missing. Add it to your local .env file "
        "or to Streamlit Cloud → Manage app → Settings → Secrets."
    )
    st.stop()

# Import only after loading the API key because rag_app initializes ChatGroq.
from rag_app import rag_graph, ingest_pdf, vectorstore


st.set_page_config(page_title="Leave Policy RAG", page_icon="📄")
st.title("Ask the Leave Policy")
st.caption("Answers are generated from the indexed policy document.")

pdf_path = Path("data/LeavePolicy.pdf")

if not pdf_path.exists():
    st.error(f"Policy PDF not found: {pdf_path}")
    st.stop()

# Create the local Chroma index when the app starts with an empty database.
if vectorstore._collection.count() == 0:
    with st.spinner("Preparing the policy index..."):
        ingest_pdf(str(pdf_path))

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Searching the policy and preparing an answer..."):
        result = rag_graph.invoke({
            "question": question.strip(),
            "context": "",
            "sources": [],
            "answer": "",
        })

    st.markdown(result["answer"])

    sources = result.get("sources", [])
    if sources:
        st.caption("Sources: " + "; ".join(sources))