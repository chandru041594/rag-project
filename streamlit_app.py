import os
import streamlit as st

# Streamlit Cloud secrets are read here and passed to your existing rag_app.py.
if "GROQ_API_KEY" in st.secrets:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]

if "LANGSMITH_API_KEY" in st.secrets:
    os.environ["LANGSMITH_API_KEY"] = st.secrets["LANGSMITH_API_KEY"]
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = st.secrets.get(
        "LANGSMITH_PROJECT", "free-rag-demo"
    )

from rag_app import rag_graph, ingest_pdf, vectorstore

st.set_page_config(page_title="Leave Policy RAG", page_icon="📄")
st.title("Ask the Leave Policy")
st.caption("Answers are generated from the indexed policy document.")

# Chroma's index is local to the deployed app. Build it if this is a fresh start.
if vectorstore._collection.count() == 0:
    with st.spinner("Preparing the policy index..."):
        ingest_pdf("data/LeavePolicy.pdf")

question = st.text_input("Your question")

if st.button("Ask") and question.strip():
    with st.spinner("Searching the policy and preparing an answer..."):
        result = rag_graph.invoke({
            "question": question,
            "context": "",
            "sources": [],
            "answer": "",
        })

    st.markdown(result["answer"])

    if result.get("sources"):
        st.caption("Sources: " + "; ".join(result["sources"]))