import os
import sys
from pathlib import Path
from typing import TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()

# No embedding API key is needed. This model runs locally.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Chroma persists its vector index in this local folder.
vectorstore = Chroma(
    collection_name="rag_documents",
    persist_directory="./chroma_db",
    embedding_function=embeddings,
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Use a model available to your Groq free-tier account.
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0,
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """Answer using only the supplied context.
If the context does not contain the answer, say you don't know.
Do not invent facts. Mention the source filenames and page numbers when useful.

Context:
{context}""",
    ),
    ("human", "{question}"),
])


def ingest_pdf(pdf_path: str) -> None:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {path}")

    pages = PyPDFLoader(str(path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)

    # Keep source names useful in the answer and trace.
    for chunk in chunks:
        chunk.metadata["source"] = path.name

    vectorstore.add_documents(chunks)
    print(f"Indexed {len(chunks)} chunks from {path.name}")


class RAGState(TypedDict):
    question: str
    context: str
    sources: list[str]
    answer: str


def retrieve_node(state: RAGState) -> dict:
    docs = retriever.invoke(state["question"])

    context_parts = []
    sources = []

    for doc in docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", "unknown source")
        page_label = f", page {page + 1}" if isinstance(page, int) else ""

        context_parts.append(
            f"[Source: {source}{page_label}]\n{doc.page_content}"
        )
        sources.append(f"{source}{page_label}")

    return {
        "context": "\n\n".join(context_parts),
        "sources": list(dict.fromkeys(sources)),
    }


def generate_node(state: RAGState) -> dict:
    messages = prompt.invoke({
        "context": state["context"],
        "question": state["question"],
    })
    response = llm.invoke(messages)
    return {"answer": response.content}


builder = StateGraph(RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("generate", generate_node)
builder.add_edge(START, "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)

rag_graph = builder.compile()


def ask(question: str) -> None:
    result = rag_graph.invoke({
        "question": question,
        "context": "",
        "sources": [],
        "answer": "",
    })

    print("\nAnswer:")
    print(result["answer"])

    if result["sources"]:
        print("\nSources:")
        for source in result["sources"]:
            print(f"- {source}")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "ingest":
        ingest_pdf(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "ask":
        ask(" ".join(sys.argv[2:]))
    else:
        print("Usage:")
        print('  python rag_app.py ingest "data/LeavePolicy.pdf"')
        print('  python rag_app.py ask "What is the leave policy?"')