import os
import re
import glob
from pathlib import Path

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.embeddings import get_embedding_function


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
DB_DIR = PROJECT_ROOT / "chroma_store"


# 1. LOAD
# Read each transcript and remove VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line == "WEBVTT" or "-->" in line:
                    continue
                lines.append(line)

        text = " ".join(lines)
        match = re.search(r"Session[ _]*(\d+)", path)
        session = match.group(1) if match else "unknown"
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "session": session,
                    "source": os.path.basename(path)
                }
            )
        )

    return docs


# 2. EMBEDDING FUNCTION
# see embeddings.py

# 3. BUILD VECTOR STORE
def load_store():

    embedding_function = get_embedding_function()
    # If database already exists, reuse it
    if os.path.exists(DB_DIR):
        return Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embedding_function,
        )
    # Load documents
    docs = load_transcripts()

    # Split documents
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
    ).split_documents(docs)

    # Create vector database
    return Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=str(DB_DIR),
    )


# 4. RETRIEVER
def build_retriever():
    return load_store().as_retriever(
        search_kwargs={"k": 5}
    )


# 5. TEST
if __name__ == "__main__":

    retriever = build_retriever()
    results = retriever.invoke(
        "what is regression testing?"
    )
    for r in results:
        print(
            f"[Session {r.metadata['session']}] "
            f"{r.page_content[:150]}...\n"
        )