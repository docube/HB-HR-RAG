# utils.py

import os
import json
import hashlib
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()


# Paths
UPLOADS_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vector_store")
CACHE_DIR = Path("cache")

# Ensure cache directory exists
CACHE_DIR.mkdir(exist_ok=True)

# Initialize embedding model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Logging
def log(msg):
    print(f"[INFO] {msg}")

def load_file_hash(filepath):
    """Generate a unique hash of a file's contents for caching."""
    with open(filepath, "rb") as f:
        file_bytes = f.read()
    return hashlib.md5(file_bytes).hexdigest()

def is_cached(file_hash):
    return (CACHE_DIR / f"{file_hash}.done").exists()

def mark_cached(file_hash):
    (CACHE_DIR / f"{file_hash}.done").touch()

def load_file(filepath):
    """Safely load text content."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            return f.read()

def ingest_files():
    all_docs = []

    log("🚀 Starting document ingestion...")

    for file in os.listdir(UPLOADS_DIR):
        file_path = UPLOADS_DIR / file
        if not file_path.suffix.lower() in {".pdf", ".txt", "docx"}:
            log(f"⚠️ Unsupported file type skipped: {file}")
            continue

        file_hash = load_file_hash(file_path)
        if is_cached(file_hash):
            log(f"✅ Skipping already-ingested file: {file}")
            continue

        # Load document
        if file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
        elif file_path.suffix == ".docx":
            loader = UnstructuredWordDocumentLoader(str(file_path))
        else:
            loader = TextLoader(str(file_path))
        docs = loader.load()

        # Attach metadata
        meta_path = file_path.with_suffix(file_path.suffix + ".json")
        if meta_path.exists():
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                for d in docs:
                    d.metadata.update(metadata)

        all_docs.extend(docs)
        mark_cached(file_hash)

    if not all_docs:
        log("⚠️ No new documents to ingest.")
        return

    log(f"📄 Loaded {len(all_docs)} documents. Splitting into chunks...")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # Embed and save
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_DB_DIR))

    log(f"✅ Ingested and saved {len(chunks)} chunks to vector store.")
