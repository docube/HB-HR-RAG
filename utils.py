# utils.py

import os
import json
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path

# Path to where uploaded docs live
UPLOADS_DIR = Path("uploads")
VECTOR_DB_DIR = Path("vector_store")

def ingest_files():
    all_docs = []

    for file in os.listdir(UPLOADS_DIR):
        file_path = UPLOADS_DIR / file

        if file.endswith(".pdf"):
            loader = PyPDFLoader(str(file_path))
        elif file.endswith(".txt"):
            loader = TextLoader(str(file_path))
        else:
            continue  # skip unsupported types for now

        docs = loader.load()

        # Attach metadata from sidecar JSON
        meta_path = str(file_path) + ".meta.json"
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                metadata = json.load(f)
                for d in docs:
                    d.metadata.update(metadata)

        all_docs.extend(docs)

    # Chunk docs
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(all_docs)

    # Create embeddings
    embeddings = OpenAIEmbeddings()  # or HuggingFaceEmbeddings() if you prefer

    # Vector Store
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local(str(VECTOR_DB_DIR))

    print(f"[INFO] Ingested and saved {len(chunks)} chunks to vector store.")
