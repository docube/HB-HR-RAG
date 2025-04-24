import os
from utils import load_file
from embedder import embed_texts
from vector_store import save_faiss_index
from config import UPLOAD_DIR, VECTOR_DB_DIR
import uuid

def chunk_text(text, chunk_size=500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

def ingest_files():
    all_texts = []
    all_metadata = []

    for filename in os.listdir(UPLOAD_DIR):
        filepath = os.path.join(UPLOAD_DIR, filename)
        text = load_file(filepath)
        chunks = chunk_text(text)
        all_texts.extend(chunks)
        all_metadata.extend([
            {
                "doc_id": str(uuid.uuid4()),
                "filename": filename,
                "chunk_index": i,
                "text_preview": chunk[:100]
            } for i, chunk in enumerate(chunks)
        ])

    embeddings = embed_texts(all_texts)
    save_faiss_index(embeddings, all_metadata)

# 👇👇👇 Add this to allow running directly
if __name__ == "__main__":
    ingest_files()
