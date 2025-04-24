import faiss
import numpy as np
import json
import os

FAISS_INDEX_PATH = "data/index/faiss.index"
METADATA_STORE_PATH = "data/index/metadata.json"

def save_faiss_index(embeddings: np.ndarray, metadatas: list[dict]):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    # ✅ Ensure the directory exists
    os.makedirs(os.path.dirname(FAISS_INDEX_PATH), exist_ok=True)

    # Save the FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)

    # Save the metadata
    with open(METADATA_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(metadatas, f, indent=2)

    print(f"[INFO] FAISS index saved to {FAISS_INDEX_PATH}")
    print(f"[INFO] Metadata saved to {METADATA_STORE_PATH}")

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH, encoding="utf-8") as f:
        metadatas = json.load(f)
    return index, metadatas
