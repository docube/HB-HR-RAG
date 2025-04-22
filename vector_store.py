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
    
    faiss.write_index(index, FAISS_INDEX_PATH)

    with open(METADATA_STORE_PATH, "w") as f:
        json.dump(metadatas, f, indent=2)

def load_faiss_index():
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(METADATA_STORE_PATH) as f:
        metadatas = json.load(f)
    return index, metadatas
