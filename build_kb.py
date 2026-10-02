import faiss
import redis
import os
import json
import fitz
import numpy as np
from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.Redis(host="localhost", port=6379, db=0)

def extract_text_from_pdf(path):
    import fitz
    doc = fitz.open(path)
    return "\n".join(page.get_text() for page in doc)

def extract_text_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chunks = []
    for item in data:
        if isinstance(item, dict):
            for k, v in item.items():
                chunks.append(f"{k}: {v}")
        elif isinstance(item, str):
            chunks.append(item)
    return "\n".join(chunks)

def chunk_text(text, max_words=200, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words - overlap):
        chunks.append(' '.join(words[i:i + max_words]))
    return chunks

def build_index(file_paths, faiss_file="kb.index"):
    all_chunks = []
    for path in file_paths:
        if path.endswith(".pdf"):
            text = extract_text_from_pdf(path)
        elif path.endswith(".json"):
            text = extract_text_from_json(path)
        else:
            continue
        all_chunks.extend(chunk_text(text))

    embeddings = embedder.encode(all_chunks)
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Save to FAISS index file
    faiss.write_index(index, faiss_file)

    # Save text and embeddings to Redis
    for i, chunk in enumerate(all_chunks):
        redis_client.set(f"kb:text:{i}", chunk)
        redis_client.set(f"kb:embedding:{i}", embeddings[i].tobytes())

    print(f"Indexed {len(all_chunks)} chunks into FAISS and Redis.")

if __name__ == "__main__":
    file_paths = [
        "pdfs/Airtel_DTH_Troubleshooting_Guide.pdf",
        "pdfs/airtel_dth.pdf",
        "pdfs/channels.pdf",
        "pdfs/package_deets.pdf",
        "json/airtel_dth_faqs.json"
    ]
    build_index(file_paths)
