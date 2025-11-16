# rag_vllm_app.py
"""
RAG + vLLM template (FastAPI)

Features:
- Add / upsert text documents (builds FAISS index)
- Optionally push embeddings to Pinecone (if PINECONE_API_KEY provided)
- Retrieval using FAISS (k-NN)
- Generation via vLLM OpenAI-compatible HTTP API (chat completions)
- Embeddings: OpenAI if OPENAI_API_KEY else sentence-transformers
- Simple in-memory short-term cache for retrieved contexts
"""

import os
import time
import json
import uuid
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import requests
import threading
import logging

# Optional libs
try:
    import faiss
except Exception:
    faiss = None

# sentence-transformers fallback
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# Pinecone optional
try:
    import pinecone
except Exception:
    pinecone = None

# Optional OpenAI python lib for embeddings convenience (we also fallback to requests)
try:
    import openai
except Exception:
    openai = None

# -----------------------
# Configuration (env)
# -----------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")  # vLLM OpenAI-compatible server
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g. "us-west1-gcp"
EMBED_MODEL_OPENAI = os.getenv("EMBED_MODEL_OPENAI", "text-embedding-3-small")
EMBED_MODEL_ST = os.getenv("EMBED_MODEL_ST", "all-MiniLM-L6-v2")
FAISS_DIM = int(os.getenv("FAISS_DIM", "384"))  # will be overwritten when embeddings loaded

# server settings
MAX_RETRIEVE = int(os.getenv("MAX_RETRIEVE", "4"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))  # seconds

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-vllm")

# -----------------------
# Basic models
# -----------------------
class UpsertDoc(BaseModel):
    id: Optional[str]
    text: str
    metadata: Optional[Dict[str, Any]] = None

class QueryReq(BaseModel):
    query: str
    top_k: Optional[int] = MAX_RETRIEVE
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.2

# -----------------------
# Embedding abstraction
# -----------------------
class EmbeddingClient:
    def __init__(self):
        self.use_openai = bool(OPENAI_API_KEY and openai)
        if self.use_openai:
            openai.api_key = OPENAI_API_KEY
            logger.info("Using OpenAI embeddings")
        else:
            if SentenceTransformer:
                logger.info("Using sentence-transformers embeddings")
                self.st = SentenceTransformer(EMBED_MODEL_ST)
            else:
                raise RuntimeError("No embedding backend available: set OPENAI_API_KEY or install sentence-transformers")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            # OpenAI embeddings (batch)
            # using openai.Embedding.create
            resp = openai.Embedding.create(model=EMBED_MODEL_OPENAI, input=texts)
            return [r["embedding"] for r in resp["data"]]
        else:
            arr = self.st.encode(texts, convert_to_numpy=True)
            return arr.tolist()

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

# -----------------------
# Vector store (FAISS) - in-memory
# -----------------------
class FaissStore:
    def __init__(self):
        self.index = None
        self.id_to_doc = []  # list of dicts {id, text, metadata}
        self.dim = None
        self.lock = threading.Lock()

    def build(self, embeddings: np.ndarray, docs: List[Dict[str, Any]]):
        with self.lock:
            dim = embeddings.shape[1]
            self.dim = dim
            if faiss is None:
                raise RuntimeError("faiss not installed")
            self.index = faiss.IndexFlatL2(dim)
            self.index.add(embeddings.astype("float32"))
            self.id_to_doc = docs.copy()
            logger.info(f"Built FAISS index with {len(docs)} docs (dim={dim})")

    def add(self, embedding: np.ndarray, doc: Dict[str, Any]):
        with self.lock:
            if self.index is None:
                # initialize
                self.dim = embedding.shape[0]
                if faiss is None:
                    raise RuntimeError("faiss not installed")
                self.index = faiss.IndexFlatL2(self.dim)
            self.index.add(np.array([embedding]).astype("float32"))
            self.id_to_doc.append(doc)

    def search(self, query_emb: np.ndarray, top_k: int = 4):
        with self.lock:
            if self.index is None or len(self.id_to_doc) == 0:
                return []
            D, I = self.index.search(np.array([query_emb]).astype("float32"), top_k)
            results = []
            for idx, dist in zip(I[0], D[0]):
                if idx < 0 or idx >= len(self.id_to_doc):
                    continue
                d = self.id_to_doc[int(idx)]
                results.append({"doc": d, "score": float(dist)})
            return results

# -----------------------
# Pinecone optional wrapper
# -----------------------
class PineconeStore:
    def __init__(self, api_key: str, environment: str, index_name: str = "rag-index"):
        if not pinecone:
            raise RuntimeError("pinecone-client not installed")
        pinecone.init(api_key=api_key, environment=environment)
        self.index_name = index_name
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=FAISS_DIM)
        self.index = pinecone.Index(index_name)

    def upsert(self, id: str, vector: List[float], metadata: Dict[str, Any]):
        self.index.upsert([(id, vector, metadata)])

    def query(self, vector: List[float], top_k: int = 4):
        res = self.index.query(vector=vector, top_k=top_k, include_metadata=True)
        matches = []
        for m in res["matches"]:
            matches.append({"doc": {"id": m["id"], "text": m["metadata"].get("text", ""), "metadata": m["metadata"]}, "score": m["score"]})
        return matches

# -----------------------
# RAG service
# -----------------------
emb_client = EmbeddingClient()
faiss_store = FaissStore()
pinecone_store = None
if PINECONE_API_KEY and pinecone:
    try:
        pinecone_store = PineconeStore(PINECONE_API_KEY, PINECONE_ENV or "us-west1-gcp")
        logger.info("Pinecone initialized")
    except Exception as e:
        logger.warning(f"Pinecone init failed: {e}")

# Simple short-term cache for retrieved contexts
cache: Dict[str, Dict[str, Any]] = {}  # key -> {"expires": ts, "context": [...]}

def cache_set(k: str, value: Any, ttl: int = CACHE_TTL):
    cache[k] = {"expires": time.time() + ttl, "value": value}

def cache_get(k: str):
    v = cache.get(k)
    if not v:
        return None
    if v["expires"] < time.time():
        cache.pop(k, None)
        return None
    return v["value"]

# -----------------------
# Helper: call vLLM (OpenAI-compatible chat completions)
# -----------------------
def call_vllm_chat(prompt: str, max_tokens: int = 256, temperature: float = 0.2, model: str = "meta-llama/Llama-3-8b-instruct") -> str:
    """
    Sends a chat completion request to vLLM OpenAI-compatible API endpoint.
    Using the /v1/chat/completions path.
    """
    url = VLLM_BASE_URL.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    try:
        resp = requests.post(url, json=payload, timeout=60)
        resp.raise_for_status()
        j = resp.json()
        # openai-compatible structure
        content = j["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        logger.exception("vLLM call failed")
        raise HTTPException(status_code=500, detail=f"vLLM call failed: {e}")

# -----------------------
# FastAPI app
# -----------------------
app = FastAPI(title="RAG + vLLM (template)")

@app.post("/docs/upsert")
def upsert_docs(docs: List[UpsertDoc]):
    """
    Upsert documents: compute embeddings and add to FAISS (and optionally Pinecone).
    """
    items = []
    texts = [d.text for d in docs]
    ids = [d.id or str(uuid.uuid4()) for d in docs]
    embeddings = emb_client.embed_texts(texts)
    for _id, txt, emb, meta in zip(ids, texts, embeddings, docs):
        doc = {"id": _id, "text": txt, "metadata": meta.metadata or {}}
        items.append(doc)
        # add to FAISS
        faiss_store.add(np.array(emb, dtype="float32"), doc)
        # push to pinecone if configured
        if pinecone_store:
            pinecone_store.upsert(_id, emb, {"text": txt, **(meta.metadata or {})})
    return {"status": "ok", "inserted": len(items)}

@app.post("/docs/bulk_build")
def build_index_from_texts(payload: dict):
    """
    Build new FAISS index from a list of {"id","text","metadata"}.
    Useful for initial indexing.
    """
    docs = payload.get("docs") or []
    if not docs:
        raise HTTPException(status_code=400, detail="docs required")
    texts = [d["text"] for d in docs]
    embeddings = emb_client.embed_texts(texts)
    emb_arr = np.array(embeddings, dtype="float32")
    faiss_store.build(emb_arr, docs)
    return {"status": "ok", "count": len(docs)}

@app.post("/query")
def query_rag(req: QueryReq):
    """
    Retrieval -> Generate:
    1. Embed query
    2. Retrieve top_k docs from FAISS (or Pinecone)
    3. Build prompt with context
    4. Call vLLM for final answer
    """
    top_k = req.top_k or MAX_RETRIEVE
    cache_key = f"q:{req.query}:{top_k}"
    cached = cache_get(cache_key)
    if cached:
        retrieved = cached
    else:
        q_emb = np.array(emb_client.embed_text(req.query), dtype="float32")
        if pinecone_store:
            retrieved = pinecone_store.query(q_emb.tolist(), top_k=top_k)
        else:
            retrieved = faiss_store.search(q_emb, top_k=top_k)
        cache_set(cache_key, retrieved)
    # prepare context string
    context_parts = []
    for r in retrieved:
        d = r["doc"]
        txt = d.get("text") if isinstance(d, dict) else d
        context_parts.append(f"- {txt[:1000]}")
    context = "\n".join(context_parts) if context_parts else "No relevant documents found."

    # prompt composition (very simple; modify per use-case)
    prompt = f"""Use the following documents to answer the user's question. Be concise and include sources when relevant.

Context:
{context}

Question:
{req.query}

Answer:
"""
    answer = call_vllm_chat(prompt=prompt, max_tokens=req.max_tokens, temperature=req.temperature)
    return {"answer": answer, "retrieved": retrieved}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "vllm_base_url": VLLM_BASE_URL,
        "has_openai_key": bool(OPENAI_API_KEY),
        "faiss_ready": faiss_store.index is not None
    }

# optional endpoint to load sample docs
@app.post("/docs/sample_load")
def load_sample():
    docs = [
        {"id": "doc1", "text": "Kubernetes uses pods to encapsulate containers.", "metadata": {"source": "k8s-doc"}},
        {"id": "doc2", "text": "Node.js uses an event loop and libuv for async I/O.", "metadata": {"source": "node-doc"}},
        {"id": "doc3", "text": "FAISS is a library for vector similarity search.", "metadata": {"source": "faiss"}}
    ]
    texts = [d["text"] for d in docs]
    embeddings = emb_client.embed_texts(texts)
    emb_arr = np.array(embeddings, dtype="float32")
    faiss_store.build(emb_arr, docs)
    return {"status": "ok", "count": len(docs)}

# -----------------------
# Run with: uvicorn rag_vllm_app:app --host 0.0.0.0 --port 8010
# -----------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
