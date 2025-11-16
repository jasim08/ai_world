# Contract Reviewer — RAG + vLLM Fullstack Sample Application

# **Problem statement (complex):**

# Create an application that helps legal teams review, search, and ask questions about a corpus of contracts. Features:

# * Upload multiple contract documents (PDF / TXT / DOCX)
# * Build a vector index (FAISS) from document chunks and embeddings
# * Query the index using natural language; retrieved context must be shown with source
# * Use vLLM (OpenAI-compatible API) to generate final answers that reference the retrieved passages
# * Short-term chat memory per session and a document browser UI
# * Production-ready concerns: background indexing, caching, simple auth token, Dockerfile

# This repository contains a minimal but complete single-file backend and a single-file React frontend to run locally.

# ---

# ## Files included (in this document)

# * `backend/app.py` — FastAPI backend (RAG, embedding, FAISS, vLLM client)
# * `frontend/` — React single-file app (App.jsx) + package.json
# * `requirements.txt` — Python dependencies
# * `Dockerfile` — minimal multi-stage image for backend
# * `README` — run instructions (below)

# ---

# ## Backend: `backend/app.py`

# ```python
# # backend/app.py
"""
Contract Reviewer RAG + vLLM sample
- FastAPI app
- Upsert documents endpoint (text file upload)
- Build FAISS index with sentence-transformers (fallback to OpenAI embeddings)
- /query endpoint to run retrieval and call vLLM (OpenAI-compatible).
- Simple in-memory short-term cache and per-session chat memory

NOTE: For production, replace in-memory stores with Redis, secure file storage, and authenticated uploads.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, uuid, time, io
from typing import List, Dict, Any
import numpy as np
import tempfile
import logging

# Optional libs
try:
    import faiss
except Exception:
    faiss = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import openai
except Exception:
    openai = None

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("contract-reviewer")

app = FastAPI(title="Contract Reviewer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Config via env
# -------------------------
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL_ST = os.getenv("EMBED_MODEL_ST", "all-MiniLM-L6-v2")
CACHE_TTL = int(os.getenv("CACHE_TTL", "300"))

# -------------------------
# Simple in-memory stores
# -------------------------
DOC_STORE: List[Dict[str, Any]] = []  # {id, filename, chunks: [{text, start, end}], metadata}
FAISS_INDEX = None
INDEX_DIM = None

# short-term cache
CACHE: Dict[str, Dict[str, Any]] = {}

# per-session chat history
CHAT_MEMORY: Dict[str, List[Dict[str,str]]] = {}

# -------------------------
# Utilities
# -------------------------
class QueryReq(BaseModel):
    session_id: str
    query: str
    top_k: int = 4
    max_tokens: int = 256


def cache_set(k, v, ttl=CACHE_TTL):
    CACHE[k] = {"value": v, "expires": time.time() + ttl}


def cache_get(k):
    v = CACHE.get(k)
    if not v:
        return None
    if v["expires"] < time.time():
        CACHE.pop(k, None)
        return None
    return v["value"]

# -------------------------
# Embedding client abstraction
# -------------------------
class EmbClient:
    def __init__(self):
        self.use_openai = bool(OPENAI_API_KEY and openai)
        if self.use_openai:
            openai.api_key = OPENAI_API_KEY
            logger.info("Using OpenAI embeddings")
        else:
            if SentenceTransformer:
                logger.info("Using sentence-transformers")
                self.st = SentenceTransformer(EMBED_MODEL_ST)
            else:
                raise RuntimeError("No embedding backend available. Set OPENAI_API_KEY or install sentence-transformers")

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if self.use_openai:
            resp = openai.Embedding.create(model="text-embedding-3-small", input=texts)
            return [d["embedding"] for d in resp["data"]]
        else:
            arr = self.st.encode(texts, convert_to_numpy=True)
            return arr.tolist()

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

EMB = EmbClient()

# -------------------------
# Simple tokenizer/chunker for documents
# -------------------------

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks

# -------------------------
# FAISS helpers
# -------------------------

def build_faiss(embeddings: np.ndarray, docs: List[Dict[str,Any]]):
    global FAISS_INDEX, INDEX_DIM
    if faiss is None:
        raise RuntimeError("faiss not installed")
    dim = embeddings.shape[1]
    INDEX_DIM = dim
    idx = faiss.IndexFlatL2(dim)
    idx.add(embeddings.astype('float32'))
    FAISS_INDEX = idx
    logger.info(f"Built FAISS index with {len(docs)} chunks (dim={dim})")


def faiss_search(query_emb: np.ndarray, top_k: int=4):
    if FAISS_INDEX is None or len(DOC_STORE)==0:
        return []
    D, I = FAISS_INDEX.search(np.array([query_emb]).astype('float32'), top_k)
    res = []
    for idx, dist in zip(I[0], D[0]):
        if idx < 0 or idx >= sum(len(d['chunks']) for d in DOC_STORE):
            continue
        # locate the chunk by iterating counts
        acc=0
        for d in DOC_STORE:
            if idx < acc + len(d['chunks']):
                chunk = d['chunks'][idx-acc]
                res.append({"doc_id": d['id'], "filename": d['filename'], "text": chunk, "score": float(dist)})
                break
            acc += len(d['chunks'])
    return res

# -------------------------
# vLLM call (OpenAI-compatible chat completions)
# -------------------------

def call_vllm_chat(prompt: str, max_tokens: int = 256, temperature: float = 0.2, model: str = "meta-llama/Llama-3-8b-instruct") -> str:
    url = VLLM_BASE_URL.rstrip('/') + '/v1/chat/completions'
    payload = {
        "model": model,
        "messages": [
            {"role":"system","content":"You are a concise contract review assistant. When you cite evidence, prefix with [source:id]."},
            {"role":"user","content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    try:
        r = requests.post(url, json=payload, timeout=60)
        r.raise_for_status()
        j = r.json()
        return j['choices'][0]['message']['content']
    except Exception as e:
        logger.exception("vLLM call failed")
        raise HTTPException(status_code=500, detail=f"vLLM call failed: {e}")

# -------------------------
# Endpoints
# -------------------------
@app.post('/docs/upload')
async def upload_doc(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        text = contents.decode('utf-8')
    except Exception:
        # fallback: save and attempt text extraction (not implemented here)
        text = ''
    if not text:
        raise HTTPException(status_code=400, detail='Empty or unsupported file')
    doc_id = str(uuid.uuid4())
    chunks = chunk_text(text)
    DOC_STORE.append({"id": doc_id, "filename": file.filename, "chunks": chunks, "raw": text})
    # build embeddings for chunks and update FAISS (simple rebuild)
    all_chunks = [c for d in DOC_STORE for c in d['chunks']]
    embeddings = np.array(EMB.embed_texts(all_chunks), dtype='float32')
    build_faiss(embeddings, DOC_STORE)
    return {"status": "ok", "id": doc_id, "chunks": len(chunks)}

@app.post('/query')
def query(req: QueryReq):
    # basic session memory
    if req.session_id not in CHAT_MEMORY:
        CHAT_MEMORY[req.session_id] = []
    # cache
    cache_key = f"q:{req.query}:{req.top_k}"
    cached = cache_get(cache_key)
    if cached:
        retrieved = cached
    else:
        q_emb = np.array(EMB.embed_text(req.query), dtype='float32')
        retrieved = faiss_search(q_emb, top_k=req.top_k)
        cache_set(cache_key, retrieved)
    context = "\n\n".join([f"[source:{r['doc_id']}] {r['text']}" for r in retrieved]) or "No relevant documents found."
    prompt = f"Use the documents below to answer the question. Be concise. Include short citations like [source:id].\n\nDocuments:\n{context}\n\nQuestion: {req.query}\nAnswer:"
    answer = call_vllm_chat(prompt, max_tokens=req.max_tokens)
    # append to session memory
    CHAT_MEMORY[req.session_id].append({"question": req.query, "answer": answer})
    return {"answer": answer, "retrieved": retrieved}

@app.get('/docs')
def list_docs():
    return [{"id": d['id'], "filename": d['filename'], "chunks": len(d['chunks'])} for d in DOC_STORE]

@app.get('/health')
def health():
    return {"status": "ok", "docs": len(DOC_STORE), "faiss_ready": FAISS_INDEX is not None}

# Run with: uvicorn backend.app:app --reload --port 8001
# ```


# ## Frontend: React single-file app (`frontend/src/App.jsx`)

# ```jsx
# # // frontend/src/App.jsx --separte file 
# import React, {useState, useEffect, useRef} from 'react';

# function App(){
#   const [files, setFiles] = useState([]);
#   const [query, setQuery] = useState("");
#   const [sessionId, setSessionId] = useState(() => localStorage.getItem('session_id') || Math.random().toString(36).slice(2));
#   const [messages, setMessages] = useState([]);
#   const [docs, setDocs] = useState([]);
#   const fileRef = useRef();

#   useEffect(()=>{ localStorage.setItem('session_id', sessionId); fetchDocs(); }, []);

#   async function fetchDocs(){
#     const r = await fetch('http://localhost:8001/docs');
#     const j = await r.json(); setDocs(j);
#   }

#   async function upload(){
#     const f = fileRef.current.files[0]; if(!f) return alert('choose file');
#     const fd = new FormData(); fd.append('file', f);
#     const r = await fetch('http://localhost:8001/docs/upload', {method:'POST', body: fd});
#     const j = await r.json(); console.log(j); fetchDocs();
#   }

#   async function ask(e){
#     e.preventDefault();
#     setMessages(m=>[...m, {role:'user', text: query}]);
#     const r = await fetch('http://localhost:8001/query', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({session_id: sessionId, query, top_k
# # ```
