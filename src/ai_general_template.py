"""
Full Python example application demonstrating different types of LLM usage patterns
and model classes you can use for real projects.

File: python_llm_models_app.py

Features included (each exposed as a FastAPI endpoint):
- Prompting / Direct completion (OpenAI API example)
- HuggingFace Transformers inference (local or remote models)
- Embeddings (OpenAI / SentenceTransformers)
- RAG (FAISS vector store + embeddings + HF / OpenAI generator)
- LoRA fine-tuning helper (example training function using PEFT)
- Example config via environment variables

Notes:
- This is a demo app: replace model names, API keys, and paths with your values.
- It intentionally avoids heavy training inside the web server. The LoRA function is
  a script you run offline (included as a callable function).

Requirements (pip):
fastapi uvicorn openai transformers accelerate torch sentence-transformers faiss-cpu
peft datasets scikit-learn pydantic python-dotenv

Run:
1) pip install -r requirements.txt
2) export OPENAI_API_KEY=... (or create a .env)
3) uvicorn python_llm_models_app:app --reload --port 8000

"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
from typing import List, Optional

# Optional: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
HF_TOKEN = os.environ.get("HF_TOKEN")  # optional for private HF models

app = FastAPI(title="LLM Models Playground")

# -------------------------------
# Request / Response Models
# -------------------------------
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 256

class EmbedRequest(BaseModel):
    texts: List[str]

class RAGQuery(BaseModel):
    query: str
    top_k: int = 4

# -------------------------------
# 1) OpenAI Prompting (completion/chat) example
# -------------------------------
@app.post("/generate_openai")
async def generate_openai(req: GenerateRequest):
    """Generate text using OpenAI (chat/completion).
    This demonstrates using a hosted API model as a black box.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")

    try:
        import openai
        openai.api_key = OPENAI_API_KEY

        # Use chat completions if available
        # If you prefer `openai.Completion.create`, change below accordingly
        resp = openai.ChatCompletion.create(
            model="gpt-4o-mini",  # example; replace with available model
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": req.prompt},
            ],
            max_tokens=req.max_tokens,
            temperature=0.2,
        )
        content = resp['choices'][0]['message']['content']
        return {"text": content, "raw": resp}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 2) HuggingFace Transformers inference
# -------------------------------
@app.post("/generate_hf")
async def generate_hf(req: GenerateRequest, model_name: Optional[str] = "gpt2"):
    """Generate text using a HuggingFace model (causal LM).
    Model name examples: 'gpt2', 'facebook/opt-350m', 'tiiuae/falcon-7b-instruct' (if you have access)
    For large models, you may need GPU and accelerate config.
    """
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(model_name)

        inputs = tokenizer(req.prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=req.max_tokens, do_sample=True, top_k=50)
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HF error: {e}")

# -------------------------------
# 3) Embeddings example (OpenAI + SentenceTransformers)
# -------------------------------
@app.post("/embed_openai")
async def embed_openai(req: EmbedRequest):
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set")
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        # Using OpenAI embeddings endpoint
        resp = openai.Embedding.create(model="text-embedding-3-small", input=req.texts)
        vectors = [item['embedding'] for item in resp['data']]
        return {"embeddings_shape": [len(vectors), len(vectors[0])], "vectors": vectors}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed_st")
async def embed_sentence_transformers(req: EmbedRequest, model_name: Optional[str] = "all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        vectors = model.encode(req.texts, show_progress_bar=False)
        return {"embeddings_shape": [len(vectors), len(vectors[0])], "vectors": vectors.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 4) RAG example (FAISS + embeddings + generator)
# -------------------------------
# This is a minimal RAG flow: build vector store from a list of docs, then
# retrieve top-k and call a generator to produce final answer.

from typing import Tuple

def build_faiss_index(docs: List[str], embed_model="all-MiniLM-L6-v2") -> Tuple[object, List[str]]:
    """Returns (index, docs) where index is a FAISS index and docs are the source texts."""
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss

        st = SentenceTransformer(embed_model)
        embeddings = st.encode(docs, convert_to_numpy=True)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)
        return index, docs
    except Exception as e:
        raise

# example in-memory knowledge base
KNOWLEDGE_DOCS = [
    "Kubernetes pods are the smallest deployable units in Kubernetes.",
    "Node.js uses an event loop and a single-threaded execution model with background worker threads for I/O.",
    "MongoDB stores documents in BSON format and supports flexible schemas.",
]

# Build FAISS index at startup (small example)
try:
    FAISS_INDEX, FAISS_DOCS = build_faiss_index(KNOWLEDGE_DOCS)
except Exception:
    FAISS_INDEX, FAISS_DOCS = None, []

@app.post("/rag_query")
async def rag_query(req: RAGQuery, embed_model: Optional[str] = "all-MiniLM-L6-v2", generator: Optional[str] = "gpt2"):
    """Run a simple RAG pipeline: retrieve + generate.
    For serious usage, use batching, caching, and an orchestration layer.
    """
    try:
        if FAISS_INDEX is None:
            raise HTTPException(status_code=500, detail="FAISS index not built")
        from sentence_transformers import SentenceTransformer
        import numpy as np
        from transformers import AutoModelForCausalLM, AutoTokenizer

        st = SentenceTransformer(embed_model)
        q_emb = st.encode([req.query], convert_to_numpy=True)
        D, I = FAISS_INDEX.search(q_emb, req.top_k)
        retrieved = [FAISS_DOCS[i] for i in I[0]]

        # Construct prompt for the generator
        prompt = """
Use the following retrieved documents to answer the question. Be concise.

Documents:\n{docs}

Question: {query}

Answer:
""".format(docs="\n\n".join(retrieved), query=req.query)

        # Use HF generator to produce final answer (or call OpenAI)
        tokenizer = AutoTokenizer.from_pretrained(generator)
        model = AutoModelForCausalLM.from_pretrained(generator)
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=150)
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {"retrieved": retrieved, "answer": answer}
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------
# 5) LoRA Fine-tuning helper (PEFT) - run offline
# -------------------------------
@app.post("/train_lora")
async def train_lora_script(dataset_path: Optional[str] = None, base_model: Optional[str] = "meta-llama/Llama-2-7b-chat"):
    """
    Example endpoint that will kick off a LoRA training *script*.
    In production you should run training on separate infrastructure (GPU nodes) rather than inside the API.

    Here we return a shell script / python snippet you can run to train via PEFT + Transformers.
    """
    # Return a training script as string (user runs it on a GPU machine)
    script = f"""
# LoRA training example (run on a GPU machine)
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model

dataset = load_dataset('json', data_files='{dataset_path}')  # prepare a JSON dataset

tokenizer = AutoTokenizer.from_pretrained('{base_model}', use_fast=True)
model = AutoModelForCausalLM.from_pretrained('{base_model}', device_map='auto', load_in_8bit=True)

config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q_proj','v_proj'],
    inference_mode=False,
)
model = get_peft_model(model, config)

# Tokenize and train (example)
def preprocess(ex):
    return tokenizer(ex['text'], truncation=True, max_length=1024)

train_dataset = dataset['train'].map(preprocess, batched=True)

training_args = TrainingArguments(
    output_dir='./lora-output',
    per_device_train_batch_size=8,
    num_train_epochs=3,
    logging_steps=10,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

trainer.train()
model.save_pretrained('./lora-output')
"""
    return {"script": script}

# -------------------------------
# 6) Utility: list available HF models (simple)
# -------------------------------
@app.get("/hf_list_examples")
async def hf_list_examples():
    # A short curated list (not an exhaustive search). Use the Hub for full listing.
    return {
        "instruction_tuned": ["OpenAssistant/ oasst-sft-1-pythia-12b", "tiiuae/falcon-7b-instruct"],
        "causal_small": ["gpt2", "distilgpt2"],
        "chat_style": ["facebook/blenderbot-400M-distill"],
        "code_models": ["codeparrot/codeparrot-small", "Salesforce/codegen-2B-mono"]
    }

# -------------------------------
# 7) Health check
# -------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "openai_key_present": bool(OPENAI_API_KEY)}

# -------------------------------
# If run as main, start uvicorn server
# -------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
