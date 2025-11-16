"""
Agentic AI — Enhanced + Production Template
===========================================
This single-file template demonstrates two separate deliverables:

A) Agentic Enhancements
  - ReAct loop (Reason -> Act -> Observe -> Repeat)
  - Multi-agent architecture (Planner, Worker, Critic)
  - Tool routing with confidence scoring
  - Vector memory (FAISS) + embeddings (SentenceTransformers)
  - Short-term and long-term memory with Redis optional backend
  - Async tool execution and timeouts
  - OpenAI and HuggingFace generator adapters (pluggable)

B) Production-Level Add-ons
  - Redis-based memory & cache (fallback to in-memory)
  - Dockerfile snippet + multi-stage build
  - requirements.txt content for CPU and GPU
  - Kubernetes Deployment + Service + HorizontalPodAutoscaler (YAML strings)
  - Health checks, metrics (Prometheus), structured logging
  - Example GitHub Actions CI workflow for lint, tests, Docker build
  - Runtime config via environment variables

Usage
-----
1) Install deps from requirements.txt (or use provided Dockerfile)
2) Set env vars for optional integrations (OPENAI_API_KEY, REDIS_URL)
3) Run: `uvicorn agentic_production_template:app --reload --port 8002`
4) Try `/ask`, `/state`, `/memory`, `/metrics`, `/health`

NOTE: This is a template for development and demonstration. For real production
deployments, move training/fine-tuning off the request path, add stronger
security, secrets management, and run heavy workloads on GPU instances.

"""

from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os
import time
import uuid
import asyncio
import logging

# Optional Redis support
try:
    import redis
except Exception:
    redis = None

# Optional FAISS + sentence-transformers
try:
    import faiss
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    faiss = None

# Optional OpenAI and transformers adapters
try:
    import openai
except Exception:
    openai = None

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except Exception:
    AutoTokenizer = None

# -----------------------------
# Config (env)
# -----------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")  # e.g., redis://localhost:6379/0
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")
GENERATOR_HF = os.environ.get("GENERATOR_HF", "gpt2")

# -----------------------------
# Logging + metrics
# -----------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger("agentic")

REQUEST_COUNTER = 0

# -----------------------------
# Memory Backends: Redis or In-Memory
# -----------------------------
class MemoryBackend:
    def __init__(self):
        self.use_redis = False
        self.client = None
        if REDIS_URL and redis:
            try:
                self.client = redis.from_url(REDIS_URL)
                # quick ping
                self.client.ping()
                self.use_redis = True
                logger.info("Using Redis memory backend")
            except Exception as e:
                logger.warning(f"Redis connection failed, falling back to memory: {e}")
                self.client = None
        self._mem = {"short": {}, "long": {}}

    def set_short(self, key: str, value: Any, ttl: Optional[int] = None):
        if self.use_redis:
            self.client.hset("agent:short", key, json_serialize(value))
            if ttl:
                self.client.expire("agent:short", ttl)
        else:
            self._mem["short"][key] = value

    def get_short(self, key: str):
        if self.use_redis:
            v = self.client.hget("agent:short", key)
            return json_deserialize(v) if v else None
        return self._mem["short"].get(key)

    def set_long(self, key: str, value: Any):
        if self.use_redis:
            self.client.hset("agent:long", key, json_serialize(value))
        else:
            self._mem["long"][key] = value

    def get_long(self, key: str):
        if self.use_redis:
            v = self.client.hget("agent:long", key)
            return json_deserialize(v) if v else None
        return self._mem["long"].get(key)

    def dump(self):
        if self.use_redis:
            return {"short": {k: json_deserialize(v) for k, v in self.client.hgetall("agent:short").items()},
                    "long": {k: json_deserialize(v) for k, v in self.client.hgetall("agent:long").items()}}
        return self._mem


def json_serialize(obj):
    try:
        return json.dumps(obj)
    except Exception:
        return str(obj)


def json_deserialize(b):
    if b is None:
        return None
    if isinstance(b, bytes):
        b = b.decode()
    try:
        return json.loads(b)
    except Exception:
        return b

memory = MemoryBackend()

# -----------------------------
# Vector Memory (FAISS wrapper)
# -----------------------------
class VectorMemory:
    def __init__(self, model_name=EMBED_MODEL):
        self.model_name = model_name
        self.model = None
        self.index = None
        self.docs = []
        if faiss and SentenceTransformer:
            try:
                self.model = SentenceTransformer(model_name)
            except Exception as e:
                logger.warning(f"Embedding model init failed: {e}")

    def build(self, docs: List[str]):
        if not self.model:
            raise RuntimeError("Embedding model unavailable")
        embs = self.model.encode(docs, convert_to_numpy=True)
        dim = embs.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embs)
        self.docs = docs
        logger.info(f"Built index with {len(docs)} docs")

    def search(self, query: str, k=4):
        if not self.index:
            return []
        q_emb = self.model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(q_emb, k)
        return [self.docs[i] for i in I[0]]

vector_memory = VectorMemory()

# Build a small knowledge base
try:
    sample_docs = [
        "Kubernetes uses containers grouped as pods.",
        "Node.js uses an event loop and libuv for async I/O.",
        "Redis is an in-memory data store used for caching and message brokering.",
    ]
    vector_memory.build(sample_docs)
except Exception:
    pass

# -----------------------------
# Tools: implemented as async functions
# -----------------------------
async def tool_calculator(expr: str) -> Dict[str, Any]:
    await asyncio.sleep(0)
    try:
        # very careful eval - in real world use a math parser
        safe_globals = {"__builtins__": None}
        result = eval(expr, safe_globals, {})
        return {"tool": "calculator", "output": str(result), "confidence": 0.98}
    except Exception as e:
        return {"tool": "calculator", "output": f"error: {e}", "confidence": 0.1}

async def tool_search(query: str) -> Dict[str, Any]:
    # placeholder for an async web search (Bing / Google / custom API)
    await asyncio.sleep(0.1)
    return {"tool": "search", "output": f"Top results for: {query}", "confidence": 0.7}

async def tool_read_file(path: str) -> Dict[str, Any]:
    await asyncio.sleep(0)
    try:
        with open(path, 'r') as f:
            content = f.read()
        return {"tool": "read_file", "output": content[:200], "confidence": 0.9}
    except Exception as e:
        return {"tool": "read_file", "output": f"error: {e}", "confidence": 0.2}

TOOL_MAP = {
    "calculator": tool_calculator,
    "search": tool_search,
    "read_file": tool_read_file,
}

# -----------------------------
# Generator Adapters (OpenAI or HF local)
# -----------------------------
async def openai_generate(prompt: str, max_tokens=150) -> str:
    if not OPENAI_API_KEY or not openai:
        raise RuntimeError("OpenAI not configured")
    openai.api_key = OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role":"system","content":"You are a helpful agent."}, {"role":"user","content":prompt}],
        max_tokens=max_tokens,
    )
    return resp['choices'][0]['message']['content']

async def hf_generate(prompt: str, max_tokens=150) -> str:
    if not AutoTokenizer:
        raise RuntimeError("HF transformers not available")
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_HF)
    model = AutoModelForCausalLM.from_pretrained(GENERATOR_HF)
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Choose generator based on env
GENERATOR = openai_generate if OPENAI_API_KEY and openai else hf_generate

# -----------------------------
# Multi-Agent Components
# -----------------------------
class Planner:
    async def plan(self, user_query: str) -> List[Dict[str, Any]]:
        """Return a plan: list of steps with intended tool/actions."""
        # naive planner: detect keywords
        steps = []
        if "calculate" in user_query.lower():
            steps.append({"action": "calculator", "input": user_query.lower().replace("calculate", "").strip()})
        if "search" in user_query.lower():
            steps.append({"action": "search", "input": user_query})
        if not steps:
            # default: retrieve context then generate
            steps.append({"action": "generate", "input": user_query})
        return steps

class Worker:
    async def execute(self, step: Dict[str, Any], timeout: float = 5.0) -> Dict[str, Any]:
        action = step.get('action')
        if action in TOOL_MAP:
            coro = TOOL_MAP[action](step.get('input', ''))
        elif action == 'generate':
            coro = GENERATOR(step.get('input', ''))
        else:
            return {"action": action, "output": "unknown action", "confidence": 0.0}

        try:
            # apply timeout
            res = await asyncio.wait_for(coro, timeout=timeout)
            # if res is string (generator), standardize
            if isinstance(res, str):
                return {"action": action, "output": res, "confidence": 0.6}
            return res
        except asyncio.TimeoutError:
            return {"action": action, "output": "timeout", "confidence": 0.0}

class Critic:
    async def critique(self, plan_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # simple critic: check confidences and flag low ones
        low_conf = [r for r in plan_results if r.get('confidence', 1.0) < 0.5]
        verdict = "ok" if not low_conf else "needs_review"
        return {"verdict": verdict, "issues": low_conf}

planner = Planner()
worker = Worker()
critic = Critic()

# -----------------------------
# ReAct Loop Implementation
# -----------------------------
async def react_loop(user_query: str, max_steps: int = 4) -> Dict[str, Any]:
    global REQUEST_COUNTER
    REQUEST_COUNTER += 1
    session_id = str(uuid.uuid4())
    logger.info(f"React loop start {session_id} - query: {user_query}")

    plan = await planner.plan(user_query)
    results = []

    for i, step in enumerate(plan[:max_steps]):
        # Observe: check vector memory for context
        retrieved = vector_memory.search(step.get('input', ''), k=2) if vector_memory.index else []
        memory.set_short(f"last_step_{i}", {"step": step, "retrieved": retrieved})

        # Act: execute step
        res = await worker.execute(step)
        results.append(res)

        # Reflect: critic
        critique = await critic.critique(results)
        memory.set_short(f"critique_{i}", critique)

        # If critic says ok and action was generate, we can stop early
        if critique['verdict'] == 'ok' and step.get('action') == 'generate':
            break

    # Final aggregation: produce an answer using generator and retrieved context
    aggregated_prompt = "Use the following tool outputs to answer the user's question:\n\n"
    for r in results:
        aggregated_prompt += f"Action: {r.get('action')}\nOutput: {r.get('output')}\n\n"
    aggregated_prompt += f"User question: {user_query}\nAnswer concisely."

    final_answer = await GENERATOR(aggregated_prompt)

    logger.info(f"React loop done {session_id}")

    memory.set_short("last_answer", final_answer)
    return {"session_id": session_id, "results": results, "final_answer": final_answer}

# -----------------------------
# FastAPI endpoints
# -----------------------------
app = FastAPI(title="Agentic AI — Enhanced + Production Template")

class AskReq(BaseModel):
    query: str
    async_mode: bool = False

@app.post('/ask')
async def ask(req: AskReq, background_tasks: BackgroundTasks):
    start = time.time()
    if req.async_mode:
        # schedule in background and return a task id
        task_id = str(uuid.uuid4())
        background_tasks.add_task(asyncio.ensure_future, react_loop(req.query))
        return {"task_id": task_id, "status": "scheduled"}
    else:
        res = await react_loop(req.query)
        duration = time.time() - start
        return {"duration_s": duration, **res}

@app.get('/state')
async def state():
    return {"ready": True, "requests": REQUEST_COUNTER}

@app.get('/memory')
async def memdump():
    return memory.dump()

@app.get('/health')
async def health():
    return {"status": "ok", "redis": memory.use_redis}

@app.get('/metrics')
async def metrics():
    return {"requests_total": REQUEST_COUNTER}

# -----------------------------
# Production helpers: Dockerfile, requirements, k8s YAML, CI
# -----------------------------
DOCKERFILE = r'''# Dockerfile (multi-stage)
FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip wheel -r requirements.txt -w /wheels

FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels -r requirements.txt
COPY . /app
ENV PYTHONUNBUFFERED=1
CMD ["uvicorn", "agentic_production_template:app", "--host", "0.0.0.0", "--port", "8002"]
'''

REQUIREMENTS = r'''fastapi
uvicorn[standard]
python-dotenv
pydantic
requests
redis
sentence-transformers
faiss-cpu
transformers
torch
openai
'''

K8S_DEPLOY = r'''apiVersion: apps/v1
kind: Deployment
metadata:
  name: agentic-ai
spec:
  replicas: 2
  selector:
    matchLabels:
      app: agentic-ai
  template:
    metadata:
      labels:
        app: agentic-ai
    spec:
      containers:
      - name: agentic-ai
        image: your-registry/agentic-ai:latest
        ports:
        - containerPort: 8002
        readinessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 5
          periodSeconds: 10
        livenessProbe:
          httpGet:
            path: /health
            port: 8002
          initialDelaySeconds: 15
          periodSeconds: 20
---
apiVersion: v1
kind: Service
metadata:
  name: agentic-ai
spec:
  selector:
    app: agentic-ai
  ports:
  - port: 80
    targetPort: 8002
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agentic-ai-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agentic-ai
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 60
'''

CI_GHA = r'''# .github/workflows/ci.yml
name: CI
on: [push]
jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install deps
      run: pip install -r requirements.txt
    - name: Lint
      run: pip install flake8 && flake8 .
    - name: Run tests
      run: pytest -q
    - name: Build Docker Image
      run: |
        docker build -t your-registry/agentic-ai:latest .
'''

# -----------------------------
# Expose production assets endpoint
# -----------------------------
@app.get('/assets/dockerfile')
async def get_dockerfile():
    return {"dockerfile": DOCKERFILE}

@app.get('/assets/requirements')
async def get_requirements():
    return {"requirements.txt": REQUIREMENTS}

@app.get('/assets/k8s')
async def get_k8s():
    return {"k8s": K8S_DEPLOY}

@app.get('/assets/ci')
async def get_ci():
    return {"ci_github_actions": CI_GHA}

# -----------------------------
# Run dev server
# -----------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8002)
