# ==============================================
# TEMPLATES FOR TOP 6 AI TECHNOLOGIES (2025)
# ==============================================
# Includes:
# 1. LangGraph Agent Workflow
# 2. MCP Tool Server
# 3. vLLM Inference Server
# 4. PEFT + QLoRA Fine-Tuning
# 5. Vector DB (Weaviate + Pinecone)
# 6. CrewAI Multi-Agent System
# ----------------------------------------------

# =====================================================
# 1. LANGGRAPH — AGENT WORKFLOW ENGINE TEMPLATE
# =====================================================

from langgraph.graph import StateGraph, END
from typing import TypedDict

class AgentState(TypedDict):
    input: str
    answer: str

def llm_node(state: AgentState):
    state["answer"] = f"Processed by LLM: {state['input']}"
    return state

graph = StateGraph(AgentState)
graph.add_node("llm", llm_node)
graph.set_entry_point("llm")
graph.set_finish_point("llm")

langgraph_app = graph.compile()

# Example usage:
# result = langgraph_app.invoke({"input": "Hello"})
# print(result)


# =====================================================
# 2. MCP SERVER — TOOL CALLING STANDARD TEMPLATE
# =====================================================

import asyncio
from mcp.server.fastmcp import MCP

mcp = MCP("my-mcp-server")

@mcp.tool()
async def calculator(expr: str):
    return eval(expr, {"__builtins__": {}}, {})

async def run_mcp():
    await mcp.run()

# Run: asyncio.run(run_mcp())


# =====================================================
# 3. vLLM — FAST INFERENCE SERVER TEMPLATE
# =====================================================
# Run command:
#   vllm serve meta-llama/Llama-3-8B --port 8000
# Example client below:

import requests

def vllm_generate(prompt: str):
    res = requests.post("http://localhost:8000/generate", json={
        "prompt": prompt,
        "max_tokens": 200
    })
    return res.json()

# print(vllm_generate("Explain LLMs"))


# =====================================================
# 4. PEFT / QLoRA FINE-TUNING TEMPLATE
# =====================================================

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from datasets import load_dataset


def qlora_finetune(model_name="gpt2", out_dir="./qlora-model"):
    dataset = load_dataset("imdb", split="train[:2%]")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True)

    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))

    def preprocess(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)

    tokenized = dataset.map(preprocess, batched=True)

    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=out_dir, per_device_train_batch_size=2, num_train_epochs=1),
        train_dataset=tokenized,
        tokenizer=tokenizer
    )

    trainer.train()
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)


# =====================================================
# 5. VECTOR DB — WEAVIATE + PINECONE TEMPLATES
# =====================================================

# ---- WEAVIATE ----
import weaviate

weaviate_client = weaviate.Client("http://localhost:8080")

def weaviate_insert(text, emb):
    weaviate_client.data_object.create({"text": text, "embedding": emb}, class_name="Doc")

# ---- PINECONE ----
from pinecone import Pinecone

pc = Pinecone(api_key="YOUR_KEY")
index = pc.Index("my-index")

def pinecone_insert(text, emb):
    index.upsert([(text, emb)])


# =====================================================
# 6. CREWAI — MULTI-AGENT TEMPLATE
# =====================================================

from crewai import Agent, Task, Crew

planner = Agent(role="Planner", goal="Plan tasks", backstory="Expert strategist.")
worker = Agent(role="Worker", goal="Execute tasks", backstory="Fast executor.")
reviewer = Agent(role="Reviewer", goal="Review output", backstory="Quality checker.")

task1 = Task(description="Create outline for AI course", agent=planner)
task2 = Task(description="Write first module", agent=worker, depends_on=[task1])
task3 = Task(description="Review module", agent=reviewer, depends_on=[task2])

crew = Crew([planner, worker, reviewer], tasks=[task1, task2, task3])

# result = crew.run()
