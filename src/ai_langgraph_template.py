"""
LangGraph General Use Case Template (Multimodal)
------------------------------------------------
Features:
- LangGraph state machine
- LLM (OpenAI / ChatOpenAI)
- Multimodal: Image understanding using GPT-4o / Vision models
- RAG (FAISS)
- Tools (calculator, search)
- Memory store
- Graph nodes: planner -> worker -> reviewer
- FastAPI wrapper

Install:
    pip install langgraph langchain langchain-openai langchain-community
    pip install sentence-transformers faiss-cpu python-dotenv fastapi uvicorn

"""

import os
from dotenv import load_dotenv
from typing import Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File

# LangChain + LangGraph
from langgraph.graph import StateGraph, END
from langgraph.checkpoint import MemorySaver
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.tools import Tool
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS

load_dotenv()

# ---------------------------------------------------------
# MODELS (LLM + Vision)
# ---------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

vision_llm = ChatOpenAI(
    model="gpt-4o-mini",  # Vision-capable
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

# ---------------------------------------------------------
# MEMORY + VECTOR DB
# ---------------------------------------------------------
emb = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
initial_docs = ["LangGraph enables agent workflows", "Multimodal LLMs can analyze images"]
vector_db = FAISS.from_texts(initial_docs, embedding=emb)
retriever = vector_db.as_retriever()

memory = ConversationBufferMemory(memory_key="history", return_messages=True)

# ---------------------------------------------------------
# TOOLS
# ---------------------------------------------------------

def calc(expr: str) -> str:
    try:
        return str(eval(expr))
    except:
        return "Invalid math expression"


def search(query: str) -> str:
    return f"Mocked search results for: {query}"

calc_tool = Tool(name="calculator", func=calc, description="Math evaluator")
search_tool = Tool(name="search", func=search, description="Search tool")

# ---------------------------------------------------------
# LANGGRAPH STATE
# ---------------------------------------------------------
class AgentState(BaseModel):
    query: str = ""
    image_bytes: bytes | None = None
    plan: str | None = None
    result: str | None = None
    context: str | None = None


# ---------------------------------------------------------
# GRAPH NODES
# ---------------------------------------------------------
# 1. Planner node

def planner(state: AgentState) -> AgentState:
    plan_prompt = f"""
    You are a planner.
    User query: {state.query}
    Decide: Should you use tools, RAG, or vision?
    Return a short plan.
    """

    state.plan = llm.predict(plan_prompt)
    return state


# 2. Worker node (executes plan)

def worker(state: AgentState) -> AgentState:
    plan = state.plan.lower()

    # Image case
    if state.image_bytes:
        state.result = vision_llm.predict_messages([
            {"role": "user", "content": [
                {"type": "input_text", "text": state.query},
                {"type": "input_image", "image": state.image_bytes},
            ]}
        ]).content
        return state

    # RAG case
    if "rag" in plan or "context" in plan:
        docs = retriever.get_relevant_documents(state.query)
        content = "\n".join([d.page_content for d in docs])
        answer = llm.predict(f"""Use this context:
        {content}
        Question: {state.query}
        """)
        state.result = answer
        state.context = content
        return state

    # Tool use
    if "calculate" in plan:
        expr = state.query.replace("calculate", "").strip()
        state.result = calc_tool.run(expr)
        return state

    if "search" in plan:
        q = state.query.replace("search", "").strip()
        state.result = search_tool.run(q)
        return state

    # Default LLM reply
    state.result = llm.predict(state.query)
    return state


# 3. Reviewer

def reviewer(state: AgentState) -> AgentState:
    review = llm.predict(f"""
    Review the answer for correctness / clarity.
    Answer: {state.result}
    Return improved answer.
    """)
    state.result = review
    return state


# ---------------------------------------------------------
# BUILD GRAPH
# ---------------------------------------------------------
graph = StateGraph(AgentState)
graph.add_node("planner", planner)
graph.add_node("worker", worker)
graph.add_node("reviewer", reviewer)

graph.set_entry_point("planner")

graph.add_edge("planner", "worker")
graph.add_edge("worker", "reviewer")
graph.add_edge("reviewer", END)

workflow = graph.compile(checkpointer=MemorySaver())

# ---------------------------------------------------------
# FASTAPI SERVER
# ---------------------------------------------------------
app = FastAPI(title="LangGraph Multimodal Agent Template")

class TextQuery(BaseModel):
    query: str


@app.post("/ask")
def ask_endpoint(body: TextQuery):
    final = workflow.invoke({"query": body.query})
    return final


@app.post("/ask-image")
def ask_image(query: str, file: UploadFile = File(...)):
    img_bytes = file.file.read()
    final = workflow.invoke({"query": query, "image_bytes": img_bytes})
    return final


@app.get("/health")
def health():
    return {"status": "OK"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)