"""
Agentic AI Template
-------------------
This template demonstrates:
- Multi-state agent architecture
- Memory + caching
- Tools / actions
- State transitions
- Modular design for expansion

Components:
1. AgentStateManager — manages conversation state (Thinking, Acting, Reflecting, Idle)
2. MemoryCache — short-term & long-term memory
3. Tools — search, calculator, file read/write, etc.
4. AgentBrain — orchestrates reasoning and tool use
5. AgentAPI (FastAPI) — interface for applications

Requirements:
fastapi
uvicorn
pydantic
python-dotenv
requests
"""

from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict, Any
import time
import json

# ------------------------------------------
# Memory Cache (Short-term + Long-term)
# ------------------------------------------
class MemoryCache:
    def __init__(self):
        self.short_term = {}   # Request to request
        self.long_term = {}    # Persisted across session

    def store_short(self, key, value):
        self.short_term[key] = value

    def get_short(self, key):
        return self.short_term.get(key)

    def store_long(self, key, value):
        self.long_term[key] = value

    def get_long(self, key):
        return self.long_term.get(key)

    def clear_short(self):
        self.short_term = {}

memory = MemoryCache()

# ------------------------------------------
# Tools (Modular Functions)
# ------------------------------------------
class Tools:

    @staticmethod
    def calculator(expr: str) -> str:
        try:
            result = eval(expr)
            return f"Result: {result}"
        except Exception:
            return "Invalid expression"

    @staticmethod
    def search_web(query: str) -> str:
        # Mock search (replace with real API)
        return f"Search results for '{query}' (mocked)."

    @staticmethod
    def read_file(path: str) -> str:
        try:
            with open(path, "r") as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"

# ------------------------------------------
# Agent States
# ------------------------------------------
class AgentStateManager:
    STATES = ["IDLE", "THINKING", "ACTING", "REFLECTING"]

    def __init__(self):
        self.current_state = "IDLE"

    def set_state(self, state: str):
        if state in self.STATES:
            self.current_state = state

    def get_state(self):
        return self.current_state

state_manager = AgentStateManager()

# ------------------------------------------
# Agent Brain
# ------------------------------------------
class AgentBrain:

    def reason(self, query: str) -> Dict[str, Any]:
        state_manager.set_state("THINKING")
        time.sleep(0.2)

        # Retrieve memory for context
        context = memory.get_long("domain_knowledge") or "None"

        # Simple routing logic
        if "calculate" in query.lower():
            state_manager.set_state("ACTING")
            expr = query.lower().replace("calculate", "").strip()
            result = Tools.calculator(expr)
        elif "search" in query.lower():
            state_manager.set_state("ACTING")
            q = query.lower().replace("search", "").strip()
            result = Tools.search_web(q)
        else:
            # fallback narrative reasoning
            result = f"I processed your input with context '{context}'."

        # Cache store
        memory.store_short("last_answer", result)

        # Reflection
        state_manager.set_state("REFLECTING")
        reflection = f"I answered based on reasoning path and memory context."

        state_manager.set_state("IDLE")

        return {
            "state": state_manager.get_state(),
            "result": result,
            "reflection": reflection,
            "cache_short_term": memory.short_term,
        }

agent = AgentBrain()

# ------------------------------------------
# API Layer
# ------------------------------------------
class QueryModel(BaseModel):
    query: str
    memory_write: Optional[Dict[str, str]] = None

app = FastAPI(title="Agentic AI Template")

@app.post("/ask")
def ask_agent(req: QueryModel):

    # Write to long-term memory
    if req.memory_write:
        for k, v in req.memory_write.items():
            memory.store_long(k, v)

    response = agent.reason(req.query)
    return response

@app.get("/state")
def get_state():
    return {"state": state_manager.get_state()}

@app.get("/memory")
def memory_dump():
    return {
        "short_term": memory.short_term,
        "long_term": memory.long_term,
    }

# ------------------------------------------
# Run program
# ------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
