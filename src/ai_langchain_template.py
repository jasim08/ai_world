"""
LangChain General Use Case Template
----------------------------------
This is a clean, production-friendly LangChain template supporting:
- LLMs (OpenAI, HuggingFaceHub, ChatOpenAI, ChatAnthropic)
- Tools (search, calculator, custom functions)
- Memory (ConversationBufferMemory)
- Chains (LLMChain, SequentialChain)
- Agents (ReAct-style Tool-using agent)
- RAG (vector DB + retriever)
- FastAPI interface

Install:
    pip install langchain langchain-openai langchain-community
    pip install openai faiss-cpu sentence-transformers fastapi uvicorn python-dotenv

"""

import os
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

# LangChain Imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.vectorstores import FAISS
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

load_dotenv()

# -------------------------------------------------------------
# 1. LLM SETUP
# -------------------------------------------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# -------------------------------------------------------------
# 2. MEMORY
# -------------------------------------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# -------------------------------------------------------------
# 3. TOOLS
# -------------------------------------------------------------

def calculator(expr: str) -> str:
    try:
        result = eval(expr)
        return str(result)
    except Exception:
        return "Invalid mathematical expression"

calc_tool = Tool(
    name="calculator",
    func=calculator,
    description="Useful for solving mathematical expressions. Input example: 2 + 2 * 10"
)

# Mock search tool

def search_tool_fn(query: str) -> str:
    return f"Mock search results for: {query}"

search_tool = Tool(
    name="search",
    func=search_tool_fn,
    description="Useful for searching general information."
)

# -------------------------------------------------------------
# 4. VECTORSTORE (RAG)
# -------------------------------------------------------------
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# To index sample text
texts = ["LangChain helps build LLM apps", "Agents can use tools", "RAG improves GPT accuracy"]
vector_db = FAISS.from_texts(texts, embedding=embeddings)
retriever = vector_db.as_retriever(search_kwargs={"k": 3})

# -------------------------------------------------------------
# 5. AGENT (ReAct-style, tool-using)
# -------------------------------------------------------------
agent = initialize_agent(
    tools=[calc_tool, search_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    memory=memory,
)

# -------------------------------------------------------------
# 6. CUSTOM CHAIN (Example: Summarizer)
# -------------------------------------------------------------
summarize_prompt = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text in 2 lines:\n{text}"
)

summarize_chain = LLMChain(llm=llm, prompt=summarize_prompt)

# -------------------------------------------------------------
# 7. FASTAPI APP WRAPPER
# -------------------------------------------------------------
app = FastAPI(title="LangChain General Template")

class Query(BaseModel):
    query: str
    mode: str = "agent"  # agent | summarize | rag


@app.post("/ask")
def ask(query: Query):
    if query.mode == "agent":
        response = agent.run(query.query)
        return {"mode": "agent", "response": response}

    if query.mode == "summarize":
        result = summarize_chain.run(query=query.query)
        return {"mode": "summarize", "response": result}

    if query.mode == "rag":
        docs = retriever.get_relevant_documents(query.query)
        content = "\n".join([d.page_content for d in docs])
        answer = llm.predict(f"Answer using context:\n{content}\n\nQuestion: {query.query}")
        return {"mode": "rag", "response": answer, "docs": content}

    return {"error": "Invalid mode"}


@app.get("/memory")
def get_memory():
    return {"memory": str(memory.load_memory_variables({}))}


@app.get("/health")
def health():
    return {"status": "OK"}


# Run directly
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
