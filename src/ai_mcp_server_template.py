import asyncio
from mcp.server.fastmcp import MCP
from mcp.types import ToolRequest, ToolResponse

mcp = MCP("production-mcp-server")


# --------------------------
# TOOL 1 — Calculator
# --------------------------
@mcp.tool(name="calculator", description="Evaluate a safe math expression.")
async def calculator(expr: str):
    try:
        result = eval(expr, {"__builtins__": {}}, {})
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


# --------------------------
# TOOL 2 — Web Fetch (fetch URL)
# --------------------------
import aiohttp

@mcp.tool(name="web_fetch", description="Fetch text from a URL.")
async def web_fetch(url: str):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as r:
                return {"status": r.status, "body": await r.text()}
    except Exception as e:
        return {"error": str(e)}


# --------------------------
# TOOL 3 — Vector Search (FAISS)
# --------------------------
import faiss
import numpy as np

# Tiny in-memory vector store
index = faiss.IndexFlatL2(384)
documents = []


@mcp.tool(name="vector_add", description="Add a document to vector DB.")
async def vector_add(text: str, embedding: list[float]):
    global index
    index.add(np.array([embedding]).astype("float32"))
    documents.append(text)
    return {"message": "added", "count": len(documents)}


@mcp.tool(name="vector_search", description="Search vectors KNN.")
async def vector_search(query_embedding: list[float], k: int = 3):
    if len(documents) == 0:
        return {"results": []}

    dists, ids = index.search(
        np.array([query_embedding]).astype("float32"), k
    )

    results = []
    for _id, dist in zip(ids[0], dists[0]):
        if _id >= 0 and _id < len(documents):
            results.append({"doc": documents[_id], "score": float(dist)})

    return {"results": results}


# --------------------------
# RUN MCP SERVER
# --------------------------
async def main():
    await mcp.run()

if __name__ == "__main__":
    asyncio.run(main())
