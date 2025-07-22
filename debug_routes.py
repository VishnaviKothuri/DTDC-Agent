# debug_routes.py

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from rag_utils import rag_system

router = APIRouter()

class DebugRequest(BaseModel):
    prompt: str
    top_k: int = 3

class DebugResponse(BaseModel):
    file_paths: list[str]
    contexts: list[str]

@router.post("/rag/debug-retrieve", response_model=DebugResponse)
async def debug_retrieve(request: DebugRequest):
    try:
        # Ensure RAG is initialized
        # Directly query ChromaDB with metadata included
        prompt_emb = rag_system.embed_model.encode([request.prompt])[0]
        results = rag_system.chroma_collection.query(
            query_embeddings=[prompt_emb],
            n_results=request.top_k,
            include=["documents","metadatas"]
        )
        docs = results["documents"][0]
        paths = [m["path"] for m in results["metadatas"][0]]
        return DebugResponse(file_paths=paths, contexts=docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
