# main.py

import os
import json
import time
import logging
import shutil
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from rag_utils import rag_system
from file_utils import parse_and_create_files, zip_folder
from debug_routes import router as debug_router
from cache_utils import get_cached_response, set_cache_response

import hashlib
import json
from datetime import datetime, timedelta
from typing import Dict, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Scheduler for periodic RAG updates
scheduler = BackgroundScheduler()

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI lifespan: initialize & schedule
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting application…")
    # 1) Initialize RAG once
    rag_system.initialize()
    rag_system.debug_embedding_step_by_step()
    rag_system.update_from_git(force_reindex=True)
    report = rag_system.validate_embeddings()
    print("REPORT ",report)

    # 2) Start scheduler
    scheduler.start()
    logger.info("Scheduler started")
    # 3) Schedule daily at midnight
    scheduler.add_job(
        rag_system.update_from_git,
        trigger=CronTrigger(hour=0, minute=0),
        id="daily_rag_update",
        replace_existing=True
    )
    logger.info("Scheduled daily RAG update @ midnight")
    yield
    # Shutdown tasks
    logger.info("Shutting down application…")
    scheduler.shutdown()
    logger.info("Scheduler stopped")

# ──────────────────────────────────────────────────────────────────────────────
# App setup
app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # TODO: restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(debug_router)

OLLAMA_URL = "http://localhost:11434/api/generate"

# ──────────────────────────────────────────────────────────────────────────────
# Request / Response Models
class PromptRequest(BaseModel):
    prompt: str
    download: bool = False

class PromptResponse(BaseModel):
    response: str

class RAGUpdateRequest(BaseModel):
    force_reindex: bool = False

class RAGStatsResponse(BaseModel):
    initialized: bool
    total_documents: int = None
    persist_path: str = None
    error: str = None

prompt_cache: Dict[str, Dict] = {}
CACHE_EXPIRY_HOURS = 24  # Cache expires after 24 hours
# ──────────────────────────────────────────────────────────────────────────────
# Shared HTTPX client
http_client = httpx.AsyncClient(timeout=None)

async def get_model_response(prompt: str, stream: bool = True) -> str:
    """Send prompt to Ollama and collect streamed response."""
    payload = {"model": "codellama", "prompt": prompt, "stream": stream}
    full = ""
    try:
        async with http_client.stream("POST", OLLAMA_URL, json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                    full += chunk.get("response", "")
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {e}")
    return full.strip()

# ──────────────────────────────────────────────────────────────────────────────
# API Endpoints

def get_prompt_hash(prompt: str) -> str:
    """Generate a consistent hash for the prompt."""
    return hashlib.md5(prompt.strip().lower().encode()).hexdigest()

def get_cached_response(prompt: str) -> Optional[str]:
    """Check if prompt exists in cache and is not expired."""
    prompt_hash = get_prompt_hash(prompt)
    
    if prompt_hash in prompt_cache:
        cache_entry = prompt_cache[prompt_hash]
        
        # Check if cache is still valid
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        if datetime.now() - cache_time < timedelta(hours=CACHE_EXPIRY_HOURS):
            logger.info(f"Cache hit for prompt hash: {prompt_hash}")
            return cache_entry['response']
        else:
            # Remove expired cache
            del prompt_cache[prompt_hash]
            logger.info(f"Cache expired for prompt hash: {prompt_hash}")
    
    logger.info(f"Cache miss for prompt hash: {prompt_hash}")
    return None

def cache_response(prompt: str, response: str) -> None:
    """Store prompt-response pair in cache."""
    prompt_hash = get_prompt_hash(prompt)
    prompt_cache[prompt_hash] = {
        'prompt': prompt,
        'response': response,
        'timestamp': datetime.now().isoformat(),
        'access_count': 1
    }
    logger.info(f"Cached response for prompt hash: {prompt_hash}")

@app.post("/prompt", response_model=PromptResponse)
async def send_prompt(request: PromptRequest):
    out = await get_model_response(request.prompt)
    return PromptResponse(response=out)

@app.post("/generate-springboot")
async def generate_spring_boot_app(request: PromptRequest, background_tasks: BackgroundTasks):
    try:
        logger.info("Generate Suggestions for the given story…")
        jira_prompt = request.prompt
        logger.info(f"Received prompt: {jira_prompt}")

        cached_response = get_cached_response(jira_prompt)
        if cached_response:
            logger.info(f"Generated Response from Model... {cached_response}")
            return {"response": cached_response, "from_cache": True}
        
        contexts, tech_hint = rag_system.retrieve_context(
            jira_prompt, top_k=5
        )
        logger.info(f"Detected layer: {tech_hint}") 
        prompt = rag_system.craft_prompt(         
            jira_prompt, contexts, technology_hint=tech_hint
        )
        logger.info(f"Crafted prompt (User Story + RAG Context): {prompt}")
        # contexts = rag_system.retrieve_context(request.prompt)
        # prompt = rag_system.craft_prompt(request.prompt, contexts, technology_hint=tech_hint)
        code = await get_model_response(prompt)
        logger.info(f"Generated Response from Model... {code}")
        set_cache_response(jira_prompt, code)

        if not request.download:
            return {"response": code}

        tmp = parse_and_create_files(code)
        zip_path = zip_folder(tmp)
        background_tasks.add_task(os.remove, zip_path)
        background_tasks.add_task(lambda: shutil.rmtree(tmp, ignore_errors=True))
        return FileResponse(zip_path, filename="springboot_app.zip")

    except Exception as e:
        logger.exception("Generation error")
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

@app.post("/rag/update")
async def update_rag_index(request: RAGUpdateRequest, background_tasks: BackgroundTasks):
    """Trigger on-demand RAG index update."""
    try:
        logger.info(f"Enqueue RAG update (force_reindex={request.force_reindex})")
        background_tasks.add_task(rag_system.update_from_git, request.force_reindex)
        return {"message": "RAG update enqueued", "status": "processing"}
    except Exception as e:
        logger.exception("RAG update error")
        raise HTTPException(status_code=500, detail=f"RAG update error: {e}")

@app.get("/rag/stats", response_model=RAGStatsResponse)
async def get_rag_stats():
    """Retrieve RAG system statistics."""
    try:
        stats = rag_system.get_stats()
        logger.info("Backend count: %s", stats["technology_breakdown"].get("backend", 0))
        logger.info(
            "Core-bank-ops count: %s",
            stats["repository_breakdown"].get("core-bank-operations", 0)
        )
        return RAGStatsResponse(**stats)
    except Exception as e:
        logger.exception("Stats error")
        raise HTTPException(status_code=500, detail=f"Stats error: {e}")

@app.get("/health")
async def health_check():
    """Basic health check."""
    return {
        "status": "healthy",
        "rag_initialized": rag_system.is_initialized,
        "timestamp": time.time()
    }

# ──────────────────────────────────────────────────────────────────────────────
# Shutdown: close HTTP client
@app.on_event("shutdown")
async def cleanup():
    await http_client.aclose()
