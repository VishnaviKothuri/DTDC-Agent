import os
import json
import time
import httpx
import logging
import shutil

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_utils import retrieve_context, craft_prompt   # index_apps should be run once separately!
from file_utils import parse_and_create_files, zip_folder

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only; restrict for production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"

class PromptRequest(BaseModel):
    prompt: str
    download: bool = False

class PromptResponse(BaseModel):
    response: str

async def get_model_response(prompt: str, stream: bool = True) -> str:
    payload = {
        "model": "codellama",
        "prompt": prompt,
        "stream": stream
    }
    full_response = ""
    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", OLLAMA_URL, json=payload) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.strip():
                        try:
                            data = json.loads(line)
                            full_response += data.get("response", "")
                        except json.JSONDecodeError:
                            continue
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama error: {str(e)}")
    return full_response.strip()

@app.post("/prompt", response_model=PromptResponse)
async def send_prompt(request: PromptRequest):
    response_text = await get_model_response(request.prompt)
    return PromptResponse(response=response_text)

@app.post("/generate-springboot")
async def generate_spring_boot_app(request: PromptRequest, background_tasks: BackgroundTasks):
    try:
        logger.info("Generating Spring Boot application with RAG context...")
        contexts = retrieve_context(request.prompt)
        final_prompt = craft_prompt(request.prompt, contexts)
        model_output = await get_model_response(final_prompt)

        if not request.download:
            # Return plain text/code
            return {"response": model_output}

        # Only run file and zip logic if download requested
        temp_dir = parse_and_create_files(model_output)
        zip_path = zip_folder(temp_dir)

        # Cleanup after response sent
        background_tasks.add_task(os.remove, zip_path)
        background_tasks.add_task(lambda: shutil.rmtree(temp_dir, ignore_errors=True))

        return FileResponse(zip_path, filename="springboot_app.zip")
    except Exception as e:
        logger.exception("Error during generation")
        raise HTTPException(status_code=500, detail=f"Generation error: {str(e)}")
