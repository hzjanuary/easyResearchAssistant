"""API Gateway - High-Availability AI Inference Gateway."""
import os
import json
import logging
import secrets
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, HTTPException, Depends, Header, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from provider_manager import (
    ProviderManager,
    InferenceNode,
    NodeType,
    SelectionStrategy,
    create_provider_manager_from_env
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("api_gateway")


class GatewayConfig:
    """Gateway configuration."""
    
    # Security
    ACCESS_TOKEN: str = os.getenv("ACCESS_TOKEN", secrets.token_urlsafe(32))
    
    # Inference settings
    DEFAULT_MODEL: str = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3-8b-instruct")
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", "3"))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    # Local fallback
    OLLAMA_ENDPOINT: str = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")
    OLLAMA_MAX_TOKENS: int = int(os.getenv("OLLAMA_MAX_TOKENS", "1024"))
    
    RESEARCH_SYSTEM_PROMPT: str = """You are an academic research assistant.
Provide clear, structured explanations with examples. Summarize key points and suggest related topics."""


config = GatewayConfig()

security = HTTPBearer(auto_error=False)


async def verify_access_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
    x_access_token: Optional[str] = Header(None, alias="X-Access-Token")
) -> bool:
    token = None
    
    if credentials:
        token = credentials.credentials
    elif x_access_token:
        token = x_access_token
    
    if not token:
        raise HTTPException(
            status_code=401,
            detail="Access token required. Use Bearer authentication or X-Access-Token header.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    if not secrets.compare_digest(token, config.ACCESS_TOKEN):
        raise HTTPException(
            status_code=403,
            detail="Invalid access token"
        )
    
    return True


class InferenceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32000)
    system_prompt: Optional[str] = Field(None, max_length=4000)
    max_tokens: int = Field(2048, ge=1, le=8192)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stream: bool = Field(True)
    research_mode: bool = Field(False)


class GatewayStatus(BaseModel):
    status: str
    version: str
    provider_status: dict
    config: dict


class InferenceResponse(BaseModel):
    response: str
    provider: str
    tokens_used: Optional[int] = None


class InferenceClient:
    """Unified client for Cloudflare and Ollama endpoints."""
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
    
    async def call_cloudflare(
        self,
        node: InferenceNode,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Call Cloudflare Workers AI endpoint.
        Yields chunks for streaming, or full response for non-streaming.
        """
        url = f"https://api.cloudflare.com/client/v4/accounts/{node.account_id}/ai/run/{config.DEFAULT_MODEL}"
        
        headers = {
            "Authorization": f"Bearer {node.api_token}",
            "Content-Type": "application/json"
        }
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "stream": stream,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            if stream:
                async with client.stream("POST", url, headers=headers, json=payload) as response:
                    if response.status_code != 200:
                        yield f"__ERROR__:{response.status_code}"
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]
                            if data == "[DONE]":
                                yield "__DONE__"
                                return
                            yield data
            else:
                response = await client.post(url, headers=headers, json=payload)
                if response.status_code != 200:
                    yield f"__ERROR__:{response.status_code}"
                    return
                
                result = response.json()
                if "result" in result and "response" in result["result"]:
                    yield result["result"]["response"]
                else:
                    yield str(result)
    
    async def call_ollama(
        self,
        node: InferenceNode,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,  # Conservative for RTX 3050
        temperature: float = 0.7,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """
        Call local Ollama endpoint for fallback inference.
        Lightweight configuration to preserve GPU resources.
        """
        url = f"{node.endpoint}/api/generate"
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\nUser: {prompt}"
        
        payload = {
            "model": node.metadata.get("model", config.OLLAMA_MODEL),
            "prompt": full_prompt,
            "stream": stream,
            "options": {
                "num_predict": min(max_tokens, config.OLLAMA_MAX_TOKENS),
                "temperature": temperature,
                "num_gpu": 1,  # Use GPU but be conservative
                "num_thread": 4  # Limit CPU threads
            }
        }
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout * 2) as client:  # Longer timeout for local
                if stream:
                    async with client.stream("POST", url, json=payload) as response:
                        if response.status_code != 200:
                            yield f"__ERROR__:{response.status_code}"
                            return
                        
                        async for line in response.aiter_lines():
                            if line:
                                try:
                                    data = json.loads(line)
                                    if "response" in data:
                                        yield json.dumps({"response": data["response"]})
                                    if data.get("done", False):
                                        yield "__DONE__"
                                        return
                                except json.JSONDecodeError:
                                    continue
                else:
                    response = await client.post(url, json=payload)
                    if response.status_code != 200:
                        yield f"__ERROR__:{response.status_code}"
                        return
                    
                    result = response.json()
                    yield result.get("response", "")
        except httpx.ConnectError:
            logger.warning("Ollama not available (connection refused)")
            yield "__ERROR__:503"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            yield f"__ERROR__:500"
    
    async def call_node(
        self,
        node: InferenceNode,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        stream: bool = True
    ) -> AsyncGenerator[str, None]:
        """Route request to appropriate provider based on node type"""
        if node.node_type == NodeType.OLLAMA:
            async for chunk in self.call_ollama(
                node, prompt, system_prompt, max_tokens, temperature, stream
            ):
                yield chunk
        else:
            async for chunk in self.call_cloudflare(
                node, prompt, system_prompt, max_tokens, temperature, stream
            ):
                yield chunk


provider_manager: Optional[ProviderManager] = None
inference_client: Optional[InferenceClient] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global provider_manager, inference_client
    
    logger.info("Initializing AI Inference Gateway...")
    provider_manager = create_provider_manager_from_env()
    inference_client = InferenceClient(timeout=config.REQUEST_TIMEOUT)
    
    logger.info(f"Loaded {len(provider_manager.all_nodes)} inference nodes")
    logger.info(f"Local fallback: {'enabled' if provider_manager.has_fallback else 'disabled'}")
    logger.info(f"Access token: {config.ACCESS_TOKEN[:8]}... (keep this secret!)")
    
    yield
    
    # Shutdown
    logger.info("Gateway shutdown complete")


app = FastAPI(
    title="easyResearchAssistant",
    description="High-Availability AI Inference Gateway with distributed provider support",
    version="2.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def execute_inference_with_fallback(
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float,
    stream: bool
) -> AsyncGenerator[str, None]:
    tried_nodes = set()
    
    # Phase 1: Try cloud providers
    for attempt in range(config.MAX_RETRIES):
        node = provider_manager.get_next_node()
        
        if not node:
            logger.warning("No cloud providers available")
            break
        
        if node.node_id in tried_nodes:
            # Cycle through - reset and continue or break
            provider_manager.reset_all_nodes()
            tried_nodes.clear()
            node = provider_manager.get_next_node()
            if not node:
                break
        
        tried_nodes.add(node.node_id)
        logger.info(f"Attempt {attempt + 1}: Using {node.name}")
        
        error_encountered = False
        error_code = None
        
        async for chunk in inference_client.call_node(
            node, prompt, system_prompt, max_tokens, temperature, stream
        ):
            if chunk.startswith("__ERROR__:"):
                error_code = int(chunk.split(":")[1])
                error_encountered = True
                break
            elif chunk == "__DONE__":
                provider_manager.mark_node_success(node.node_id)
                yield "__DONE__"
                return
            else:
                yield chunk
        
        if error_encountered and error_code is not None:
            is_rate_limit = error_code == 429
            is_server_error = 500 <= error_code < 600
            
            if is_rate_limit or is_server_error:
                provider_manager.mark_node_failed(node.node_id, is_rate_limit)
                logger.warning(f"Node {node.name} error {error_code}, trying next...")
                
                if stream:
                    yield json.dumps({
                        "info": f"Provider {node.name} unavailable, switching..."
                    })
                continue
            else:
                yield json.dumps({"error": f"Provider error: {error_code}"})
                return
        else:
            # Success without explicit DONE (non-streaming)
            provider_manager.mark_node_success(node.node_id)
            return
    
    # Phase 2: Local fallback
    if provider_manager.has_fallback:
        fallback = provider_manager.get_fallback_node()
        if fallback is None:
            yield json.dumps({"error": "Fallback node not available"})
            return
        
        logger.info(f"Cloud providers exhausted, using local fallback: {fallback.name}")
        
        if stream:
            yield json.dumps({
                "info": "Switching to local inference (may be slower)..."
            })
        
        async for chunk in inference_client.call_node(
            fallback, prompt, system_prompt,
            min(max_tokens, config.OLLAMA_MAX_TOKENS),  # Respect local limits
            temperature, stream
        ):
            if chunk.startswith("__ERROR__:"):
                yield json.dumps({"error": "Local fallback also failed"})
                return
            elif chunk == "__DONE__":
                yield "__DONE__"
                return
            else:
                yield chunk
        return
    
    # Phase 3: All options exhausted
    yield json.dumps({"error": "All inference providers exhausted"})


async def stream_response(
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float
) -> AsyncGenerator[str, None]:
    """Format streaming response as Server-Sent Events"""
    async for chunk in execute_inference_with_fallback(
        prompt, system_prompt, max_tokens, temperature, stream=True
    ):
        if chunk == "__DONE__":
            yield "data: [DONE]\n\n"
        else:
            yield f"data: {chunk}\n\n"


@app.get("/")
async def root():
    return {
        "service": "easyResearchAssistant",
        "version": "2.0.0",
        "status": "operational"
    }


@app.get("/health")
async def health_check():
    """Detailed health status (public)"""
    return {
        "status": "healthy",
        "providers": {
            "cloud_available": len(provider_manager.available_nodes) if provider_manager else 0,
            "cloud_total": len(provider_manager.all_nodes) if provider_manager else 0,
            "local_fallback": provider_manager.has_fallback if provider_manager else False
        }
    }


@app.post("/v1/inference", response_model=None)
async def inference(
    request: InferenceRequest,
    _: bool = Depends(verify_access_token)
):
    """
    Main inference endpoint.
    
    Accepts prompts and returns AI-generated responses with automatic
    provider failover and local fallback for high availability.
    """
    system_prompt = request.system_prompt
    
    # Apply research mode if enabled
    if request.research_mode:
        research_prefix = config.RESEARCH_SYSTEM_PROMPT
        if system_prompt:
            system_prompt = f"{research_prefix}\n\n{system_prompt}"
        else:
            system_prompt = research_prefix
    
    logger.info(f"Inference request: {request.prompt[:50]}... (research_mode={request.research_mode})")
    
    if request.stream:
        return StreamingResponse(
            stream_response(
                request.prompt,
                system_prompt,
                request.max_tokens,
                request.temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming: collect full response
        response_text = ""
        provider_used = "unknown"
        
        async for chunk in execute_inference_with_fallback(
            request.prompt,
            system_prompt,
            request.max_tokens,
            request.temperature,
            stream=False
        ):
            if chunk == "__DONE__":
                break
            try:
                data = json.loads(chunk)
                if "response" in data:
                    response_text += data["response"]
                elif "error" in data:
                    raise HTTPException(status_code=503, detail=data["error"])
            except json.JSONDecodeError:
                response_text += chunk
        
        return JSONResponse(content={
            "response": response_text,
            "provider": provider_used
        })


# Legacy endpoint for backward compatibility
@app.post("/ask")
async def ask_legacy(
    request: InferenceRequest,
    _: bool = Depends(verify_access_token)
):
    """Legacy /ask endpoint - redirects to /v1/inference"""
    return await inference(request, _)


@app.get("/v1/status", response_model=GatewayStatus)
async def get_status(_: bool = Depends(verify_access_token)):
    """Get detailed gateway and provider status"""
    return GatewayStatus(
        status="operational",
        version="2.0.0",
        provider_status=provider_manager.get_status(),
        config={
            "model": config.DEFAULT_MODEL,
            "max_retries": config.MAX_RETRIES,
            "timeout": config.REQUEST_TIMEOUT,
            "ollama_model": config.OLLAMA_MODEL
        }
    )


@app.post("/v1/providers/strategy/{strategy}")
async def set_strategy(strategy: str, _: bool = Depends(verify_access_token)):
    """Change provider selection strategy"""
    try:
        provider_manager.set_strategy(strategy)
        return {"message": f"Strategy changed to {strategy}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/v1/providers/reset")
async def reset_providers(_: bool = Depends(verify_access_token)):
    """Reset all providers to healthy status"""
    provider_manager.reset_all_nodes()
    return {"message": "All providers reset", "available": len(provider_manager.available_nodes)}


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "api_gateway:app",
        host=host,
        port=port,
        reload=os.getenv("DEBUG", "false").lower() == "true"
    )
