"""
easyResearchAssistant - Cloudflare AI Load Balancer
A FastAPI backend that acts as a load balancer for multiple Cloudflare AI Workers accounts.
"""
import json
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException, Depends, Header, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import settings
from load_balancer import load_balancer
from cloudflare_client import CloudflareAIClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="easyResearchAssistant",
    description="Cloudflare AI Load Balancer - Distributes requests across multiple Cloudflare accounts",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request/Response models
class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    prompt: str = Field(..., description="The prompt to send to the AI")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    max_tokens: int = Field(2048, ge=1, le=8192, description="Maximum tokens in response")
    temperature: float = Field(0.7, ge=0, le=2, description="Temperature for response generation")
    stream: bool = Field(True, description="Whether to stream the response")


class StatusResponse(BaseModel):
    """Response model for status endpoint"""
    status: str
    load_balancer: dict
    settings: dict


# Security dependency
async def verify_password(
    x_access_password: Optional[str] = Header(None, alias="X-Access-Password"),
    access_password: Optional[str] = Query(None)
):
    """
    Verify the access password from either header or query parameter.
    """
    password = x_access_password or access_password
    
    if not password:
        raise HTTPException(
            status_code=401,
            detail="Access password required. Provide via X-Access-Password header or access_password query param."
        )
    
    if password != settings.access_password:
        raise HTTPException(
            status_code=403,
            detail="Invalid access password"
        )
    
    return True


async def stream_with_retry(
    prompt: str,
    system_prompt: Optional[str],
    max_tokens: int,
    temperature: float
):
    """
    Stream response with automatic retry on rate limit (429) errors.
    """
    max_retries = settings.max_retries
    tried_accounts = set()
    
    for attempt in range(max_retries):
        account = load_balancer.get_next_account()
        
        if not account:
            yield f"data: {json.dumps({'error': 'No available accounts'})}\n\n"
            return
        
        if account.account_id in tried_accounts:
            # We've tried all accounts, reset and try again
            load_balancer.reset_failed_accounts()
            tried_accounts.clear()
            account = load_balancer.get_next_account()
            if not account:
                yield f"data: {json.dumps({'error': 'No available accounts after reset'})}\n\n"
                return
        
        tried_accounts.add(account.account_id)
        logger.info(f"Attempt {attempt + 1}: Using account {account.name} ({account.account_id[:8]}...)")
        
        client = CloudflareAIClient(account)
        stream_gen, _ = await client.ask_stream(
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        
        encountered_error = False
        error_status = None
        
        async for chunk in stream_gen:
            # Check for error marker
            if chunk.startswith("__ERROR__:"):
                error_status = int(chunk.split(":")[1])
                encountered_error = True
                break
            
            # Parse and forward the response
            try:
                yield f"data: {chunk}\n\n"
            except Exception as e:
                logger.error(f"Error forwarding chunk: {e}")
                continue
        
        if encountered_error:
            if error_status == 429:
                logger.warning(f"Rate limit hit for account {account.name}, switching to next...")
                load_balancer.mark_account_failed(account.account_id)
                yield f"data: {json.dumps({'info': f'Rate limited on {account.name}, retrying with another account...'})}\n\n"
                continue
            else:
                yield f"data: {json.dumps({'error': f'API error: {error_status}'})}\n\n"
                return
        else:
            # Success - mark account as recovered if it was failed
            load_balancer.mark_account_recovered(account.account_id)
            yield "data: [DONE]\n\n"
            return
    
    yield f"data: {json.dumps({'error': 'Max retries exceeded'})}\n\n"


@app.get("/")
async def root():
    """Health check endpoint"""
    return {"message": "easyResearchAssistant is running", "version": "1.0.0"}


@app.get("/health")
async def health():
    """Health check with account status"""
    return {
        "status": "healthy",
        "accounts_active": len(load_balancer.accounts),
        "accounts_total": len(load_balancer.all_accounts)
    }


@app.post("/ask")
async def ask(
    request: AskRequest,
    _: bool = Depends(verify_password)
):
    """
    Main endpoint to ask the AI a question.
    Routes requests through the load balancer and handles rate limiting automatically.
    """
    if not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    
    logger.info(f"Received request - Prompt: {request.prompt[:50]}...")
    
    if request.stream:
        return StreamingResponse(
            stream_with_retry(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # Non-streaming response
        max_retries = settings.max_retries
        tried_accounts = set()
        
        for attempt in range(max_retries):
            account = load_balancer.get_next_account()
            
            if not account:
                raise HTTPException(status_code=503, detail="No available accounts")
            
            if account.account_id in tried_accounts:
                load_balancer.reset_failed_accounts()
                tried_accounts.clear()
                account = load_balancer.get_next_account()
                if not account:
                    raise HTTPException(status_code=503, detail="No available accounts")
            
            tried_accounts.add(account.account_id)
            logger.info(f"Attempt {attempt + 1}: Using account {account.name}")
            
            client = CloudflareAIClient(account)
            response, status_code = await client.ask_non_stream(
                prompt=request.prompt,
                system_prompt=request.system_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature
            )
            
            if status_code == 429:
                logger.warning(f"Rate limit hit for account {account.name}, switching...")
                load_balancer.mark_account_failed(account.account_id)
                continue
            
            if status_code == 200:
                load_balancer.mark_account_recovered(account.account_id)
                return JSONResponse(content={"response": response, "account": account.name})
            
            raise HTTPException(status_code=status_code, detail=f"API error: {response}")
        
        raise HTTPException(status_code=429, detail="All accounts rate limited")


@app.get("/status", response_model=StatusResponse)
async def get_status(_: bool = Depends(verify_password)):
    """Get the current status of the load balancer"""
    return StatusResponse(
        status="running",
        load_balancer=load_balancer.get_status(),
        settings={
            "rotation_strategy": settings.rotation_strategy,
            "max_retries": settings.max_retries,
            "model": settings.cloudflare_model
        }
    )


@app.post("/strategy/{strategy}")
async def set_strategy(strategy: str, _: bool = Depends(verify_password)):
    """Change the rotation strategy (round_robin or random)"""
    try:
        load_balancer.set_strategy(strategy)
        return {"message": f"Strategy changed to {strategy}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/reset")
async def reset_accounts(_: bool = Depends(verify_password)):
    """Reset all failed accounts to active status"""
    load_balancer.reset_failed_accounts()
    return {"message": "All accounts reset to active status"}


@app.post("/reload")
async def reload_accounts(_: bool = Depends(verify_password)):
    """Reload accounts from configuration"""
    load_balancer.reload_accounts()
    return {"message": "Accounts reloaded", "count": len(load_balancer.all_accounts)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
