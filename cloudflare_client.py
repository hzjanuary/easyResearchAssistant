"""
Cloudflare Workers AI Client
Handles API calls to Cloudflare AI with streaming support
"""
import httpx
from typing import AsyncGenerator, Optional, Tuple
from config import CloudflareAccount, settings


class CloudflareAIClient:
    """
    Client for interacting with Cloudflare Workers AI API.
    Supports streaming responses.
    """
    
    BASE_URL = "https://api.cloudflare.com/client/v4/accounts"
    
    def __init__(self, account: CloudflareAccount):
        self.account = account
        self.model = settings.cloudflare_model
        self.timeout = settings.request_timeout
    
    def _get_api_url(self) -> str:
        """Build the API URL for the AI model"""
        return f"{self.BASE_URL}/{self.account.account_id}/ai/run/{self.model}"
    
    def _get_headers(self) -> dict:
        """Build request headers with authentication"""
        return {
            "Authorization": f"Bearer {self.account.api_token}",
            "Content-Type": "application/json"
        }
    
    async def ask_stream(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Tuple[AsyncGenerator[str, None], Optional[int]]:
        """
        Send a prompt to Cloudflare AI and stream the response.
        
        Returns:
            Tuple of (async generator of response chunks, error status code or None)
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "stream": True,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async def stream_generator():
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST",
                    self._get_api_url(),
                    headers=self._get_headers(),
                    json=payload
                ) as response:
                    if response.status_code != 200:
                        # Yield error information to be handled by caller
                        yield f"__ERROR__:{response.status_code}"
                        return
                    
                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data = line[6:]  # Remove "data: " prefix
                            if data == "[DONE]":
                                break
                            yield data
        
        return stream_generator(), None
    
    async def check_rate_limit(self, prompt: str) -> Tuple[bool, int]:
        """
        Check if the account is rate limited by making a test request.
        
        Returns:
            Tuple of (is_ok, status_code)
        """
        payload = {
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_tokens": 10
        }
        
        async with httpx.AsyncClient(timeout=10) as client:
            try:
                response = await client.post(
                    self._get_api_url(),
                    headers=self._get_headers(),
                    json=payload
                )
                return response.status_code == 200, response.status_code
            except Exception:
                return False, 500

    async def ask_non_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.7
    ) -> Tuple[Optional[str], int]:
        """
        Send a prompt to Cloudflare AI and get a non-streaming response.
        
        Returns:
            Tuple of (response text or None, status code)
        """
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "stream": False,
            "max_tokens": max_tokens,
            "temperature": temperature
        }
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            try:
                response = await client.post(
                    self._get_api_url(),
                    headers=self._get_headers(),
                    json=payload
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "result" in result and "response" in result["result"]:
                        return result["result"]["response"], 200
                    return str(result), 200
                
                return None, response.status_code
            except Exception as e:
                return str(e), 500
