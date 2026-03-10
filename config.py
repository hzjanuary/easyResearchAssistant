"""
Configuration module for easyResearchAssistant
Manages Cloudflare accounts and application settings
"""
import os
from typing import List, Dict
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class CloudflareAccount(BaseModel):
    """Cloudflare account configuration"""
    account_id: str
    api_token: str
    name: str = ""
    is_active: bool = True


class Settings(BaseModel):
    """Application settings"""
    access_password: str = os.getenv("ACCESS_PASSWORD", "family_secret_2024")
    rotation_strategy: str = os.getenv("ROTATION_STRATEGY", "round_robin")  # round_robin or random
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "60"))
    cloudflare_model: str = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3-8b-instruct")


# Load Cloudflare accounts from environment variables
def load_accounts_from_env() -> List[CloudflareAccount]:
    """
    Load Cloudflare accounts from environment variables.
    Expects: CLOUDFLARE_ACCOUNT_1_ID, CLOUDFLARE_ACCOUNT_1_TOKEN, etc.
    """
    accounts = []
    for i in range(1, 11):  # Support up to 10 accounts
        account_id = os.getenv(f"CLOUDFLARE_ACCOUNT_{i}_ID")
        api_token = os.getenv(f"CLOUDFLARE_ACCOUNT_{i}_TOKEN")
        name = os.getenv(f"CLOUDFLARE_ACCOUNT_{i}_NAME", f"Account_{i}")
        
        if account_id and api_token:
            accounts.append(CloudflareAccount(
                account_id=account_id,
                api_token=api_token,
                name=name,
                is_active=True
            ))
    
    return accounts


# Default accounts for development (replace with real credentials)
DEFAULT_ACCOUNTS: List[Dict] = [
    {
        "account_id": "your_account_id_1",
        "api_token": "your_api_token_1",
        "name": "Account_1"
    },
    {
        "account_id": "your_account_id_2",
        "api_token": "your_api_token_2",
        "name": "Account_2"
    },
    {
        "account_id": "your_account_id_3",
        "api_token": "your_api_token_3",
        "name": "Account_3"
    },
    {
        "account_id": "your_account_id_4",
        "api_token": "your_api_token_4",
        "name": "Account_4"
    },
    {
        "account_id": "your_account_id_5",
        "api_token": "your_api_token_5",
        "name": "Account_5"
    },
]


def get_accounts() -> List[CloudflareAccount]:
    """Get list of Cloudflare accounts, preferring env vars over defaults"""
    env_accounts = load_accounts_from_env()
    if env_accounts:
        return env_accounts
    
    # Fallback to default accounts (for development)
    return [CloudflareAccount(**acc, is_active=True) for acc in DEFAULT_ACCOUNTS]


settings = Settings()
