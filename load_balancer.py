"""
Load Balancer module for token rotation
Supports Round Robin and Random selection strategies
"""
import random
import threading
from typing import Optional, List
from config import CloudflareAccount, get_accounts, settings


class LoadBalancer:
    """
    Load balancer for distributing requests across multiple Cloudflare accounts.
    Supports Round Robin and Random selection strategies.
    """
    
    def __init__(self, strategy: str = "round_robin"):
        self._accounts: List[CloudflareAccount] = get_accounts()
        self._strategy = strategy
        self._current_index = 0
        self._lock = threading.Lock()
        self._failed_accounts: set = set()  # Track temporarily failed accounts
    
    @property
    def accounts(self) -> List[CloudflareAccount]:
        """Get list of active accounts"""
        return [acc for acc in self._accounts if acc.is_active and acc.account_id not in self._failed_accounts]
    
    @property
    def all_accounts(self) -> List[CloudflareAccount]:
        """Get all accounts regardless of status"""
        return self._accounts
    
    def get_next_account(self) -> Optional[CloudflareAccount]:
        """
        Get the next account based on the rotation strategy.
        Returns None if no active accounts are available.
        """
        active_accounts = self.accounts
        
        if not active_accounts:
            # Reset failed accounts and try again
            self._failed_accounts.clear()
            active_accounts = self.accounts
            if not active_accounts:
                return None
        
        if self._strategy == "random":
            return self._get_random_account(active_accounts)
        else:
            return self._get_round_robin_account(active_accounts)
    
    def _get_round_robin_account(self, accounts: List[CloudflareAccount]) -> CloudflareAccount:
        """Round Robin selection - cycles through accounts sequentially"""
        with self._lock:
            if self._current_index >= len(accounts):
                self._current_index = 0
            
            account = accounts[self._current_index]
            self._current_index = (self._current_index + 1) % len(accounts)
            return account
    
    def _get_random_account(self, accounts: List[CloudflareAccount]) -> CloudflareAccount:
        """Random selection - picks a random account"""
        return random.choice(accounts)
    
    def mark_account_failed(self, account_id: str):
        """Mark an account as temporarily failed (e.g., due to rate limiting)"""
        with self._lock:
            self._failed_accounts.add(account_id)
    
    def mark_account_recovered(self, account_id: str):
        """Mark an account as recovered and available again"""
        with self._lock:
            self._failed_accounts.discard(account_id)
    
    def reset_failed_accounts(self):
        """Reset all failed accounts to active status"""
        with self._lock:
            self._failed_accounts.clear()
    
    def get_status(self) -> dict:
        """Get current load balancer status"""
        return {
            "strategy": self._strategy,
            "total_accounts": len(self._accounts),
            "active_accounts": len(self.accounts),
            "failed_accounts": list(self._failed_accounts),
            "current_index": self._current_index
        }
    
    def set_strategy(self, strategy: str):
        """Change the rotation strategy"""
        if strategy in ["round_robin", "random"]:
            self._strategy = strategy
        else:
            raise ValueError(f"Invalid strategy: {strategy}. Use 'round_robin' or 'random'")
    
    def reload_accounts(self):
        """Reload accounts from configuration"""
        with self._lock:
            self._accounts = get_accounts()
            self._current_index = 0
            self._failed_accounts.clear()


# Global load balancer instance
load_balancer = LoadBalancer(strategy=settings.rotation_strategy)
