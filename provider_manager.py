"""
Provider Manager - Distributed Inference Node Orchestration
============================================================
Manages multiple inference providers (Cloudflare Workers AI nodes) with
intelligent load balancing, health monitoring, and automatic failover.

Architecture:
- InferenceNode: Abstract representation of a compute endpoint
- ProviderManager: Orchestrates node selection and health tracking
- Supports Round Robin and Random selection strategies

This module follows the Single Responsibility Principle (SRP) by focusing
solely on provider orchestration and selection logic.
"""
import json
import random
import asyncio
import logging
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Supported inference node types"""
    CLOUDFLARE = "cloudflare"
    OLLAMA = "ollama"


class NodeStatus(Enum):
    """Health status of an inference node"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"


class SelectionStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    LEAST_USED = "least_used"


@dataclass
class InferenceNode:
    """
    Represents a distributed inference endpoint.
    
    This abstraction allows treating different provider types uniformly,
    enabling seamless failover between cloud and local inference.
    """
    node_id: str
    node_type: NodeType
    name: str
    endpoint: str
    credentials: Dict[str, str] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.HEALTHY
    priority: int = 1  # Lower = higher priority
    request_count: int = 0
    error_count: int = 0
    last_used: Optional[datetime] = None
    last_error: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for requests"""
        if self.status == NodeStatus.UNAVAILABLE:
            return False
        if self.cooldown_until and datetime.now() < self.cooldown_until:
            return False
        return True
    
    @property
    def account_id(self) -> str:
        """Get Cloudflare account ID (compatibility property)"""
        return self.credentials.get("account_id", self.node_id)
    
    @property
    def api_token(self) -> str:
        """Get API token (compatibility property)"""
        return self.credentials.get("api_token", "")
    
    def record_success(self):
        """Record a successful request"""
        self.request_count += 1
        self.last_used = datetime.now()
        self.status = NodeStatus.HEALTHY
        # Decay error count on success
        self.error_count = max(0, self.error_count - 1)
    
    def record_error(self, is_rate_limit: bool = False):
        """Record a failed request"""
        self.error_count += 1
        self.last_error = datetime.now()
        
        if is_rate_limit:
            self.status = NodeStatus.RATE_LIMITED
            # Cooldown for rate-limited nodes (exponential backoff)
            cooldown_seconds = min(300, 30 * (2 ** min(self.error_count - 1, 4)))
            self.cooldown_until = datetime.now() + timedelta(seconds=cooldown_seconds)
            logger.warning(f"Node {self.name} rate limited, cooldown: {cooldown_seconds}s")
        elif self.error_count >= 3:
            self.status = NodeStatus.UNAVAILABLE
            self.cooldown_until = datetime.now() + timedelta(minutes=5)
            logger.error(f"Node {self.name} marked unavailable after {self.error_count} errors")
        else:
            self.status = NodeStatus.DEGRADED
    
    def reset_status(self):
        """Reset node to healthy status"""
        self.status = NodeStatus.HEALTHY
        self.error_count = 0
        self.cooldown_until = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node for status reporting (excludes sensitive data)"""
        return {
            "node_id": self.node_id[:8] + "...",
            "name": self.name,
            "type": self.node_type.value,
            "status": self.status.value,
            "priority": self.priority,
            "request_count": self.request_count,
            "is_available": self.is_available
        }


class ProviderManagerInterface(ABC):
    """Abstract interface for provider management (Interface Segregation)"""
    
    @abstractmethod
    def get_next_node(self) -> Optional[InferenceNode]:
        """Get the next available inference node"""
        pass
    
    @abstractmethod
    def mark_node_failed(self, node_id: str, is_rate_limit: bool = False):
        """Mark a node as failed"""
        pass
    
    @abstractmethod
    def get_fallback_node(self) -> Optional[InferenceNode]:
        """Get the local fallback node"""
        pass


class ProviderManager(ProviderManagerInterface):
    """
    Orchestrates distributed inference nodes with intelligent selection.
    
    Features:
    - Multiple selection strategies (Round Robin, Random, Least Used)
    - Automatic health tracking and recovery
    - Local fallback support for high availability
    - Thread-safe operations
    
    This class follows Open/Closed Principle - extend via new strategies
    without modifying existing code.
    """
    
    def __init__(
        self,
        strategy: SelectionStrategy = SelectionStrategy.ROUND_ROBIN,
        config_path: Optional[Path] = None
    ):
        self._nodes: List[InferenceNode] = []
        self._fallback_node: Optional[InferenceNode] = None
        self._strategy = strategy
        self._current_index = 0
        self._lock = asyncio.Lock()
        
        # Load configuration
        if config_path:
            self._load_from_file(config_path)
    
    def _load_from_file(self, config_path: Path):
        """Load provider configuration from JSON file"""
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            return
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load cloud providers
            for provider in config.get("providers", []):
                node = InferenceNode(
                    node_id=provider["account_id"],
                    node_type=NodeType.CLOUDFLARE,
                    name=provider.get("name", f"Node-{provider['account_id'][:8]}"),
                    endpoint=provider.get("endpoint", "https://api.cloudflare.com"),
                    credentials={
                        "account_id": provider["account_id"],
                        "api_token": provider["api_token"]
                    },
                    priority=provider.get("priority", 1),
                    metadata=provider.get("metadata", {})
                )
                self._nodes.append(node)
            
            # Load local fallback
            if "local_fallback" in config:
                fallback = config["local_fallback"]
                self._fallback_node = InferenceNode(
                    node_id="local-ollama",
                    node_type=NodeType.OLLAMA,
                    name=fallback.get("name", "Local Ollama"),
                    endpoint=fallback.get("endpoint", "http://localhost:11434"),
                    credentials={},
                    priority=999,  # Lowest priority (fallback)
                    metadata={
                        "model": fallback.get("model", "llama3"),
                        "max_concurrent": fallback.get("max_concurrent", 1)
                    }
                )
            
            logger.info(f"Loaded {len(self._nodes)} inference nodes, fallback: {self._fallback_node is not None}")
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
    
    def add_node(self, node: InferenceNode):
        """Add an inference node to the pool"""
        self._nodes.append(node)
        logger.info(f"Added node: {node.name} ({node.node_type.value})")
    
    def set_fallback(self, node: InferenceNode):
        """Set the local fallback node"""
        self._fallback_node = node
        logger.info(f"Set fallback node: {node.name}")
    
    @property
    def available_nodes(self) -> List[InferenceNode]:
        """Get list of currently available nodes"""
        return [n for n in self._nodes if n.is_available]
    
    @property
    def all_nodes(self) -> List[InferenceNode]:
        """Get all registered nodes"""
        return self._nodes
    
    @property
    def has_fallback(self) -> bool:
        """Check if local fallback is configured"""
        return self._fallback_node is not None
    
    def get_next_node(self) -> Optional[InferenceNode]:
        """
        Select the next inference node based on current strategy.
        Returns None if no nodes are available.
        """
        available = self.available_nodes
        
        if not available:
            # Try to recover rate-limited nodes
            self._attempt_recovery()
            available = self.available_nodes
            if not available:
                logger.warning("No available inference nodes")
                return None
        
        # Sort by priority
        available.sort(key=lambda n: n.priority)
        
        if self._strategy == SelectionStrategy.RANDOM:
            return random.choice(available)
        elif self._strategy == SelectionStrategy.LEAST_USED:
            return min(available, key=lambda n: n.request_count)
        else:  # ROUND_ROBIN
            if self._current_index >= len(available):
                self._current_index = 0
            node = available[self._current_index]
            self._current_index = (self._current_index + 1) % len(available)
            return node
    
    def get_fallback_node(self) -> Optional[InferenceNode]:
        """Get the local fallback node (Ollama)"""
        return self._fallback_node
    
    def mark_node_failed(self, node_id: str, is_rate_limit: bool = False):
        """Mark a node as failed due to error or rate limiting"""
        for node in self._nodes:
            if node.node_id == node_id or node.account_id == node_id:
                node.record_error(is_rate_limit)
                return
        logger.warning(f"Node not found: {node_id}")
    
    def mark_node_success(self, node_id: str):
        """Record successful request for a node"""
        for node in self._nodes:
            if node.node_id == node_id or node.account_id == node_id:
                node.record_success()
                return
    
    def _attempt_recovery(self):
        """Attempt to recover nodes that have been in cooldown"""
        now = datetime.now()
        for node in self._nodes:
            if node.cooldown_until and now >= node.cooldown_until:
                logger.info(f"Recovering node: {node.name}")
                node.status = NodeStatus.DEGRADED  # Not fully healthy yet
                node.cooldown_until = None
    
    def reset_all_nodes(self):
        """Reset all nodes to healthy status"""
        for node in self._nodes:
            node.reset_status()
        logger.info("All nodes reset to healthy status")
    
    def set_strategy(self, strategy: str):
        """Change the selection strategy"""
        try:
            self._strategy = SelectionStrategy(strategy)
            logger.info(f"Strategy changed to: {strategy}")
        except ValueError:
            raise ValueError(f"Invalid strategy: {strategy}. Use: {[s.value for s in SelectionStrategy]}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status report"""
        return {
            "strategy": self._strategy.value,
            "total_nodes": len(self._nodes),
            "available_nodes": len(self.available_nodes),
            "has_fallback": self.has_fallback,
            "nodes": [n.to_dict() for n in self._nodes],
            "fallback": self._fallback_node.to_dict() if self._fallback_node else None
        }
    
    def reload_config(self, config_path: Path):
        """Reload configuration from file"""
        self._nodes.clear()
        self._fallback_node = None
        self._current_index = 0
        self._load_from_file(config_path)


def create_provider_manager_from_env() -> ProviderManager:
    """
    Factory function to create ProviderManager from environment variables.
    Provides backward compatibility with existing .env configuration.
    """
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    strategy_str = os.getenv("SELECTION_STRATEGY", "round_robin")
    try:
        strategy = SelectionStrategy(strategy_str)
    except ValueError:
        strategy = SelectionStrategy.ROUND_ROBIN
    
    manager = ProviderManager(strategy=strategy)
    
    # Load Cloudflare nodes from environment
    for i in range(1, 11):
        account_id = os.getenv(f"CLOUDFLARE_ACCOUNT_{i}_ID")
        api_token = os.getenv(f"CLOUDFLARE_ACCOUNT_{i}_TOKEN")
        name = os.getenv(f"CLOUDFLARE_ACCOUNT_{i}_NAME", f"Node-{i}")
        
        if account_id and api_token:
            node = InferenceNode(
                node_id=account_id,
                node_type=NodeType.CLOUDFLARE,
                name=name,
                endpoint="https://api.cloudflare.com",
                credentials={
                    "account_id": account_id,
                    "api_token": api_token
                },
                priority=i
            )
            manager.add_node(node)
    
    # Configure local fallback (Ollama)
    ollama_enabled = os.getenv("OLLAMA_ENABLED", "true").lower() == "true"
    if ollama_enabled:
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "llama3")
        
        fallback = InferenceNode(
            node_id="local-ollama",
            node_type=NodeType.OLLAMA,
            name="Local Ollama (RTX 3050)",
            endpoint=ollama_endpoint,
            credentials={},
            priority=999,
            metadata={
                "model": ollama_model,
                "max_concurrent": 1  # Lightweight for 4GB VRAM
            }
        )
        manager.set_fallback(fallback)
    
    return manager
