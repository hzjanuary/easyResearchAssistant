"""Provider Manager - Distributed Inference Node Orchestration."""
import json
import random
import asyncio
import logging
import os
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any, Deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque

logger = logging.getLogger(__name__)

# In-memory log buffer for monitoring dashboard
LOG_BUFFER_SIZE = 50
log_buffer: Deque[Dict[str, Any]] = deque(maxlen=LOG_BUFFER_SIZE)


def _add_log(level: str, message: str, node_name: Optional[str] = None):
    """Add a log entry to the buffer and standard logger"""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "level": level,
        "message": message,
        "node": node_name
    }
    log_buffer.append(entry)
    
    log_func = getattr(logger, level.lower(), logger.info)
    log_func(message)


def get_recent_logs(count: int = 10) -> List[Dict[str, Any]]:
    """Get the most recent log entries"""
    return list(log_buffer)[-count:]


def clear_logs():
    """Clear the log buffer"""
    log_buffer.clear()


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
    """Represents a distributed inference endpoint."""
    node_id: str
    node_type: NodeType
    name: str
    endpoint: str
    credentials: Dict[str, str] = field(default_factory=dict)
    status: NodeStatus = NodeStatus.HEALTHY
    priority: int = 1  # Lower = higher priority
    request_count: int = 0
    error_count: int = 0
    success_count: int = 0
    rate_limit_count: int = 0
    last_used: Optional[datetime] = None
    last_error: Optional[datetime] = None
    cooldown_until: Optional[datetime] = None
    cooldown_started: Optional[datetime] = None
    average_response_time: float = 0.0
    _response_times: List[float] = field(default_factory=list)
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
    
    def record_success(self, response_time: float = 0.0):
        """Record a successful request"""
        self.request_count += 1
        self.success_count += 1
        self.last_used = datetime.now()
        self.status = NodeStatus.HEALTHY
        self.error_count = max(0, self.error_count - 1)
        
        if response_time > 0:
            self._response_times.append(response_time)
            if len(self._response_times) > 100:
                self._response_times = self._response_times[-100:]
            self.average_response_time = sum(self._response_times) / len(self._response_times)
        
        _add_log("INFO", f"Node {self.name} completed request successfully", self.name)
    
    def record_error(self, is_rate_limit: bool = False, cooldown_minutes: int = 30):
        """Record a failed request"""
        self.error_count += 1
        self.request_count += 1
        self.last_error = datetime.now()
        
        if is_rate_limit:
            self.rate_limit_count += 1
            self.status = NodeStatus.RATE_LIMITED
            self.cooldown_started = datetime.now()
            self.cooldown_until = datetime.now() + timedelta(minutes=cooldown_minutes)
            _add_log("WARNING", f"Node {self.name} rate limited (429), cooldown: {cooldown_minutes}min", self.name)
        elif self.error_count >= 3:
            self.status = NodeStatus.UNAVAILABLE
            self.cooldown_started = datetime.now()
            self.cooldown_until = datetime.now() + timedelta(minutes=5)
            _add_log("ERROR", f"Node {self.name} marked unavailable after {self.error_count} errors", self.name)
        else:
            self.status = NodeStatus.DEGRADED
            _add_log("WARNING", f"Node {self.name} degraded, error count: {self.error_count}", self.name)
    
    def reset_status(self):
        """Reset node to healthy status"""
        self.status = NodeStatus.HEALTHY
        self.error_count = 0
        self.cooldown_until = None
        self.cooldown_started = None
        _add_log("INFO", f"Node {self.name} reset to healthy status", self.name)
    
    @property
    def cooldown_remaining_seconds(self) -> int:
        """Get remaining cooldown time in seconds"""
        if not self.cooldown_until:
            return 0
        remaining = (self.cooldown_until - datetime.now()).total_seconds()
        return max(0, int(remaining))
    
    @property
    def display_status(self) -> str:
        """Human-readable status for dashboard"""
        if self.status == NodeStatus.HEALTHY:
            return "Active"
        elif self.status == NodeStatus.RATE_LIMITED:
            return f"Cooldown ({self.cooldown_remaining_seconds}s)"
        elif self.status == NodeStatus.UNAVAILABLE:
            return "Offline"
        else:
            return "Degraded"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage"""
        if self.request_count == 0:
            return 100.0
        return (self.success_count / self.request_count) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize node for status reporting (excludes sensitive data)"""
        return {
            "node_id": self.node_id[:8] + "...",
            "name": self.name,
            "type": self.node_type.value,
            "status": self.status.value,
            "display_status": self.display_status,
            "priority": self.priority,
            "request_count": self.request_count,
            "success_count": self.success_count,
            "error_count": self.error_count,
            "rate_limit_count": self.rate_limit_count,
            "success_rate": round(self.success_rate, 1),
            "average_response_time": round(self.average_response_time, 2),
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "last_error": self.last_error.isoformat() if self.last_error else None,
            "cooldown_remaining": self.cooldown_remaining_seconds,
            "is_available": self.is_available
        }
    
    def to_monitoring_dict(self) -> Dict[str, Any]:
        """Extended serialization for monitoring dashboard"""
        base = self.to_dict()
        base.update({
            "cooldown_started": self.cooldown_started.isoformat() if self.cooldown_started else None,
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
        })
        return base


class ProviderManagerInterface(ABC):
    """Abstract interface for provider management."""
    
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
    """Orchestrates distributed inference nodes with intelligent selection."""
    
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
    
    def mark_node_failed(self, node_id: str, is_rate_limit: bool = False, cooldown_minutes: int = 30):
        """Mark a node as failed due to error or rate limiting"""
        for node in self._nodes:
            if node.node_id == node_id or node.account_id == node_id:
                node.record_error(is_rate_limit, cooldown_minutes)
                return
        logger.warning(f"Node not found: {node_id}")
    
    def mark_node_success(self, node_id: str, response_time: float = 0.0):
        """Record successful request for a node"""
        for node in self._nodes:
            if node.node_id == node_id or node.account_id == node_id:
                node.record_success(response_time)
                return
    
    def _attempt_recovery(self):
        """Attempt to recover nodes that have been in cooldown"""
        now = datetime.now()
        for node in self._nodes:
            if node.cooldown_until and now >= node.cooldown_until:
                _add_log("INFO", f"Auto-recovering node: {node.name} after cooldown", node.name)
                node.status = NodeStatus.DEGRADED  # Not fully healthy yet
                node.cooldown_until = None
                node.cooldown_started = None
    
    def reset_all_nodes(self):
        """Reset all nodes to healthy status"""
        for node in self._nodes:
            node.reset_status()
        _add_log("INFO", "All nodes reset to healthy status")
    
    def set_strategy(self, strategy: str):
        """Change the selection strategy"""
        try:
            self._strategy = SelectionStrategy(strategy)
            _add_log("INFO", f"Selection strategy changed to: {strategy}")
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
    
    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get detailed monitoring statistics for dashboard"""
        total_requests = sum(n.request_count for n in self._nodes)
        total_errors = sum(n.error_count for n in self._nodes)
        total_rate_limits = sum(n.rate_limit_count for n in self._nodes)
        
        fallback_stats = None
        if self._fallback_node:
            fallback_stats = self._fallback_node.to_monitoring_dict()
        
        nodes_by_status = {
            "active": [],
            "cooldown": [],
            "offline": [],
            "degraded": []
        }
        
        for node in self._nodes:
            node_data = node.to_monitoring_dict()
            if node.status == NodeStatus.HEALTHY:
                nodes_by_status["active"].append(node_data)
            elif node.status == NodeStatus.RATE_LIMITED:
                nodes_by_status["cooldown"].append(node_data)
            elif node.status == NodeStatus.UNAVAILABLE:
                nodes_by_status["offline"].append(node_data)
            else:
                nodes_by_status["degraded"].append(node_data)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "strategy": self._strategy.value,
            "summary": {
                "total_nodes": len(self._nodes),
                "available_nodes": len(self.available_nodes),
                "total_requests": total_requests,
                "total_errors": total_errors,
                "total_rate_limits": total_rate_limits,
                "overall_success_rate": round((total_requests - total_errors) / max(total_requests, 1) * 100, 1),
                "has_fallback": self.has_fallback
            },
            "nodes_by_status": nodes_by_status,
            "all_nodes": [n.to_monitoring_dict() for n in self._nodes],
            "fallback": fallback_stats,
            "recent_logs": get_recent_logs(10)
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
