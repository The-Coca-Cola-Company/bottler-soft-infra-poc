"""
MCP Discovery Service
=====================

Discovers and manages MCP servers dynamically without hardcoding.
Supports various discovery mechanisms.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

import os
import logging
import asyncio
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
import json
from enum import Enum

# HTTP client with fallback
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

from .mcp_config import MCPServerConfig

logger = logging.getLogger(__name__)

class DiscoveryMethod(Enum):
    """Discovery methods for finding MCP servers"""
    STATIC = "static"  # From configuration
    DNS_SRV = "dns_srv"  # DNS SRV records
    HTTP_REGISTRY = "http_registry"  # HTTP-based registry
    COSMOS_DB = "cosmos_db"  # Azure Cosmos DB registry
    KUBERNETES = "kubernetes"  # Kubernetes service discovery
    CONSUL = "consul"  # Consul service discovery

class MCPServerInfo:
    """Information about a discovered MCP server"""
    def __init__(self, 
                 name: str,
                 url: str,
                 metadata: Optional[Dict[str, Any]] = None):
        self.name = name
        self.url = url
        self.metadata = metadata or {}
        self.discovered_at = datetime.utcnow()
        self.last_seen = datetime.utcnow()
        self.capabilities: List[str] = []
        self.health_status = "unknown"
        self.version: Optional[str] = None
        self.tags: Set[str] = set()
    
    def to_server_config(self) -> MCPServerConfig:
        """Convert to MCPServerConfig"""
        return MCPServerConfig(
            name=self.name,
            url=self.url,
            transport=self.metadata.get("transport", "sse"),
            auth_type=self.metadata.get("auth_type"),
            auth_credentials=self.metadata.get("auth_credentials"),
            timeout=self.metadata.get("timeout", 30000),
            retry_count=self.metadata.get("retry_count", 3),
            retry_delay=self.metadata.get("retry_delay", 1000),
            health_check_endpoint=self.metadata.get("health_check_endpoint", "health"),
            metadata=self.metadata
        )

class MCPDiscoveryService:
    """
    Service for discovering MCP servers dynamically.
    Supports multiple discovery methods without hardcoding.
    """
    
    def __init__(self, 
                 discovery_methods: Optional[List[DiscoveryMethod]] = None,
                 registry_url: Optional[str] = None,
                 cosmos_config: Optional[Dict[str, Any]] = None):
        """
        Initialize discovery service.
        
        Args:
            discovery_methods: List of discovery methods to use
            registry_url: URL for HTTP registry if used
            cosmos_config: Cosmos DB configuration if used
        """
        self.discovery_methods = discovery_methods or [DiscoveryMethod.STATIC]
        self.registry_url = registry_url or os.getenv("MCP_REGISTRY_URL")
        self.cosmos_config = cosmos_config
        self.discovered_servers: Dict[str, MCPServerInfo] = {}
        self._discovery_task: Optional[asyncio.Task] = None
        self._discovery_interval = 60  # seconds
        self._session: Optional[aiohttp.ClientSession] = None
        self._callbacks: List[callable] = []
    
    async def start_discovery(self, interval: Optional[int] = None):
        """Start continuous discovery process"""
        if self._discovery_task and not self._discovery_task.done():
            logger.warning("Discovery already running")
            return
        
        if interval:
            self._discovery_interval = interval
        
        self._discovery_task = asyncio.create_task(self._discovery_loop())
        logger.info(f"Started discovery with interval {self._discovery_interval}s")
    
    async def stop_discovery(self):
        """Stop continuous discovery process"""
        if self._discovery_task:
            self._discovery_task.cancel()
            try:
                await self._discovery_task
            except asyncio.CancelledError:
                pass
            self._discovery_task = None
        
        if self._session:
            await self._session.close()
            self._session = None
        
        logger.info("Stopped discovery")
    
    async def discover_once(self) -> Dict[str, MCPServerInfo]:
        """Perform one-time discovery"""
        discovered = {}
        
        for method in self.discovery_methods:
            try:
                if method == DiscoveryMethod.STATIC:
                    servers = await self._discover_static()
                elif method == DiscoveryMethod.HTTP_REGISTRY:
                    servers = await self._discover_http_registry()
                elif method == DiscoveryMethod.COSMOS_DB:
                    servers = await self._discover_cosmos_db()
                elif method == DiscoveryMethod.DNS_SRV:
                    servers = await self._discover_dns_srv()
                elif method == DiscoveryMethod.KUBERNETES:
                    servers = await self._discover_kubernetes()
                elif method == DiscoveryMethod.CONSUL:
                    servers = await self._discover_consul()
                else:
                    logger.warning(f"Unknown discovery method: {method}")
                    continue
                
                # Merge discovered servers
                for server in servers:
                    discovered[server.name] = server
                    
            except Exception as e:
                logger.error(f"Error in {method.value} discovery: {str(e)}")
        
        # Update internal registry
        self._update_discovered_servers(discovered)
        
        # Notify callbacks
        await self._notify_callbacks(discovered)
        
        return discovered
    
    def register_callback(self, callback: callable):
        """Register callback for server discovery events"""
        self._callbacks.append(callback)
    
    def unregister_callback(self, callback: callable):
        """Unregister callback"""
        if callback in self._callbacks:
            self._callbacks.remove(callback)
    
    def get_server(self, name: str) -> Optional[MCPServerInfo]:
        """Get a specific discovered server"""
        return self.discovered_servers.get(name)
    
    def list_servers(self, tags: Optional[Set[str]] = None) -> List[MCPServerInfo]:
        """List discovered servers, optionally filtered by tags"""
        servers = list(self.discovered_servers.values())
        
        if tags:
            servers = [s for s in servers if tags.issubset(s.tags)]
        
        return servers
    
    async def validate_server(self, server_info: MCPServerInfo) -> bool:
        """Validate that a server is accessible and supports MCP"""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available - cannot validate servers")
            return False
            
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            # Try to get server capabilities
            url = f"{server_info.url}/mcp/capabilities"
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    server_info.capabilities = data.get("capabilities", [])
                    server_info.version = data.get("version")
                    server_info.health_status = "healthy"
                    return True
                else:
                    server_info.health_status = "unhealthy"
                    return False
                    
        except Exception as e:
            logger.warning(f"Failed to validate server {server_info.name}: {str(e)}")
            server_info.health_status = "error"
            return False
    
    async def _discovery_loop(self):
        """Continuous discovery loop"""
        while True:
            try:
                await self.discover_once()
                await asyncio.sleep(self._discovery_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in discovery loop: {str(e)}")
                await asyncio.sleep(self._discovery_interval)
    
    async def _discover_static(self) -> List[MCPServerInfo]:
        """Discover servers from static configuration"""
        servers = []
        
        # Get from environment variables
        for key, value in os.environ.items():
            if key.startswith("MCP_SERVER_") and key.endswith("_URL"):
                # Extract server name from key
                parts = key.split("_")
                if len(parts) >= 4:
                    server_name = parts[2].lower()
                    
                    # Get metadata
                    metadata = {
                        "transport": os.getenv(f"MCP_SERVER_{parts[2]}_TRANSPORT", "sse"),
                        "auth_type": os.getenv(f"MCP_SERVER_{parts[2]}_AUTH_TYPE"),
                        "source": "environment"
                    }
                    
                    server = MCPServerInfo(
                        name=server_name,
                        url=value,
                        metadata=metadata
                    )
                    
                    # Add tags
                    tags_str = os.getenv(f"MCP_SERVER_{parts[2]}_TAGS", "")
                    if tags_str:
                        server.tags = set(tags_str.split(","))
                    
                    servers.append(server)
        
        logger.info(f"Discovered {len(servers)} servers from static configuration")
        return servers
    
    async def _discover_http_registry(self) -> List[MCPServerInfo]:
        """Discover servers from HTTP registry"""
        if not self.registry_url or not AIOHTTP_AVAILABLE:
            return []
        
        servers = []
        
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            async with self._session.get(
                self.registry_url,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    for server_data in data.get("servers", []):
                        server = MCPServerInfo(
                            name=server_data["name"],
                            url=server_data["url"],
                            metadata=server_data.get("metadata", {})
                        )
                        
                        # Set additional properties
                        server.capabilities = server_data.get("capabilities", [])
                        server.version = server_data.get("version")
                        server.tags = set(server_data.get("tags", []))
                        
                        servers.append(server)
                    
                    logger.info(f"Discovered {len(servers)} servers from HTTP registry")
                else:
                    logger.error(f"HTTP registry returned status {response.status}")
                    
        except Exception as e:
            logger.error(f"Error accessing HTTP registry: {str(e)}")
        
        return servers
    
    async def _discover_cosmos_db(self) -> List[MCPServerInfo]:
        """Discover servers from Cosmos DB"""
        if not self.cosmos_config:
            return []
        
        servers = []
        
        try:
            # Try to import Azure Cosmos client
            try:
                from azure.cosmos import CosmosClient
            except ImportError:
                logger.warning("Azure Cosmos SDK not available")
                return []
            
            # Create Cosmos client
            client = CosmosClient(
                url=self.cosmos_config["endpoint"],
                credential=self.cosmos_config["key"]
            )
            
            database = client.get_database_client(self.cosmos_config["database"])
            container = database.get_container_client(self.cosmos_config.get("container", "mcp_servers"))
            
            # Query for active servers
            query = "SELECT * FROM c WHERE c.status = 'active' AND c.type = 'mcp_server'"
            
            for item in container.query_items(query=query, enable_cross_partition_query=True):
                server = MCPServerInfo(
                    name=item["name"],
                    url=item["url"],
                    metadata=item.get("metadata", {})
                )
                
                # Set additional properties
                server.capabilities = item.get("capabilities", [])
                server.version = item.get("version")
                server.tags = set(item.get("tags", []))
                
                servers.append(server)
            
            logger.info(f"Discovered {len(servers)} servers from Cosmos DB")
            
        except Exception as e:
            logger.error(f"Error accessing Cosmos DB: {str(e)}")
        
        return servers
    
    async def _discover_dns_srv(self) -> List[MCPServerInfo]:
        """Discover servers from DNS SRV records"""
        # This would use aiodns to query SRV records
        # Example: _mcp._tcp.example.com
        logger.info("DNS SRV discovery not implemented yet")
        return []
    
    async def _discover_kubernetes(self) -> List[MCPServerInfo]:
        """Discover servers from Kubernetes services"""
        # This would use kubernetes-asyncio to query services
        logger.info("Kubernetes discovery not implemented yet")
        return []
    
    async def _discover_consul(self) -> List[MCPServerInfo]:
        """Discover servers from Consul"""
        # This would use aiohttp to query Consul API
        logger.info("Consul discovery not implemented yet")
        return []
    
    def _update_discovered_servers(self, discovered: Dict[str, MCPServerInfo]):
        """Update internal registry with discovered servers"""
        # Mark existing servers as last seen
        current_time = datetime.utcnow()
        
        # Update or add discovered servers
        for name, server in discovered.items():
            if name in self.discovered_servers:
                # Update existing
                existing = self.discovered_servers[name]
                existing.last_seen = current_time
                existing.url = server.url  # URL might have changed
                existing.metadata.update(server.metadata)
                existing.capabilities = server.capabilities or existing.capabilities
                existing.tags.update(server.tags)
            else:
                # Add new
                self.discovered_servers[name] = server
                logger.info(f"New MCP server discovered: {name}")
        
        # Remove servers not seen recently (optional)
        stale_threshold = timedelta(minutes=10)
        for name in list(self.discovered_servers.keys()):
            if name not in discovered:
                server = self.discovered_servers[name]
                if current_time - server.last_seen > stale_threshold:
                    del self.discovered_servers[name]
                    logger.info(f"Removed stale MCP server: {name}")
    
    async def _notify_callbacks(self, discovered: Dict[str, MCPServerInfo]):
        """Notify registered callbacks about discovery events"""
        for callback in self._callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(discovered)
                else:
                    callback(discovered)
            except Exception as e:
                logger.error(f"Error in discovery callback: {str(e)}")
    
    def export_servers(self) -> List[Dict[str, Any]]:
        """Export discovered servers as list of dictionaries"""
        return [
            {
                "name": server.name,
                "url": server.url,
                "discovered_at": server.discovered_at.isoformat(),
                "last_seen": server.last_seen.isoformat(),
                "capabilities": server.capabilities,
                "health_status": server.health_status,
                "version": server.version,
                "tags": list(server.tags),
                "metadata": server.metadata
            }
            for server in self.discovered_servers.values()
        ]