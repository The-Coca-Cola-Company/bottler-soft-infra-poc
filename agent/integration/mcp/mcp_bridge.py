"""
MCP Bridge - Core Bridge Implementation
=======================================

Main bridge between Azure Functions and MCP servers.
Orchestrates all components without hardcoding.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Tuple, TYPE_CHECKING
from datetime import datetime, timedelta
import json
import hashlib
from collections import OrderedDict

# Type checking imports
if TYPE_CHECKING:
    from .mcp_config import MCPConfiguration, MCPServerConfig
    from .mcp_client_manager import MCPClientManager
    from .mcp_protocol import MCPProtocolTranslator
    from .mcp_discovery import MCPDiscoveryService, DiscoveryMethod

logger = logging.getLogger(__name__)

class MCPCache:
    """Simple cache implementation for MCP responses"""
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.cache: OrderedDict[str, Tuple[Any, datetime]] = OrderedDict()
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get item from cache"""
        if key in self.cache:
            value, timestamp = self.cache[key]
            if datetime.utcnow() - timestamp < timedelta(seconds=self.ttl):
                self.hits += 1
                # Move to end (LRU)
                self.cache.move_to_end(key)
                return value
            else:
                # Expired
                del self.cache[key]
        
        self.misses += 1
        return None
    
    def set(self, key: str, value: Any):
        """Set item in cache"""
        # Remove oldest if at capacity
        if len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        
        self.cache[key] = (value, datetime.utcnow())
    
    def clear(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.hits = 0
        self.misses = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "ttl": self.ttl
        }

class MCPBridge:
    """
    Main MCP Bridge implementation.
    Coordinates all components to provide MCP functionality in Azure Functions.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize MCP Bridge.
        
        Args:
            config_path: Path to configuration file (optional)
        """
        # Import here to avoid circular imports
        from .mcp_config import MCPConfiguration
        from .mcp_client_manager import MCPClientManager
        from .mcp_protocol import MCPProtocolTranslator
        from .mcp_discovery import MCPDiscoveryService, DiscoveryMethod
        
        # Initialize configuration
        self.config = MCPConfiguration(config_path)
        
        # Initialize components
        self.protocol = MCPProtocolTranslator()
        self.client_manager = MCPClientManager(self.protocol)
        self.discovery_service = MCPDiscoveryService(
            discovery_methods=[DiscoveryMethod.STATIC, DiscoveryMethod.HTTP_REGISTRY]
        )
        
        # Initialize cache if enabled
        self.cache = MCPCache(
            max_size=1000,
            ttl=self.config.bridge_config.cache_ttl
        ) if self.config.bridge_config.cache_enabled else None
        
        # Metrics
        self.metrics = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "average_latency_ms": 0,
            "servers_active": 0,
            "last_discovery": None
        }
        
        # State
        self._initialized = False
        self._discovery_task: Optional[asyncio.Task] = None
    
    async def initialize(self):
        """Initialize the bridge and all components"""
        if self._initialized:
            logger.warning("Bridge already initialized")
            return
        
        logger.info("Initializing MCP Bridge...")
        
        # Initialize servers from configuration
        for server_name, server_config in self.config.servers.items():
            await self.client_manager.add_server(server_config)
            logger.info(f"Added server from config: {server_name}")
        
        # Start discovery if enabled
        if self.config.bridge_config.enable_discovery:
            await self.start_discovery()
        
        # Register discovery callback
        self.discovery_service.register_callback(self._on_servers_discovered)
        
        self._initialized = True
        self.metrics["servers_active"] = len(self.client_manager.connections)
        
        logger.info(f"MCP Bridge initialized with {self.metrics['servers_active']} servers")
    
    async def shutdown(self):
        """Shutdown the bridge and cleanup resources"""
        logger.info("Shutting down MCP Bridge...")
        
        # Stop discovery
        if self._discovery_task:
            await self.discovery_service.stop_discovery()
        
        # Close all client connections
        await self.client_manager.close_all()
        
        # Clear cache
        if self.cache:
            self.cache.clear()
        
        self._initialized = False
        logger.info("MCP Bridge shutdown complete")
    
    async def execute_tool(self, 
                          server_name: str,
                          tool_name: str,
                          arguments: Optional[Dict[str, Any]] = None,
                          timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute a tool on a specific MCP server.
        
        Args:
            server_name: Name of the MCP server
            tool_name: Name of the tool to execute
            arguments: Tool arguments
            timeout: Request timeout in milliseconds
            
        Returns:
            Tool execution result
        """
        start_time = datetime.utcnow()
        self.metrics["requests_total"] += 1
        
        # Check cache if enabled
        cache_key = None
        if self.cache and self._is_cacheable_tool(tool_name):
            cache_key = self._generate_cache_key(server_name, tool_name, arguments)
            cached_result = self.cache.get(cache_key)
            if cached_result:
                logger.debug(f"Cache hit for {server_name}.{tool_name}")
                return cached_result
        
        # Create MCP request
        mcp_request = {
            "jsonrpc": "2.0",
            "id": f"{server_name}-{tool_name}-{start_time.timestamp()}",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments or {}
            }
        }
        
        try:
            # Execute request
            mcp_response = await self.client_manager.execute_request(
                server_name,
                mcp_request,
                timeout
            )
            
            # Process response
            if "error" in mcp_response:
                self.metrics["requests_failed"] += 1
                return {
                    "success": False,
                    "error": mcp_response["error"]
                }
            
            result = mcp_response.get("result", {})
            self.metrics["requests_success"] += 1
            
            # Update latency metric
            latency = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_latency_metric(latency)
            
            # Cache result if applicable
            if cache_key and self.cache:
                self.cache.set(cache_key, result)
            
            return {
                "success": True,
                "result": result,
                "latency_ms": latency,
                "server": server_name,
                "tool": tool_name
            }
            
        except Exception as e:
            self.metrics["requests_failed"] += 1
            logger.error(f"Error executing tool {tool_name} on {server_name}: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def list_tools(self, server_name: Optional[str] = None) -> Dict[str, Any]:
        """
        List available tools from one or all servers.
        
        Args:
            server_name: Specific server or None for all
            
        Returns:
            Dictionary of tools by server
        """
        try:
            tools_by_server = await self.client_manager.list_tools(server_name)
            
            # Format response
            result = {
                "success": True,
                "servers": {}
            }
            
            for srv, tools in tools_by_server.items():
                result["servers"][srv] = {
                    "tools": tools,
                    "count": len(tools)
                }
            
            result["total_tools"] = sum(
                len(tools) for tools in tools_by_server.values()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error listing tools: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_server(self, server_config: "MCPServerConfig") -> Dict[str, Any]:
        """
        Add a new MCP server dynamically.
        
        Args:
            server_config: Server configuration
            
        Returns:
            Operation result
        """
        try:
            # Add to configuration
            success = self.config.add_server(server_config)
            if not success:
                return {
                    "success": False,
                    "error": "Failed to add server to configuration"
                }
            
            # Add to client manager
            await self.client_manager.add_server(server_config)
            
            # Update metrics
            self.metrics["servers_active"] = len(self.client_manager.connections)
            
            return {
                "success": True,
                "server": server_config.name,
                "message": f"Server {server_config.name} added successfully"
            }
            
        except Exception as e:
            logger.error(f"Error adding server: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def remove_server(self, server_name: str) -> Dict[str, Any]:
        """
        Remove an MCP server.
        
        Args:
            server_name: Name of server to remove
            
        Returns:
            Operation result
        """
        try:
            # Remove from client manager
            success = await self.client_manager.remove_server(server_name)
            if not success:
                return {
                    "success": False,
                    "error": f"Server {server_name} not found"
                }
            
            # Remove from configuration
            self.config.remove_server(server_name)
            
            # Update metrics
            self.metrics["servers_active"] = len(self.client_manager.connections)
            
            return {
                "success": True,
                "message": f"Server {server_name} removed successfully"
            }
            
        except Exception as e:
            logger.error(f"Error removing server: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status of the bridge"""
        # Get server status
        server_status = await self.client_manager.get_server_status()
        
        # Get cache stats
        cache_stats = self.cache.get_stats() if self.cache else None
        
        # Get discovery info
        discovery_info = {
            "enabled": self.config.bridge_config.enable_discovery,
            "servers_discovered": len(self.discovery_service.discovered_servers),
            "last_discovery": self.metrics["last_discovery"]
        }
        
        return {
            "bridge": {
                "initialized": self._initialized,
                "version": "1.0.0",
                "config": {
                    "cache_enabled": self.config.bridge_config.cache_enabled,
                    "discovery_enabled": self.config.bridge_config.enable_discovery,
                    "max_concurrent_requests": self.config.bridge_config.max_concurrent_requests
                }
            },
            "servers": server_status,
            "metrics": self.metrics,
            "cache": cache_stats,
            "discovery": discovery_info
        }
    
    async def start_discovery(self):
        """Start the discovery service"""
        if self.config.bridge_config.enable_discovery:
            await self.discovery_service.start_discovery(
                self.config.bridge_config.discovery_interval
            )
            logger.info("Started MCP server discovery")
    
    async def _on_servers_discovered(self, discovered_servers: Dict[str, Any]):
        """Callback when new servers are discovered"""
        logger.info(f"Discovered {len(discovered_servers)} MCP servers")
        
        for server_name, server_info in discovered_servers.items():
            if server_name not in self.client_manager.connections:
                # Add new server
                server_config = server_info.to_server_config()
                await self.client_manager.add_server(server_config)
                logger.info(f"Added discovered server: {server_name}")
        
        self.metrics["last_discovery"] = datetime.utcnow().isoformat()
        self.metrics["servers_active"] = len(self.client_manager.connections)
    
    def _is_cacheable_tool(self, tool_name: str) -> bool:
        """Determine if a tool's results can be cached"""
        # Don't cache tools that modify state
        non_cacheable = {
            "write", "create", "update", "delete", "upsert",
            "send", "publish", "execute", "run"
        }
        
        return not any(keyword in tool_name.lower() for keyword in non_cacheable)
    
    def _generate_cache_key(self, 
                           server_name: str,
                           tool_name: str,
                           arguments: Optional[Dict[str, Any]]) -> str:
        """Generate cache key for a tool call"""
        # Create deterministic key
        key_parts = [server_name, tool_name]
        
        if arguments:
            # Sort arguments for consistency
            args_str = json.dumps(arguments, sort_keys=True)
            key_parts.append(args_str)
        
        key_string = ":".join(key_parts)
        
        # Hash for shorter key
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _update_latency_metric(self, latency_ms: float):
        """Update average latency metric"""
        # Simple moving average
        total_requests = self.metrics["requests_success"]
        if total_requests == 0:
            self.metrics["average_latency_ms"] = latency_ms
        else:
            current_avg = self.metrics["average_latency_ms"]
            self.metrics["average_latency_ms"] = (
                (current_avg * (total_requests - 1) + latency_ms) / total_requests
            )
    
    # HTTP Integration Methods
    
    async def handle_http_request(self,
                                 method: str,
                                 path: str,
                                 body: Optional[Dict[str, Any]] = None,
                                 params: Optional[Dict[str, str]] = None,
                                 headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Handle HTTP request and convert to MCP operations.
        
        Args:
            method: HTTP method
            path: Request path
            body: Request body
            params: Query parameters
            headers: Request headers
            
        Returns:
            HTTP response data
        """
        # Ensure initialized
        if not self._initialized:
            await self.initialize()
        
        # Route based on path
        path = path.strip("/")
        
        if path == "mcp/tools/list":
            server = params.get("server") if params else None
            return await self.list_tools(server)
        
        elif path == "mcp/tools/execute":
            if not body:
                return {
                    "success": False,
                    "error": "Request body required"
                }
            
            return await self.execute_tool(
                server_name=body.get("server", "default"),
                tool_name=body.get("tool"),
                arguments=body.get("arguments"),
                timeout=body.get("timeout")
            )
        
        elif path == "mcp/servers/list":
            status = await self.get_status()
            return {
                "success": True,
                "servers": list(status["servers"].keys())
            }
        
        elif path == "mcp/servers/status":
            return await self.get_status()
        
        elif path.startswith("mcp/servers/") and method == "DELETE":
            server_name = path.split("/")[-1]
            return await self.remove_server(server_name)
        
        else:
            return {
                "success": False,
                "error": f"Unknown endpoint: {path}"
            }