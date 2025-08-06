"""
MCP Client Manager
==================

Manages connections to multiple MCP servers without hardcoding.
Handles connection pooling, retries, and load balancing.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

import logging
import asyncio
from typing import Dict, Any, Optional, List, Set, TYPE_CHECKING
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json

# HTTP client with fallback
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None

# Type checking imports
if TYPE_CHECKING:
    from .mcp_config import MCPServerConfig
    from .mcp_protocol import MCPProtocolTranslator, MCPError

logger = logging.getLogger(__name__)

@dataclass
class MCPClientConnection:
    """Represents a connection to an MCP server"""
    server_config: "MCPServerConfig"
    session: Optional[aiohttp.ClientSession] = None
    connected: bool = False
    last_used: datetime = field(default_factory=datetime.utcnow)
    last_health_check: datetime = field(default_factory=datetime.utcnow)
    health_status: str = "unknown"  # healthy, unhealthy, unknown
    connection_id: str = field(default_factory=lambda: f"conn-{datetime.utcnow().timestamp()}")
    error_count: int = 0
    total_requests: int = 0
    total_errors: int = 0
    
    def update_stats(self, success: bool):
        """Update connection statistics"""
        self.total_requests += 1
        self.last_used = datetime.utcnow()
        
        if success:
            self.error_count = 0  # Reset error count on success
        else:
            self.error_count += 1
            self.total_errors += 1

class MCPClientManager:
    """
    Manages multiple MCP client connections.
    Provides connection pooling, health checks, and failover.
    """
    
    def __init__(self, protocol_translator: Optional["MCPProtocolTranslator"] = None):
        """Initialize the client manager"""
        self.connections: Dict[str, MCPClientConnection] = {}
        self.protocol = protocol_translator
        self._lock = asyncio.Lock()
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._connection_pools: Dict[str, List[MCPClientConnection]] = {}
        self._max_pool_size = 5  # Max connections per server
        self._connection_timeout = 30  # seconds
        self._read_timeout = 60  # seconds
    
    async def add_server(self, server_config: "MCPServerConfig") -> bool:
        """
        Add a new MCP server to manage.
        
        Args:
            server_config: Server configuration
            
        Returns:
            True if successfully added
        """
        if not AIOHTTP_AVAILABLE:
            logger.error("aiohttp not available - cannot add MCP servers")
            return False
            
        async with self._lock:
            try:
                # Create initial connection
                connection = MCPClientConnection(server_config=server_config)
                
                # Initialize connection pool for this server
                if server_config.name not in self._connection_pools:
                    self._connection_pools[server_config.name] = []
                
                # Add to connections
                self.connections[server_config.name] = connection
                
                # Start health check if enabled
                if server_config.health_check_endpoint:
                    await self._start_health_check(server_config.name)
                
                logger.info(f"Added MCP server: {server_config.name}")
                return True
                
            except Exception as e:
                logger.error(f"Error adding server {server_config.name}: {str(e)}")
                return False
    
    async def remove_server(self, server_name: str) -> bool:
        """Remove an MCP server and close all connections"""
        async with self._lock:
            if server_name not in self.connections:
                return False
            
            # Stop health check
            await self._stop_health_check(server_name)
            
            # Close all connections in pool
            if server_name in self._connection_pools:
                for conn in self._connection_pools[server_name]:
                    await self._close_connection(conn)
                del self._connection_pools[server_name]
            
            # Remove main connection
            main_conn = self.connections[server_name]
            await self._close_connection(main_conn)
            del self.connections[server_name]
            
            logger.info(f"Removed MCP server: {server_name}")
            return True
    
    async def execute_request(self, 
                            server_name: str,
                            mcp_request: Dict[str, Any],
                            timeout: Optional[int] = None) -> Dict[str, Any]:
        """
        Execute an MCP request on a specific server.
        
        Args:
            server_name: Name of the server
            mcp_request: MCP protocol request
            timeout: Request timeout in seconds
            
        Returns:
            MCP protocol response
        """
        if not AIOHTTP_AVAILABLE:
            return self._create_error_response("aiohttp not available", mcp_request.get("id"))
            
        if server_name not in self.connections:
            return self._create_error_response(
                f"Server '{server_name}' not found",
                mcp_request.get("id")
            )
        
        # Get connection from pool or create new
        connection = await self._get_connection(server_name)
        if not connection:
            return self._create_error_response(
                f"Failed to get connection to '{server_name}'",
                mcp_request.get("id")
            )
        
        # Execute request with retries
        retry_count = connection.server_config.retry_count
        retry_delay = connection.server_config.retry_delay / 1000  # Convert to seconds
        
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                response = await self._send_request(
                    connection,
                    mcp_request,
                    timeout or connection.server_config.timeout
                )
                
                # Update connection stats
                connection.update_stats(success=True)
                
                return response
                
            except asyncio.TimeoutError:
                last_error = "Request timeout"
                logger.warning(f"Timeout on attempt {attempt + 1} for {server_name}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Error on attempt {attempt + 1} for {server_name}: {str(e)}")
            
            # Update error stats
            connection.update_stats(success=False)
            
            # Wait before retry (except on last attempt)
            if attempt < retry_count:
                await asyncio.sleep(retry_delay)
        
        # All retries failed
        return self._create_error_response(
            f"Request failed after {retry_count + 1} attempts: {last_error}",
            mcp_request.get("id")
        )
    
    def _create_error_response(self, message: str, request_id: Optional[str] = None) -> Dict[str, Any]:
        """Create an error response in MCP format"""
        if self.protocol:
            return self.protocol.create_mcp_error(-32603, message, request_id=request_id)
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32603,
                    "message": message
                }
            }
    
    async def list_tools(self, server_name: Optional[str] = None) -> Dict[str, List[Dict[str, Any]]]:
        """
        List tools from one or all servers.
        
        Args:
            server_name: Specific server name or None for all
            
        Returns:
            Dictionary of server names to tool lists
        """
        servers = [server_name] if server_name else list(self.connections.keys())
        results = {}
        
        # Create list tools request
        list_request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {}
        }
        
        # Execute requests in parallel
        tasks = []
        for srv in servers:
            if srv in self.connections:
                task = self.execute_request(srv, {**list_request, "id": f"list-{srv}"})
                tasks.append((srv, task))
        
        # Gather results
        for srv, task in tasks:
            try:
                response = await task
                if "result" in response:
                    results[srv] = response["result"].get("tools", [])
                else:
                    results[srv] = []
                    logger.warning(f"Failed to list tools from {srv}")
            except Exception as e:
                logger.error(f"Error listing tools from {srv}: {str(e)}")
                results[srv] = []
        
        return results
    
    async def get_server_status(self, server_name: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
        """Get status information for one or all servers"""
        servers = [server_name] if server_name else list(self.connections.keys())
        status = {}
        
        for srv in servers:
            if srv in self.connections:
                conn = self.connections[srv]
                pool_size = len(self._connection_pools.get(srv, []))
                
                status[srv] = {
                    "connected": conn.connected,
                    "health_status": conn.health_status,
                    "last_used": conn.last_used.isoformat(),
                    "last_health_check": conn.last_health_check.isoformat(),
                    "error_count": conn.error_count,
                    "total_requests": conn.total_requests,
                    "total_errors": conn.total_errors,
                    "error_rate": conn.total_errors / max(conn.total_requests, 1),
                    "pool_size": pool_size,
                    "url": conn.server_config.url
                }
        
        return status
    
    async def _get_connection(self, server_name: str) -> Optional[MCPClientConnection]:
        """Get a connection from pool or create new one"""
        # Check if we have available connection in pool
        pool = self._connection_pools.get(server_name, [])
        
        # Find healthy connection
        for conn in pool:
            if conn.connected and conn.error_count < 3:
                return conn
        
        # Create new connection if pool not full
        if len(pool) < self._max_pool_size:
            main_conn = self.connections[server_name]
            new_conn = MCPClientConnection(server_config=main_conn.server_config)
            
            # Initialize session
            await self._ensure_session(new_conn)
            
            # Add to pool
            pool.append(new_conn)
            self._connection_pools[server_name] = pool
            
            return new_conn
        
        # Use main connection as fallback
        main_conn = self.connections[server_name]
        await self._ensure_session(main_conn)
        return main_conn
    
    async def _ensure_session(self, connection: MCPClientConnection):
        """Ensure connection has an active session"""
        if not AIOHTTP_AVAILABLE:
            return
            
        if not connection.session or connection.session.closed:
            # Create connector with timeout
            connector = aiohttp.TCPConnector(
                limit=10,
                ttl_dns_cache=300
            )
            
            # Create session with timeouts
            timeout = aiohttp.ClientTimeout(
                total=self._read_timeout,
                connect=self._connection_timeout,
                sock_connect=self._connection_timeout,
                sock_read=self._read_timeout
            )
            
            # Build headers
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json"
            }
            
            # Add authentication headers
            if connection.server_config.auth_type == "bearer" and connection.server_config.auth_credentials:
                token = connection.server_config.auth_credentials.get("token")
                if token:
                    headers["Authorization"] = f"Bearer {token}"
            elif connection.server_config.auth_type == "basic" and connection.server_config.auth_credentials:
                import base64
                username = connection.server_config.auth_credentials.get("username", "")
                password = connection.server_config.auth_credentials.get("password", "")
                credentials = base64.b64encode(f"{username}:{password}".encode()).decode()
                headers["Authorization"] = f"Basic {credentials}"
            
            connection.session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers=headers
            )
            
            connection.connected = True
    
    async def _close_connection(self, connection: MCPClientConnection):
        """Close a connection and cleanup"""
        if connection.session and not connection.session.closed:
            await connection.session.close()
        connection.connected = False
        connection.session = None
    
    async def _send_request(self, 
                          connection: MCPClientConnection,
                          request: Dict[str, Any],
                          timeout: int) -> Dict[str, Any]:
        """Send request to MCP server"""
        if connection.server_config.transport == "sse":
            return await self._send_sse_request(connection, request, timeout)
        else:
            return await self._send_http_request(connection, request, timeout)
    
    async def _send_http_request(self,
                               connection: MCPClientConnection,
                               request: Dict[str, Any],
                               timeout: int) -> Dict[str, Any]:
        """Send standard HTTP request"""
        url = connection.server_config.url
        
        async with connection.session.post(
            url,
            json=request,
            timeout=aiohttp.ClientTimeout(total=timeout/1000)  # Convert to seconds
        ) as response:
            response_text = await response.text()
            
            if response.status != 200:
                return self._create_error_response(
                    f"HTTP {response.status}: {response_text}",
                    request.get("id")
                )
            
            try:
                return json.loads(response_text)
            except json.JSONDecodeError:
                return self._create_error_response(
                    "Invalid JSON response",
                    request.get("id")
                )
    
    async def _send_sse_request(self,
                              connection: MCPClientConnection,
                              request: Dict[str, Any],
                              timeout: int) -> Dict[str, Any]:
        """Send Server-Sent Events request"""
        url = f"{connection.server_config.url}/sse"
        
        # For SSE, we need to handle streaming response
        async with connection.session.post(
            url,
            json=request,
            timeout=aiohttp.ClientTimeout(total=timeout/1000)
        ) as response:
            if response.status != 200:
                text = await response.text()
                return self._create_error_response(
                    f"HTTP {response.status}: {text}",
                    request.get("id")
                )
            
            # Read SSE stream
            result = None
            async for line in response.content:
                line = line.decode('utf-8').strip()
                if line.startswith('data: '):
                    data = line[6:]  # Remove 'data: ' prefix
                    try:
                        message = json.loads(data)
                        # Check if this is our response
                        if message.get("id") == request.get("id"):
                            result = message
                            break
                    except json.JSONDecodeError:
                        continue
            
            if result:
                return result
            else:
                return self._create_error_response(
                    "No response received",
                    request.get("id")
                )
    
    async def _start_health_check(self, server_name: str):
        """Start health check task for a server"""
        if server_name in self._health_check_tasks:
            return
        
        async def health_check_loop():
            connection = self.connections[server_name]
            while server_name in self.connections:
                try:
                    # Perform health check
                    health_url = connection.server_config.health_check_endpoint
                    if health_url:
                        if not health_url.startswith('http'):
                            health_url = f"{connection.server_config.url}/{health_url}"
                        
                        await self._ensure_session(connection)
                        if connection.session:
                            async with connection.session.get(
                                health_url,
                                timeout=aiohttp.ClientTimeout(total=5)
                            ) as response:
                                if response.status == 200:
                                    connection.health_status = "healthy"
                                else:
                                    connection.health_status = "unhealthy"
                    
                    connection.last_health_check = datetime.utcnow()
                    
                except Exception as e:
                    logger.warning(f"Health check failed for {server_name}: {str(e)}")
                    connection.health_status = "unhealthy"
                
                # Wait for next check
                await asyncio.sleep(30)  # Check every 30 seconds
        
        task = asyncio.create_task(health_check_loop())
        self._health_check_tasks[server_name] = task
    
    async def _stop_health_check(self, server_name: str):
        """Stop health check task for a server"""
        if server_name in self._health_check_tasks:
            self._health_check_tasks[server_name].cancel()
            try:
                await self._health_check_tasks[server_name]
            except asyncio.CancelledError:
                pass
            del self._health_check_tasks[server_name]
    
    async def close_all(self):
        """Close all connections and cleanup"""
        # Stop all health checks
        for server_name in list(self._health_check_tasks.keys()):
            await self._stop_health_check(server_name)
        
        # Close all connections
        for server_name in list(self.connections.keys()):
            await self.remove_server(server_name)