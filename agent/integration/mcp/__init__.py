"""
MCP Bridge Layer for Azure Functions
====================================

This module provides a bridge between HTTP requests in Azure Functions
and MCP (Model Context Protocol) servers.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

from .mcp_bridge import MCPBridge
from .mcp_client_manager import MCPClientManager
from .mcp_protocol import MCPProtocolTranslator
from .mcp_discovery import MCPDiscoveryService
from .mcp_config import MCPConfiguration

__all__ = [
    'MCPBridge',
    'MCPClientManager', 
    'MCPProtocolTranslator',
    'MCPDiscoveryService',
    'MCPConfiguration'
]