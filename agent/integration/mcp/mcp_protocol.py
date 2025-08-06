"""
MCP Protocol Translator
=======================

Translates between HTTP requests and MCP protocol messages.
Handles all protocol-specific transformations without hardcoding.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

import json
import logging
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import uuid
from enum import Enum

logger = logging.getLogger(__name__)

class MCPMessageType(Enum):
    """MCP protocol message types"""
    # Requests
    INITIALIZE = "initialize"
    LIST_TOOLS = "tools/list"
    CALL_TOOL = "tools/call"
    LIST_RESOURCES = "resources/list"
    READ_RESOURCE = "resources/read"
    LIST_PROMPTS = "prompts/list"
    GET_PROMPT = "prompts/get"
    
    # Responses
    RESULT = "result"
    ERROR = "error"
    
    # Notifications
    PROGRESS = "progress"
    LOG = "log"
    TOOL_LIST_CHANGED = "tools/list/changed"

class MCPError:
    """MCP protocol error codes"""
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # Custom errors
    TOOL_NOT_FOUND = -32001
    RESOURCE_NOT_FOUND = -32002
    AUTHENTICATION_FAILED = -32003
    TIMEOUT = -32004

class MCPProtocolTranslator:
    """
    Translates between HTTP/REST requests and MCP protocol messages.
    Supports JSON-RPC 2.0 format used by MCP.
    """
    
    def __init__(self):
        """Initialize the protocol translator"""
        self.version = "2.0"  # JSON-RPC version
        self.protocol_version = "2024-11-05"  # MCP protocol version
        self._request_counter = 0
    
    def http_to_mcp_request(self, 
                           http_method: str,
                           http_path: str,
                           http_body: Optional[Dict[str, Any]] = None,
                           http_params: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """
        Convert HTTP request to MCP protocol request.
        
        Args:
            http_method: HTTP method (GET, POST, etc.)
            http_path: HTTP path (e.g., /tools/list)
            http_body: HTTP request body
            http_params: HTTP query parameters
            
        Returns:
            MCP protocol message
        """
        # Generate unique request ID
        request_id = self._generate_request_id()
        
        # Map HTTP path to MCP method
        mcp_method = self._map_http_to_mcp_method(http_path)
        
        # Build MCP request
        mcp_request = {
            "jsonrpc": self.version,
            "id": request_id,
            "method": mcp_method
        }
        
        # Add parameters based on method
        params = self._build_mcp_params(mcp_method, http_body, http_params)
        if params:
            mcp_request["params"] = params
        
        # Add metadata
        mcp_request["_meta"] = {
            "http_method": http_method,
            "http_path": http_path,
            "timestamp": datetime.utcnow().isoformat(),
            "protocol_version": self.protocol_version
        }
        
        logger.debug(f"Translated HTTP to MCP: {mcp_request}")
        return mcp_request
    
    def mcp_to_http_response(self, 
                            mcp_response: Dict[str, Any],
                            original_request: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Convert MCP protocol response to HTTP response format.
        
        Args:
            mcp_response: MCP protocol response
            original_request: Original MCP request for context
            
        Returns:
            HTTP response with status code and body
        """
        http_response = {
            "status_code": 200,
            "headers": {
                "Content-Type": "application/json",
                "X-MCP-Version": self.protocol_version
            },
            "body": {}
        }
        
        # Check if it's an error response
        if "error" in mcp_response:
            error = mcp_response["error"]
            http_response["status_code"] = self._map_mcp_error_to_http_status(error.get("code", -32603))
            http_response["body"] = {
                "error": True,
                "code": error.get("code"),
                "message": error.get("message", "Unknown error"),
                "data": error.get("data")
            }
        
        # Success response
        elif "result" in mcp_response:
            result = mcp_response["result"]
            
            # Map based on original request method if available
            if original_request and "_meta" in original_request:
                http_path = original_request["_meta"].get("http_path", "")
                http_response["body"] = self._format_result_for_http(result, http_path)
            else:
                http_response["body"] = {
                    "success": True,
                    "data": result
                }
        
        # Add response metadata
        http_response["body"]["_meta"] = {
            "mcp_id": mcp_response.get("id"),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return http_response
    
    def create_mcp_error(self, 
                        code: int,
                        message: str,
                        data: Optional[Any] = None,
                        request_id: Optional[Union[str, int]] = None) -> Dict[str, Any]:
        """Create a properly formatted MCP error response"""
        error_response = {
            "jsonrpc": self.version,
            "error": {
                "code": code,
                "message": message
            }
        }
        
        if data is not None:
            error_response["error"]["data"] = data
        
        if request_id is not None:
            error_response["id"] = request_id
        else:
            error_response["id"] = None
        
        return error_response
    
    def create_mcp_result(self,
                         result: Any,
                         request_id: Union[str, int]) -> Dict[str, Any]:
        """Create a properly formatted MCP result response"""
        return {
            "jsonrpc": self.version,
            "id": request_id,
            "result": result
        }
    
    def parse_mcp_message(self, message: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse and validate an MCP message.
        
        Args:
            message: Raw MCP message (string or dict)
            
        Returns:
            Parsed and validated message
            
        Raises:
            ValueError: If message is invalid
        """
        # Parse JSON if string
        if isinstance(message, str):
            try:
                message = json.loads(message)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON: {str(e)}")
        
        # Validate required fields
        if not isinstance(message, dict):
            raise ValueError("Message must be a JSON object")
        
        if message.get("jsonrpc") != self.version:
            raise ValueError(f"Invalid JSON-RPC version: {message.get('jsonrpc')}")
        
        # Request validation
        if "method" in message:
            if not isinstance(message.get("method"), str):
                raise ValueError("Method must be a string")
            
            if "params" in message and not isinstance(message["params"], (dict, list)):
                raise ValueError("Params must be an object or array")
        
        # Response validation
        elif "result" in message or "error" in message:
            if "id" not in message:
                raise ValueError("Response must have an id")
            
            if "result" in message and "error" in message:
                raise ValueError("Response cannot have both result and error")
        
        else:
            raise ValueError("Message must be a request or response")
        
        return message
    
    def _generate_request_id(self) -> str:
        """Generate unique request ID"""
        self._request_counter += 1
        return f"mcp-{self._request_counter}-{uuid.uuid4().hex[:8]}"
    
    def _map_http_to_mcp_method(self, http_path: str) -> str:
        """Map HTTP path to MCP method name"""
        # Remove leading slash and API prefix
        path = http_path.strip("/")
        if path.startswith("mcp/"):
            path = path[4:]
        
        # Direct mappings (no hardcoding, uses path structure)
        path_parts = path.split("/")
        
        # Build MCP method from path parts
        if len(path_parts) >= 2:
            # e.g., tools/list, resources/read
            return "/".join(path_parts[:2])
        elif len(path_parts) == 1:
            # Single word paths
            return path_parts[0]
        else:
            return "unknown"
    
    def _build_mcp_params(self, 
                         method: str,
                         body: Optional[Dict[str, Any]],
                         params: Optional[Dict[str, str]]) -> Optional[Dict[str, Any]]:
        """Build MCP parameters from HTTP request data"""
        mcp_params = {}
        
        # Merge query params and body
        if params:
            mcp_params.update(params)
        
        if body:
            mcp_params.update(body)
        
        # Method-specific parameter handling (based on MCP spec)
        if method == "tools/call":
            # Ensure required fields for tool calls
            if "name" not in mcp_params:
                raise ValueError("Tool name is required")
            
            # Rename 'parameters' to 'arguments' if present (MCP convention)
            if "parameters" in mcp_params and "arguments" not in mcp_params:
                mcp_params["arguments"] = mcp_params.pop("parameters")
        
        elif method == "resources/read":
            # Ensure URI is present
            if "uri" not in mcp_params:
                raise ValueError("Resource URI is required")
        
        elif method == "prompts/get":
            # Ensure name is present
            if "name" not in mcp_params:
                raise ValueError("Prompt name is required")
        
        return mcp_params if mcp_params else None
    
    def _map_mcp_error_to_http_status(self, error_code: int) -> int:
        """Map MCP error codes to HTTP status codes"""
        error_map = {
            MCPError.PARSE_ERROR: 400,
            MCPError.INVALID_REQUEST: 400,
            MCPError.METHOD_NOT_FOUND: 404,
            MCPError.INVALID_PARAMS: 400,
            MCPError.INTERNAL_ERROR: 500,
            MCPError.TOOL_NOT_FOUND: 404,
            MCPError.RESOURCE_NOT_FOUND: 404,
            MCPError.AUTHENTICATION_FAILED: 401,
            MCPError.TIMEOUT: 504
        }
        
        return error_map.get(error_code, 500)
    
    def _format_result_for_http(self, result: Any, http_path: str) -> Dict[str, Any]:
        """Format MCP result for HTTP response based on endpoint"""
        # Remove API prefix
        path = http_path.strip("/")
        if path.startswith("mcp/"):
            path = path[4:]
        
        # Format based on endpoint type
        if path == "tools/list":
            return {
                "success": True,
                "tools": result.get("tools", []),
                "count": len(result.get("tools", []))
            }
        
        elif path == "tools/execute" or path == "tools/call":
            return {
                "success": True,
                "result": result.get("content", []) if isinstance(result, dict) else result,
                "tool": result.get("tool") if isinstance(result, dict) else None
            }
        
        elif path == "resources/list":
            return {
                "success": True,
                "resources": result.get("resources", []),
                "count": len(result.get("resources", []))
            }
        
        elif path == "prompts/list":
            return {
                "success": True,
                "prompts": result.get("prompts", []),
                "count": len(result.get("prompts", []))
            }
        
        else:
            # Default format
            return {
                "success": True,
                "data": result
            }
    
    def create_batch_request(self, requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create a batch of MCP requests"""
        batch = []
        for req in requests:
            if "id" not in req:
                req["id"] = self._generate_request_id()
            
            if "jsonrpc" not in req:
                req["jsonrpc"] = self.version
            
            batch.append(req)
        
        return batch
    
    def parse_batch_response(self, responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Parse a batch of MCP responses"""
        results = {}
        errors = {}
        
        for response in responses:
            response_id = response.get("id")
            if not response_id:
                continue
            
            if "error" in response:
                errors[response_id] = response["error"]
            elif "result" in response:
                results[response_id] = response["result"]
        
        return {
            "results": results,
            "errors": errors,
            "total": len(responses),
            "successful": len(results),
            "failed": len(errors)
        }