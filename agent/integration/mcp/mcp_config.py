"""
MCP Configuration Management
============================

Manages all configuration for MCP servers without hardcoding.
Supports environment variables, configuration files, and dynamic updates.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path

# YAML import with fallback
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    yaml = None

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server"""
    name: str
    url: str
    transport: str = "sse"  # sse, http, stdio
    auth_type: Optional[str] = None  # bearer, basic, oauth2
    auth_credentials: Optional[Dict[str, str]] = None
    timeout: int = 30000  # milliseconds
    retry_count: int = 3
    retry_delay: int = 1000  # milliseconds
    health_check_endpoint: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding sensitive data"""
        result = {
            "name": self.name,
            "url": self.url,
            "transport": self.transport,
            "auth_type": self.auth_type,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "retry_delay": self.retry_delay,
            "health_check_endpoint": self.health_check_endpoint,
            "metadata": self.metadata
        }
        return result

@dataclass
class MCPBridgeConfig:
    """Configuration for the MCP Bridge"""
    cache_enabled: bool = True
    cache_ttl: int = 300  # seconds
    max_concurrent_requests: int = 10
    request_timeout: int = 30000  # milliseconds
    enable_discovery: bool = True
    discovery_interval: int = 60  # seconds
    enable_health_checks: bool = True
    health_check_interval: int = 30  # seconds
    enable_metrics: bool = True
    metrics_endpoint: Optional[str] = None

class MCPConfiguration:
    """
    Central configuration manager for MCP Bridge.
    NO HARDCODING - everything is configurable.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration from multiple sources:
        1. Environment variables (highest priority)
        2. Configuration file (YAML/JSON)
        3. Default values (lowest priority)
        """
        self.servers: Dict[str, MCPServerConfig] = {}
        self.bridge_config = MCPBridgeConfig()
        self.config_path = config_path or os.getenv("MCP_CONFIG_PATH", "mcp_config.yaml")
        self._dynamic_config: Dict[str, Any] = {}
        
        # Load configuration in order of priority
        self._load_defaults()
        self._load_config_file()
        self._load_environment_variables()
        self._validate_configuration()
    
    def _load_defaults(self):
        """Load default configuration values"""
        # Bridge defaults are already in MCPBridgeConfig dataclass
        logger.info("Loaded default configuration")
    
    def _load_config_file(self):
        """Load configuration from YAML or JSON file"""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            logger.info(f"Configuration file {self.config_path} not found, using defaults")
            return
        
        try:
            with open(config_file, 'r') as f:
                if config_file.suffix in ['.yaml', '.yml'] and YAML_AVAILABLE:
                    config_data = yaml.safe_load(f)
                elif config_file.suffix == '.json':
                    config_data = json.load(f)
                else:
                    if not YAML_AVAILABLE and config_file.suffix in ['.yaml', '.yml']:
                        logger.warning(f"YAML support not available, skipping {config_file}")
                        return
                    logger.warning(f"Unknown config file format: {config_file.suffix}")
                    return
            
            # Load bridge configuration
            if 'bridge' in config_data:
                bridge_data = config_data['bridge']
                self.bridge_config = MCPBridgeConfig(
                    cache_enabled=bridge_data.get('cache_enabled', True),
                    cache_ttl=bridge_data.get('cache_ttl', 300),
                    max_concurrent_requests=bridge_data.get('max_concurrent_requests', 10),
                    request_timeout=bridge_data.get('request_timeout', 30000),
                    enable_discovery=bridge_data.get('enable_discovery', True),
                    discovery_interval=bridge_data.get('discovery_interval', 60),
                    enable_health_checks=bridge_data.get('enable_health_checks', True),
                    health_check_interval=bridge_data.get('health_check_interval', 30),
                    enable_metrics=bridge_data.get('enable_metrics', True),
                    metrics_endpoint=bridge_data.get('metrics_endpoint')
                )
            
            # Load server configurations
            if 'servers' in config_data:
                for server_data in config_data['servers']:
                    server_config = self._create_server_config(server_data)
                    self.servers[server_config.name] = server_config
            
            logger.info(f"Loaded configuration from {self.config_path}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {str(e)}")
    
    def _load_environment_variables(self):
        """
        Load configuration from environment variables.
        Environment variables override file configuration.
        
        Naming convention:
        - MCP_BRIDGE_<SETTING> for bridge settings
        - MCP_SERVER_<NAME>_<SETTING> for server settings
        """
        # Bridge configuration from environment
        env_mappings = {
            'MCP_BRIDGE_CACHE_ENABLED': ('cache_enabled', lambda x: x.lower() == 'true'),
            'MCP_BRIDGE_CACHE_TTL': ('cache_ttl', int),
            'MCP_BRIDGE_MAX_CONCURRENT': ('max_concurrent_requests', int),
            'MCP_BRIDGE_TIMEOUT': ('request_timeout', int),
            'MCP_BRIDGE_DISCOVERY_ENABLED': ('enable_discovery', lambda x: x.lower() == 'true'),
            'MCP_BRIDGE_DISCOVERY_INTERVAL': ('discovery_interval', int),
            'MCP_BRIDGE_HEALTH_CHECK_ENABLED': ('enable_health_checks', lambda x: x.lower() == 'true'),
            'MCP_BRIDGE_HEALTH_CHECK_INTERVAL': ('health_check_interval', int),
            'MCP_BRIDGE_METRICS_ENABLED': ('enable_metrics', lambda x: x.lower() == 'true'),
            'MCP_BRIDGE_METRICS_ENDPOINT': ('metrics_endpoint', str),
        }
        
        for env_var, (attr, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    setattr(self.bridge_config, attr, converter(value))
                    logger.info(f"Set bridge config {attr} from environment variable {env_var}")
                except Exception as e:
                    logger.error(f"Error converting {env_var}: {str(e)}")
        
        # Dynamic server configuration from environment
        # Format: MCP_SERVER_<NAME>_URL, MCP_SERVER_<NAME>_AUTH_TYPE, etc.
        server_env_vars = {}
        for key, value in os.environ.items():
            if key.startswith('MCP_SERVER_'):
                parts = key.split('_', 3)
                if len(parts) >= 4:
                    _, _, server_name, setting = parts[0], parts[1], parts[2], parts[3]
                    if server_name not in server_env_vars:
                        server_env_vars[server_name] = {}
                    server_env_vars[server_name][setting.lower()] = value
        
        # Create server configs from environment variables
        for server_name, settings in server_env_vars.items():
            if 'url' in settings:  # URL is required
                server_config = MCPServerConfig(
                    name=server_name.lower(),
                    url=settings['url'],
                    transport=settings.get('transport', 'sse'),
                    auth_type=settings.get('auth_type'),
                    auth_credentials=self._parse_auth_credentials(settings),
                    timeout=int(settings.get('timeout', 30000)),
                    retry_count=int(settings.get('retry_count', 3)),
                    retry_delay=int(settings.get('retry_delay', 1000)),
                    health_check_endpoint=settings.get('health_check_endpoint')
                )
                self.servers[server_config.name] = server_config
                logger.info(f"Configured server '{server_name}' from environment variables")
    
    def _parse_auth_credentials(self, settings: Dict[str, str]) -> Optional[Dict[str, str]]:
        """Parse authentication credentials from settings"""
        auth_type = settings.get('auth_type')
        if not auth_type:
            return None
        
        credentials = {}
        if auth_type == 'bearer':
            token = settings.get('auth_token') or settings.get('token')
            if token:
                credentials['token'] = token
        elif auth_type == 'basic':
            username = settings.get('auth_username') or settings.get('username')
            password = settings.get('auth_password') or settings.get('password')
            if username and password:
                credentials['username'] = username
                credentials['password'] = password
        elif auth_type == 'oauth2':
            credentials['client_id'] = settings.get('client_id', '')
            credentials['client_secret'] = settings.get('client_secret', '')
            credentials['token_url'] = settings.get('token_url', '')
        
        return credentials if credentials else None
    
    def _create_server_config(self, data: Dict[str, Any]) -> MCPServerConfig:
        """Create server configuration from dictionary"""
        return MCPServerConfig(
            name=data['name'],
            url=data['url'],
            transport=data.get('transport', 'sse'),
            auth_type=data.get('auth_type'),
            auth_credentials=data.get('auth_credentials'),
            timeout=data.get('timeout', 30000),
            retry_count=data.get('retry_count', 3),
            retry_delay=data.get('retry_delay', 1000),
            health_check_endpoint=data.get('health_check_endpoint'),
            metadata=data.get('metadata', {})
        )
    
    def _validate_configuration(self):
        """Validate the loaded configuration"""
        # Validate bridge config
        if self.bridge_config.cache_ttl < 0:
            logger.warning("Cache TTL is negative, disabling cache")
            self.bridge_config.cache_enabled = False
        
        if self.bridge_config.max_concurrent_requests < 1:
            logger.warning("Max concurrent requests < 1, setting to 1")
            self.bridge_config.max_concurrent_requests = 1
        
        # Validate server configs
        for name, server in list(self.servers.items()):
            if not server.url:
                logger.error(f"Server '{name}' has no URL, removing from configuration")
                del self.servers[name]
                continue
            
            if server.auth_type and not server.auth_credentials:
                logger.warning(f"Server '{name}' has auth_type but no credentials")
    
    def add_server(self, server_config: MCPServerConfig) -> bool:
        """Add or update a server configuration dynamically"""
        try:
            self.servers[server_config.name] = server_config
            logger.info(f"Added/updated server configuration: {server_config.name}")
            return True
        except Exception as e:
            logger.error(f"Error adding server configuration: {str(e)}")
            return False
    
    def remove_server(self, server_name: str) -> bool:
        """Remove a server configuration"""
        if server_name in self.servers:
            del self.servers[server_name]
            logger.info(f"Removed server configuration: {server_name}")
            return True
        return False
    
    def get_server(self, server_name: str) -> Optional[MCPServerConfig]:
        """Get a specific server configuration"""
        return self.servers.get(server_name)
    
    def list_servers(self) -> List[str]:
        """List all configured server names"""
        return list(self.servers.keys())
    
    def update_dynamic_config(self, key: str, value: Any):
        """Update dynamic configuration that can change at runtime"""
        self._dynamic_config[key] = value
        logger.info(f"Updated dynamic config: {key}")
    
    def get_dynamic_config(self, key: str, default: Any = None) -> Any:
        """Get dynamic configuration value"""
        return self._dynamic_config.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Export configuration as dictionary (excluding sensitive data)"""
        return {
            "bridge": {
                "cache_enabled": self.bridge_config.cache_enabled,
                "cache_ttl": self.bridge_config.cache_ttl,
                "max_concurrent_requests": self.bridge_config.max_concurrent_requests,
                "request_timeout": self.bridge_config.request_timeout,
                "enable_discovery": self.bridge_config.enable_discovery,
                "discovery_interval": self.bridge_config.discovery_interval,
                "enable_health_checks": self.bridge_config.enable_health_checks,
                "health_check_interval": self.bridge_config.health_check_interval,
                "enable_metrics": self.bridge_config.enable_metrics,
                "metrics_endpoint": self.bridge_config.metrics_endpoint
            },
            "servers": {
                name: server.to_dict() 
                for name, server in self.servers.items()
            },
            "dynamic_config": self._dynamic_config
        }
    
    def save_to_file(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        path = file_path or self.config_path
        try:
            config_data = self.to_dict()
            
            with open(path, 'w') as f:
                if path.endswith(('.yaml', '.yml')) and YAML_AVAILABLE:
                    yaml.safe_dump(config_data, f, default_flow_style=False)
                else:
                    json.dump(config_data, f, indent=2)
            
            logger.info(f"Saved configuration to {path}")
        except Exception as e:
            logger.error(f"Error saving configuration: {str(e)}")