"""
AutoGen Orchestrator for Bottler SPOKE
=====================================

This module implements AutoGen multi-agent orchestration for the bottler.
Uses MCP tools to access real Cosmos DB and Blob Storage data.

Author: TCCC Emerging Technology  
Version: 1.0.0
"""

import os
import logging
from typing import Dict, Any, List, Optional, Callable, TYPE_CHECKING
from datetime import datetime
import json
import asyncio

# Type checking imports
if TYPE_CHECKING:
    from integration.mcp import MCPBridge
    from integration.semantic_kernel_integration import BottlerSemanticKernelIntegration

# AutoGen imports
try:
    import autogen
    from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Create dummy classes for type hints
    class AssistantAgent: pass
    class UserProxyAgent: pass
    class GroupChat: pass
    class GroupChatManager: pass

logger = logging.getLogger(__name__)


class BottlerAutoGenOrchestrator:
    """
    AutoGen orchestrator for Bottler SPOKE.
    Coordinates specialized agents to process hub queries using real data.
    """
    
    def __init__(self, 
                 mcp_bridge: Optional["MCPBridge"] = None, 
                 semantic_kernel_integration: Optional["BottlerSemanticKernelIntegration"] = None):
        """Initialize AutoGen orchestrator for bottler"""
        self.mcp_bridge = mcp_bridge
        self.sk_integration = semantic_kernel_integration
        self.bottler_id = os.getenv("BOTTLER_ID", "unknown")
        self.bottler_name = os.getenv("BOTTLER_NAME", "Unknown Bottler")
        
        # Check if AutoGen is available
        if not AUTOGEN_AVAILABLE:
            logger.error("AutoGen not available - multi-agent processing will be limited")
            return
            
        # Configuration
        self.config_list = self._get_config_list()
        
        # Initialize agents if AutoGen is available
        if AUTOGEN_AVAILABLE:
            self._initialize_agents()
        
        # Note: Call initialize() after creation for async initialization
        
    def _get_config_list(self) -> List[Dict[str, Any]]:
        """Get LLM configuration for agents using Azure AI Foundry"""
        # Get Azure AI Foundry configuration from environment variables
        api_key = os.getenv("AZURE_AI_FOUNDRY_KEY")
        endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
        deployment = os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", "tccc-model-router")
        api_version = os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-12-01-preview")
        
        if not api_key or not endpoint:
            raise ValueError("Azure AI Foundry configuration missing: AZURE_AI_FOUNDRY_KEY and AZURE_AI_FOUNDRY_ENDPOINT are required")
        
        # Always use Azure AI Foundry configuration (never standard OpenAI)
        return [{
            "model": deployment,
            "api_type": "azure",
            "base_url": endpoint,
            "api_version": api_version,
            "api_key": api_key
        }]
            
    def _initialize_agents(self):
        """Initialize specialized agents for the bottler"""
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available - agents not initialized")
            return
            
        # 1. Data Analyst Agent - Queries Cosmos DB
        self.data_analyst = AssistantAgent(
            name=f"DataAnalyst_{self.bottler_id}",
            system_message=f"""You are a data analyst for {self.bottler_name} bottler.
Your role is to:
1. Query financial and sales data from Cosmos DB
2. Analyze data patterns and trends
3. Calculate metrics like revenue, costs, margins
4. Aggregate data by product, period, or region

Bottler ID: {self.bottler_id}
Region: {os.getenv('BOTTLER_REGION', 'Unknown')}

When querying data, always filter by bottler_id = '{self.bottler_id}'.
Use the query_cosmos_db function to access real data.""",
            llm_config={"config_list": self.config_list}
        )
        
        # 2. Financial Expert Agent - Interprets financial data
        self.financial_expert = AssistantAgent(
            name=f"FinancialExpert_{self.bottler_id}",
            system_message=f"""You are a financial expert for {self.bottler_name} bottler.
Your role is to:
1. Interpret financial metrics and KPIs
2. Calculate profitability and margins
3. Identify trends and anomalies
4. Provide insights on financial performance

Focus on bottler-specific metrics and comparisons.
Always consider the local market context.""",
            llm_config={"config_list": self.config_list}
        )
        
        # 3. Product Specialist Agent - Analyzes product performance
        self.product_specialist = AssistantAgent(
            name=f"ProductSpecialist_{self.bottler_id}",
            system_message=f"""You are a product specialist for {self.bottler_name} bottler.
Your role is to:
1. Analyze product-specific performance (Coca-Cola, Sprite, Fanta, etc.)
2. Track sales volumes and market share
3. Identify best and worst performing products
4. Provide recommendations for product mix

Focus on products distributed in your region.""",
            llm_config={"config_list": self.config_list}
        )
        
        # 4. Report Generator Agent - Formats responses for hub
        self.report_generator = AssistantAgent(
            name=f"ReportGenerator_{self.bottler_id}",
            system_message=f"""You are a report generator for {self.bottler_name} bottler.
Your role is to:
1. Consolidate findings from other agents
2. Format professional responses for TCCC Hub
3. Include key metrics, insights, and recommendations
4. Ensure data accuracy and clarity

Format all responses as structured JSON when possible.""",
            llm_config={"config_list": self.config_list}
        )
        
        # 5. User Proxy - Executes MCP tool calls
        self.user_proxy = UserProxyAgent(
            name=f"BottlerProxy_{self.bottler_id}",
            system_message="Execute tool calls and return results.",
            human_input_mode="NEVER",
            max_consecutive_auto_reply=10,
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
            code_execution_config={
                "work_dir": f"bottler_{self.bottler_id}_workspace",
                "use_docker": False
            }
        )
        
    async def initialize(self):
        """Initialize async components and register MCP tools"""
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available - initialization skipped")
            return
            
        if self.mcp_bridge:
            try:
                # Register MCP tools with agents
                await self._register_mcp_tools()
                
                # Discover available tools
                tools = await self.mcp_bridge.list_tools()
                logger.info(f"Registered {len(tools.get('tools', {}))} MCP tools with AutoGen for bottler {self.bottler_id}")
                
            except Exception as e:
                logger.error(f"Failed to initialize AutoGen MCP tools: {str(e)}")
                
    async def _register_mcp_tools(self):
        """Register MCP tools as functions for agents"""
        if not AUTOGEN_AVAILABLE:
            return
            
        # Define MCP tool wrapper functions
        async def query_cosmos_db(container: str, query: str, max_items: int = 100) -> Dict[str, Any]:
            """Query Cosmos DB using MCP"""
            if not self.mcp_bridge:
                return {"error": "MCP Bridge not available"}
                
            result = await self.mcp_bridge.execute_tool(
                server_name="cosmos",
                tool_name="query_documents",
                arguments={
                    "container": container,
                    "query": query,
                    "max_items": max_items
                }
            )
            
            return result
            
        async def read_blob_storage(blob_name: str) -> Dict[str, Any]:
            """Read from Blob Storage using MCP"""
            if not self.mcp_bridge:
                return {"error": "MCP Bridge not available"}
                
            result = await self.mcp_bridge.execute_tool(
                server_name="blob",
                tool_name="read_blob",
                arguments={
                    "blob_name": blob_name
                }
            )
            
            return result
            
        async def write_blob_storage(blob_name: str, content: str, metadata: Dict[str, str] = None) -> Dict[str, Any]:
            """Write to Blob Storage using MCP"""
            if not self.mcp_bridge:
                return {"error": "MCP Bridge not available"}
                
            result = await self.mcp_bridge.execute_tool(
                server_name="blob",
                tool_name="write_blob",
                arguments={
                    "blob_name": blob_name,
                    "content": content,
                    "content_type": "application/json",
                    "metadata": metadata or {}
                }
            )
            
            return result
            
        # Register functions with user proxy
        self.user_proxy.register_function(
            function_map={
                "query_cosmos_db": query_cosmos_db,
                "read_blob_storage": read_blob_storage,
                "write_blob_storage": write_blob_storage
            }
        )
        
        # Also register with data analyst for direct access
        self.data_analyst.register_function(
            function_map={
                "query_cosmos_db": query_cosmos_db
            }
        )
        
    async def process_hub_query(self, query: str, sk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a query from the hub using AutoGen agents
        
        Args:
            query: Original query from hub
            sk_analysis: Analysis from Semantic Kernel
            
        Returns:
            Processed response from agent collaboration
        """
        try:
            if not AUTOGEN_AVAILABLE:
                return {
                    "success": False,
                    "error": "AutoGen not available",
                    "fallback_response": self._fallback_processing(query, sk_analysis)
                }
                
            logger.info(f"AutoGen processing hub query for bottler {self.bottler_id}: {query}")
            
            # Create group chat with all agents
            group_chat = GroupChat(
                agents=[
                    self.data_analyst,
                    self.financial_expert,
                    self.product_specialist,
                    self.report_generator,
                    self.user_proxy
                ],
                messages=[],
                max_round=10
            )
            
            # Create group chat manager
            manager = GroupChatManager(
                groupchat=group_chat,
                llm_config={"config_list": self.config_list}
            )
            
            # Prepare initial message with SK analysis
            initial_message = f"""Process this query from TCCC Hub:

Original Query: {query}

Semantic Kernel Analysis:
{json.dumps(sk_analysis, indent=2)}

Instructions:
1. Data Analyst: Query Cosmos DB for relevant financial data
2. Financial Expert: Analyze the financial metrics
3. Product Specialist: Provide product-specific insights
4. Report Generator: Create a comprehensive response for the hub

Remember to filter all data by bottler_id = '{self.bottler_id}'"""
            
            # Start the conversation
            await self.user_proxy.a_initiate_chat(
                manager,
                message=initial_message
            )
            
            # Extract the final response
            messages = group_chat.messages
            
            # Find the last message from report generator
            final_response = None
            for msg in reversed(messages):
                if msg.get("name") == f"ReportGenerator_{self.bottler_id}":
                    final_response = msg.get("content", "")
                    break
                    
            if not final_response:
                # Fallback to last message
                final_response = messages[-1].get("content", "") if messages else ""
                
            # Parse response if it's JSON
            try:
                if final_response.startswith("{") or final_response.startswith("["):
                    response_data = json.loads(final_response)
                else:
                    response_data = {"response": final_response}
            except json.JSONDecodeError:
                response_data = {"response": final_response}
                
            # Add metadata
            response_data["bottler_id"] = self.bottler_id
            response_data["bottler_name"] = self.bottler_name
            response_data["processing_method"] = "autogen_multiagent"
            response_data["agent_count"] = len(group_chat.agents)
            response_data["rounds_executed"] = len(messages)
            response_data["timestamp"] = datetime.utcnow().isoformat()
            
            return {
                "success": True,
                "response": response_data,
                "messages": [
                    {"role": msg.get("name", "unknown"), "content": msg.get("content", "")}
                    for msg in messages[-5:]  # Last 5 messages for context
                ]
            }
            
        except Exception as e:
            logger.error(f"Error in AutoGen processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "bottler_id": self.bottler_id,
                "fallback_response": self._fallback_processing(query, sk_analysis)
            }
            
    def _fallback_processing(self, query: str, sk_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback processing when AutoGen is not available"""
        return {
            "bottler_id": self.bottler_id,
            "bottler_name": self.bottler_name,
            "query": query,
            "analysis": sk_analysis,
            "message": "AutoGen not available - basic response generated",
            "processing_method": "fallback",
            "timestamp": datetime.utcnow().isoformat()
        }
            
    async def generate_financial_report(self, period: str = None) -> Dict[str, Any]:
        """
        Generate a comprehensive financial report using agents
        
        Args:
            period: Time period for the report
            
        Returns:
            Financial report data
        """
        try:
            if not AUTOGEN_AVAILABLE:
                return {
                    "success": False,
                    "error": "AutoGen not available for report generation"
                }
                
            # Create specialized group chat for reporting
            report_chat = GroupChat(
                agents=[
                    self.data_analyst,
                    self.financial_expert,
                    self.report_generator,
                    self.user_proxy
                ],
                messages=[],
                max_round=6
            )
            
            manager = GroupChatManager(
                groupchat=report_chat,
                llm_config={"config_list": self.config_list}
            )
            
            period_filter = period or datetime.utcnow().strftime("%Y-%m")
            
            initial_message = f"""Generate a financial report for {self.bottler_name} for period: {period_filter}

Data Analyst: Query financial_data container for:
1. Total revenue and costs
2. Product-wise breakdown
3. Monthly trends

Financial Expert: Analyze:
1. Profit margins
2. Cost efficiency
3. Revenue growth

Report Generator: Create a structured financial report with all findings."""
            
            await self.user_proxy.a_initiate_chat(
                manager,
                message=initial_message
            )
            
            # Extract report
            messages = report_chat.messages
            report = messages[-1].get("content", "") if messages else ""
            
            return {
                "success": True,
                "report": report,
                "period": period_filter,
                "generated_at": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating financial report: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
