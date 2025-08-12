"""
AutoGen Orchestrator for Bottler SPOKE
=====================================

This module implements AutoGen multi-agent orchestration for the bottler.
Uses direct Cosmos DB access for financial data retrieval.
Includes collaborative reasoning system for complex task processing.

Author: Cesar Vanegas Castro (cvanegas@coca-cola.com)  
Version: 1.2.0
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

# Import collaborative reasoning
try:
    from .collaborative_reasoning import CollaborativeReasoningSystem
    COLLABORATIVE_REASONING_AVAILABLE = True
except ImportError:
    COLLABORATIVE_REASONING_AVAILABLE = False
    CollaborativeReasoningSystem = None

# AutoGen imports
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
    
    # Create dummy classes for missing imports
    class GroupChat: pass
    class GroupChatManager: pass
except ImportError:
    AUTOGEN_AVAILABLE = False
    # Create dummy classes for type hints
    class AssistantAgent: pass
    class UserProxyAgent: pass
    class TextMessage: pass
    class CancellationToken: pass
    class GroupChat: pass
    class GroupChatManager: pass
    AzureOpenAIChatCompletionClient = None

logger = logging.getLogger(__name__)


class BottlerAutoGenOrchestrator:
    """
    AutoGen orchestrator for Bottler SPOKE.
    Coordinates specialized agents to process hub queries using real data.
    """
    
    def __init__(self, 
                 semantic_kernel_integration: Optional["BottlerSemanticKernelIntegration"] = None):
        """Initialize AutoGen orchestrator for bottler"""
        # Direct database access instead of MCP bridge
        self.sk_integration = semantic_kernel_integration
        self.bottler_id = os.getenv("BOTTLER_ID", "unknown")
        self.bottler_name = os.getenv("BOTTLER_NAME", "Unknown Bottler")
        self.cosmos_container = None  # Will be initialized in _register_direct_tools
        self.direct_tools = {}  # Will store tool functions
        
        # Check if AutoGen is available
        if not AUTOGEN_AVAILABLE:
            logger.error("AutoGen not available - multi-agent processing will be limited")
            self.collaborative_reasoning = None
            self.model_client = None
            return
            
        # Configuration
        try:
            self.config_list = self._get_config_list()
        except Exception as e:
            logger.error(f"Failed to get config list: {e}")
            self.config_list = []
            self.collaborative_reasoning = None
            self.model_client = None
            return
        
        # Create model client for new AutoGen pattern
        self.model_client = None
        if AUTOGEN_AVAILABLE and AzureOpenAIChatCompletionClient:
            self._create_model_client()
        
        # Initialize agents if AutoGen is available
        if AUTOGEN_AVAILABLE and self.model_client:
            self._initialize_agents()
            
        # Initialize collaborative reasoning system
        self.collaborative_reasoning = None
        
        # Debug: Check prerequisites
        logger.info(f"COLLABORATIVE_REASONING_AVAILABLE: {COLLABORATIVE_REASONING_AVAILABLE}")
        logger.info(f"config_list available: {bool(self.config_list)}")
        logger.info(f"config_list length: {len(self.config_list) if self.config_list else 0}")
        
        # Force initialize collaborative reasoning even if some dependencies are missing
        try:
            logger.info("Force initializing collaborative reasoning system...")
            
            # Create basic config if missing
            if not self.config_list:
                logger.info("Creating basic config list for collaborative reasoning")
                self.config_list = self._get_config_list()
                
            self.collaborative_reasoning = CollaborativeReasoningSystem(
                bottler_id=self.bottler_id,
                sk_integration=self.sk_integration,
                config_list=self.config_list
            )
            logger.info("Successfully force-initialized collaborative reasoning system")
        except Exception as e:
            logger.error(f"Failed to force-initialize collaborative reasoning: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            self.collaborative_reasoning = None
            
            # Try a more basic initialization
            try:
                logger.info("Trying basic collaborative reasoning initialization...")
                # Create a minimal config list
                basic_config = [{
                    "model": "tccc-model-router",
                    "api_type": "azure",
                    "base_url": os.getenv("AI_FOUNDRY_ENDPOINT", ""),
                    "api_version": "2024-12-01-preview", 
                    "api_key": os.getenv("AI_FOUNDRY_API_KEY", "")
                }]
                
                self.collaborative_reasoning = CollaborativeReasoningSystem(
                    bottler_id=self.bottler_id,
                    sk_integration=None,  # Skip SK integration
                    config_list=basic_config
                )
                logger.info("Successfully initialized collaborative reasoning with basic config")
            except Exception as e2:
                logger.error(f"Even basic collaborative reasoning failed: {e2}")
                self.collaborative_reasoning = None
        
        # Note: Call initialize() after creation for async initialization
        
    def _get_config_list(self) -> List[Dict[str, Any]]:
        """Get LLM configuration for agents using Azure AI Foundry"""
        # Get Azure AI Foundry configuration from environment variables
        # Try both AZURE_ prefix and without prefix for compatibility
        api_key = os.getenv("AZURE_AI_FOUNDRY_KEY") or os.getenv("AI_FOUNDRY_KEY") or os.getenv("AI_FOUNDRY_API_KEY")
        endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT") or os.getenv("AI_FOUNDRY_ENDPOINT")
        deployment = os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", os.getenv("AI_FOUNDRY_DEPLOYMENT", "tccc-model-router"))
        api_version = os.getenv("AZURE_AI_FOUNDRY_API_VERSION", os.getenv("AI_FOUNDRY_API_VERSION", "2024-12-01-preview"))
        
        if not api_key or not endpoint:
            # Log what we tried to find
            logger.error(f"API Key found: {bool(api_key)}, Endpoint found: {bool(endpoint)}")
            logger.error(f"Checked env vars: AZURE_AI_FOUNDRY_KEY, AI_FOUNDRY_KEY, AI_FOUNDRY_API_KEY")
            raise ValueError("Azure AI Foundry configuration missing: API key and endpoint are required")
        
        # Always use Azure AI Foundry configuration (never standard OpenAI)
        return [{
            "model": deployment,
            "api_type": "azure",
            "base_url": endpoint,
            "api_version": api_version,
            "api_key": api_key
        }]
    
    def _create_model_client(self):
        """Create Azure OpenAI model client for new AutoGen pattern"""
        if not self.config_list or not self.config_list[0]:
            logger.error("No configuration available for model client")
            return
            
        config = self.config_list[0]
        api_key = config.get("api_key")
        endpoint = config.get("base_url")
        deployment = config.get("model", "tccc-model-router")
        api_version = config.get("api_version", "2024-12-01-preview")
        
        if not api_key or not endpoint:
            logger.error("Missing API key or endpoint for model client")
            return
            
        try:
            # Create model_info for custom Azure deployment
            model_info = {
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "structured_output": True,  # AÑADIDO: campo requerido
                "family": "gpt-4.1"  # CAMBIADO: de gpt-4 a gpt-4.1
            }
            
            # Create AzureOpenAIChatCompletionClient with proper configuration
            self.model_client = AzureOpenAIChatCompletionClient(
                model=deployment,
                api_key=api_key,
                azure_endpoint=endpoint,
                api_version=api_version,
                model_info=model_info
            )
            logger.info("Created AzureOpenAIChatCompletionClient successfully for autogen_orchestrator")
        except Exception as e:
            logger.error(f"Failed to create model client: {e}")
            self.model_client = None
            
    def _initialize_agents(self):
        """Initialize specialized agents for the bottler"""
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available - agents not initialized")
            return
            
        if not self.model_client:
            logger.warning("Model client not available - agents not initialized")
            return
            
        # Clean bottler_id to make it a valid Python identifier (replace - with _)
        clean_bottler_id = self.bottler_id.replace("-", "_").replace(" ", "_")
            
        # 1. Data Analyst Agent - Queries Cosmos DB
        self.data_analyst = AssistantAgent(
            name=f"DataAnalyst_{clean_bottler_id}",
            model_client=self.model_client,
            system_message=f"""You are a data analyst for {self.bottler_name} bottler.
Your role is to:
1. Query financial and sales data from Cosmos DB
2. Analyze data patterns and trends
3. Calculate metrics like revenue, costs, margins
4. Aggregate data by product, period, or region

Bottler ID: {self.bottler_id}
Region: {os.getenv('BOTTLER_REGION', 'Unknown')}

When querying data, always filter by bottler_id = '{self.bottler_id}'.
Use the query_cosmos_db function to access real data."""
        )
        
        # 2. Financial Expert Agent - Interprets financial data
        self.financial_expert = AssistantAgent(
            name=f"FinancialExpert_{clean_bottler_id}",
            model_client=self.model_client,
            system_message=f"""You are a financial expert for {self.bottler_name} bottler.
Your role is to:
1. Interpret financial metrics and KPIs
2. Calculate profitability and margins
3. Identify trends and anomalies
4. Provide insights on financial performance

Focus on bottler-specific metrics and comparisons.
Always consider the local market context."""
        )
        
        # 3. Product Specialist Agent - Analyzes product performance
        self.product_specialist = AssistantAgent(
            name=f"ProductSpecialist_{clean_bottler_id}",
            model_client=self.model_client,
            system_message=f"""You are a product specialist for {self.bottler_name} bottler.
Your role is to:
1. Analyze product-specific performance (Coca-Cola, Sprite, Fanta, etc.)
2. Track sales volumes and market share
3. Identify best and worst performing products
4. Provide recommendations for product mix

Focus on products distributed in your region."""
        )
        
        # 4. Report Generator Agent - Formats responses for hub
        self.report_generator = AssistantAgent(
            name=f"ReportGenerator_{clean_bottler_id}",
            model_client=self.model_client,
            system_message=f"""You are a report generator for {self.bottler_name} bottler.
Your role is to:
1. Consolidate findings from other agents
2. Format professional responses for TCCC Hub
3. Include key metrics, insights, and recommendations
4. Ensure data accuracy and clarity

Format all responses as structured JSON when possible."""
        )
        
        # 5. User Proxy - Executes MCP tool calls
        # UserProxyAgent in new version has minimal parameters
        self.user_proxy = UserProxyAgent(
            name=f"BottlerProxy_{clean_bottler_id}"
        )
        
    async def initialize(self):
        """Initialize async components and register MCP tools"""
        try:
            # Initialize collaborative reasoning even if AutoGen is not fully available
            if self.collaborative_reasoning:
                logger.info(f"Collaborative reasoning initialized for bottler {self.bottler_id}")
            
            if AUTOGEN_AVAILABLE:
                # Register direct database access tools with agents
                await self._register_direct_tools()
                logger.info(f"Initialized direct database tools for bottler {self.bottler_id}")
            else:
                logger.warning("AutoGen not available - skipping agent tools initialization")
                
        except Exception as e:
            logger.error(f"Failed to initialize AutoGen tools: {str(e)}")
                
    async def _register_direct_tools(self):
        """Register direct database access tools for agents"""
        if not AUTOGEN_AVAILABLE:
            return
            
        # Import Cosmos DB client
        try:
            from azure.cosmos import CosmosClient
            cosmos_endpoint = os.getenv("COSMOS_DB_ENDPOINT", os.getenv("COSMOS_ENDPOINT", ""))
            cosmos_key = os.getenv("COSMOS_DB_KEY", os.getenv("COSMOS_KEY", ""))
            
            if cosmos_endpoint and cosmos_key:
                cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
                database = cosmos_client.get_database_client("bottler-db")  # Correct database name
                self.cosmos_container = database.get_container_client("financial_data")
            else:
                logger.error("Cosmos DB credentials not found")
                self.cosmos_container = None
        except ImportError:
            logger.error("Azure Cosmos SDK not available")
            self.cosmos_container = None
            
        # Store tools for later use in process_hub_query
        self.direct_tools = {
            "query_cosmos_db": self._query_cosmos_db,
            "read_blob_storage": self._read_blob_storage,
            "write_blob_storage": self._write_blob_storage
        }
        
        logger.info("Direct database tools initialized (will be called directly, not registered)")
    
    async def _query_cosmos_db(self, container_name: str, query: str, max_items: int = 100) -> Dict[str, Any]:
        """Query Cosmos DB directly"""
        if not self.cosmos_container:
            return {"error": "Cosmos DB not configured"}
            
        try:
            items = list(self.cosmos_container.query_items(
                query=query,
                enable_cross_partition_query=True,
                max_item_count=max_items
            ))
            
            return {
                "success": True,
                "result": items,
                "count": len(items)
            }
        except Exception as e:
            logger.error(f"Cosmos DB query error: {e}")
            return {"error": str(e)}
        
    async def _read_blob_storage(self, blob_name: str) -> Dict[str, Any]:
        """Read from Blob Storage - placeholder"""
        return {"error": "Blob storage direct access not implemented"}
        
    async def _write_blob_storage(self, blob_name: str, content: str, metadata: Dict[str, str] = None) -> Dict[str, Any]:
        """Write to Blob Storage - placeholder"""
        return {"error": "Blob storage direct access not implemented"}
        
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
            
            # For now, use the user_proxy as the manager since GroupChatManager might not exist
            # in the new version or might have different initialization
            manager = self.user_proxy  # Simplified approach
            
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
            
            # Use user proxy as manager
            manager = self.user_proxy
            
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
            
    async def process_complex_task(self, task_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process complex tasks using collaborative reasoning system
        
        Args:
            task_payload: Task specification from upstream agent with structure:
                - task_description: High-level goal
                - data_context: Optional structured data
                - origin_agent: Source agent (e.g., TranslatorAgent)
                - priority: low/medium/high
                - timestamp: ISO format timestamp
                
        Returns:
            Execution plan or results
        """
        try:
            if not self.collaborative_reasoning:
                return {
                    "success": False,
                    "error": "Collaborative reasoning not available",
                    "fallback": "standard_processing"
                }
                
            logger.info(f"Processing complex task with collaborative reasoning: {task_payload.get('task_description', 'Unknown')}")
            
            # Process through collaborative reasoning
            result = await self.collaborative_reasoning.process_task(task_payload)
            
            # If plan is approved, check if we need to execute tools or just return analysis
            if result.get("status") == "success" and result.get("plan", {}).get("status") == "approved":
                # Check if this is an analysis query (not a task requiring tool execution)
                task_description = task_payload.get("task_description", "").lower()
                is_analysis_query = any(keyword in task_description for keyword in [
                    "cuales", "what", "analyze", "explain", "describe", "ventas", "sales", 
                    "revenue", "trends", "metrics", "performance", "comparison", "total",
                    "q1", "q2", "q3", "q4", "quarter", "month", "year"
                ])
                
                logger.info(f"Task description: {task_description}")
                logger.info(f"Is analysis query: {is_analysis_query}")
                
                if is_analysis_query:
                    # For analysis queries, we MUST execute the plan to query Cosmos DB
                    logger.info(f"Analysis query detected - Will execute plan to query Cosmos DB")
                    
                    # Debug: Log the plan structure
                    outer_plan = result.get("plan", {})
                    logger.info(f"Outer plan type: {type(outer_plan)}, Outer plan keys: {outer_plan.keys() if isinstance(outer_plan, dict) else 'N/A'}")
                    
                    # The actual plan might be nested
                    if isinstance(outer_plan, dict) and "plan" in outer_plan:
                        inner_plan = outer_plan.get("plan", {})
                        logger.info(f"Inner plan found, type: {type(inner_plan)}, keys: {inner_plan.keys() if isinstance(inner_plan, dict) else 'N/A'}")
                        plan_steps = inner_plan.get("steps", [])
                        logger.info(f"Plan steps from inner plan: {plan_steps}")
                        logger.info(f"Inner plan content: {inner_plan.get('content', 'NO CONTENT')[:200]}")
                    else:
                        plan_steps = outer_plan.get("steps", [])
                    has_cosmos_query = False
                    
                    for step in plan_steps:
                        if isinstance(step, dict):
                            # Check various fields for Cosmos DB reference
                            if (step.get("function", "").lower() in ["cosmos_query_documents", "query_documents"] or
                                "cosmos" in step.get("tool", "").lower() or
                                "cosmos" in step.get("action", "").lower() or
                                "cosmos" in step.get("description", "").lower() or
                                "query" in step.get("action", "").lower() and "financial_data" in step.get("action", "").lower()):
                                has_cosmos_query = True
                                logger.info(f"Found Cosmos query in step: {step}")
                                break
                        elif isinstance(step, str) and ("cosmos" in step.lower() or "query" in step.lower()):
                            has_cosmos_query = True
                            break
                    
                    if has_cosmos_query:
                        logger.info("Plan includes Cosmos DB query - executing to get real data")
                        # Make sure we pass the correct plan structure with steps
                        plan_to_execute = result.get("plan", {})
                        
                        # If the plan has a nested plan structure, use the inner plan
                        if isinstance(plan_to_execute, dict) and "plan" in plan_to_execute:
                            inner_plan = plan_to_execute["plan"]
                            # Ensure the inner plan has the steps
                            if isinstance(inner_plan, dict) and "steps" in inner_plan:
                                execution_plan = inner_plan
                            else:
                                # Create a plan with the detected steps
                                execution_plan = {
                                    "status": "approved",
                                    "steps": plan_steps,  # Use the steps we found
                                    "content": inner_plan.get("content", "")
                                }
                        else:
                            # Create a plan with the detected steps
                            execution_plan = {
                                "status": "approved",
                                "steps": plan_steps,  # Use the steps we found
                                "content": plan_to_execute.get("content", "")
                            }
                        
                        logger.info(f"Executing plan with {len(execution_plan.get('steps', []))} steps")
                        execution_result = await self.collaborative_reasoning.execute_approved_plan(execution_plan)
                        result["execution"] = execution_result
                    else:
                        logger.warning("Plan does NOT include Cosmos DB query in expected format - forcing execution anyway")
                        
                        # For analysis queries, force execution even if we can't detect the Cosmos query
                        logger.info("Forcing plan execution for analysis query")
                        
                        # Execute the plan anyway - the collaborative reasoning should handle it
                        try:
                            # Get the actual plan object from the nested structure
                            plan_to_execute = result.get("plan", {})
                            
                            # If the plan is nested, extract it
                            if isinstance(plan_to_execute, dict) and "plan" in plan_to_execute:
                                actual_plan = plan_to_execute["plan"]
                            else:
                                actual_plan = plan_to_execute
                            
                            # Ensure the plan has approved status
                            if isinstance(actual_plan, dict):
                                actual_plan["status"] = "approved"
                            else:
                                # Create a minimal plan structure
                                actual_plan = {
                                    "status": "approved",
                                    "steps": [],
                                    "content": str(actual_plan)
                                }
                            
                            logger.info(f"Executing plan with status: {actual_plan.get('status')}")
                            execution_result = await self.collaborative_reasoning.execute_approved_plan(actual_plan)
                            result["execution"] = execution_result
                        except Exception as exec_error:
                            logger.error(f"Error executing plan: {exec_error}")
                            # Fallback
                            plan_content = result.get("plan", {}).get("content", "")
                            final_plan = result.get("plan", {}).get("final_plan", "")
                            
                            logger.info(f"Plan content length: {len(plan_content)}")
                            logger.info(f"Final plan: {final_plan[:100] if final_plan else 'None'}...")
                            
                            # Create execution result with warning
                            execution_result = {
                                "status": "completed",
                                "results": [],
                                "summary": "WARNING: Plan execution failed",
                                "analysis": final_plan if final_plan else "No Cosmos DB data retrieved"
                            }
                            result["execution"] = execution_result
                elif task_payload.get("auto_execute", False):
                    # For actual tasks, execute the plan
                    execution_result = await self.collaborative_reasoning.execute_approved_plan(result["plan"])
                    result["execution"] = execution_result
                    
            return result
            
        except Exception as e:
            logger.error(f"Error in complex task processing: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "bottler_id": self.bottler_id
            }