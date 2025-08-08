"""
Semantic Kernel Integration for Bottler SPOKE
===========================================

This module integrates Semantic Kernel with MCP Bridge for the bottler agent.
Processes queries from the hub and analyzes them using SK.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

import os
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from datetime import datetime
import json

# Type checking imports
if TYPE_CHECKING:
    from integration.mcp import MCPBridge

# Semantic Kernel imports
try:
    import semantic_kernel as sk
    from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
    from semantic_kernel.contents.chat_history import ChatHistory
    from semantic_kernel.functions import kernel_function
    SEMANTIC_KERNEL_AVAILABLE = True
except ImportError:
    SEMANTIC_KERNEL_AVAILABLE = False
    # Create dummy classes for type hints
    class sk:
        class Kernel: pass

logger = logging.getLogger(__name__)


class BottlerSemanticKernelIntegration:
    """
    Semantic Kernel integration for Bottler SPOKE.
    Analyzes queries from hub and prepares data for AutoGen processing.
    """
    
    def __init__(self, mcp_bridge: Optional["MCPBridge"] = None):
        """Initialize Semantic Kernel integration for bottler"""
        self.mcp_bridge = mcp_bridge
        self.bottler_id = os.getenv("BOTTLER_ID", "unknown")
        self.bottler_name = os.getenv("BOTTLER_NAME", "Unknown Bottler")
        self.bottler_region = os.getenv("BOTTLER_REGION", "Unknown")
        
        if not SEMANTIC_KERNEL_AVAILABLE:
            logger.error("Semantic Kernel not available - query analysis will be limited")
            return
            
        # Initialize kernel
        self.kernel = sk.Kernel()
        
        # Initialize AI service
        self._setup_ai_service()
        
        # Initialize functions
        self._setup_functions()
        
        # Note: Call initialize() after creation for async initialization
        
    def _setup_ai_service(self):
        """Setup AI service for Semantic Kernel using Azure AI Foundry"""
        if not SEMANTIC_KERNEL_AVAILABLE:
            return
            
        try:
            # Get Azure AI Foundry configuration from environment variables
            api_key = os.getenv("AZURE_AI_FOUNDRY_KEY")
            endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT")
            deployment = os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", "tccc-model-router")
            api_version = os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-12-01-preview")
            
            if not api_key or not endpoint:
                raise ValueError("Azure AI Foundry configuration missing: AZURE_AI_FOUNDRY_KEY and AZURE_AI_FOUNDRY_ENDPOINT are required")
            
            # Always use Azure AI Foundry (never standard OpenAI)
            chat_completion = AzureChatCompletion(
                deployment_name=deployment,
                endpoint=endpoint,
                api_key=api_key,
                api_version=api_version,
                service_id="chat_completion"
            )
            
            # Add the service to the kernel
            self.kernel.add_service(chat_completion)
                
            logger.info(f"Initialized Azure AI Foundry service for bottler {self.bottler_id}")
            logger.info(f"Using endpoint: {endpoint}")
            logger.info(f"Using deployment: {deployment}")
            logger.info(f"Using API version: {api_version}")
            
        except Exception as e:
            logger.error(f"Failed to setup AI service: {str(e)}")
            
    def _setup_functions(self):
        """Setup bottler-specific functions"""
        if not SEMANTIC_KERNEL_AVAILABLE:
            return
            
        try:
            # Create query analyzer function
            @kernel_function(
                description="Analyze a query from TCCC Hub and extract key information",
                name="analyze_query"
            )
            def analyze_query(query: str) -> str:
                """Analyze query and return structured information as JSON"""
                prompt = f"""You are a financial analyst for {self.bottler_name} bottler in {self.bottler_region}.
Analyze the following query from TCCC Hub and extract key information:

Query: {query}

Extract and return as JSON:
1. query_type: (sales, revenue, costs, inventory, performance)
2. products: List of products mentioned (e.g., ["Coca-Cola", "Sprite", "Fanta"])
3. time_period: Time period requested (e.g., "last month", "Q1 2024", "2024")
4. metrics: Specific metrics requested (e.g., ["total_sales", "revenue", "units_sold"])
5. filters: Any filters mentioned (e.g., {{"region": "Mexico", "channel": "retail"}})
6. aggregation: How to aggregate data (sum, average, by_product, by_period)

Return ONLY valid JSON."""
                
                return self._execute_prompt(prompt)
            
            # Create data query builder function
            @kernel_function(
                description="Build a Cosmos DB query based on analysis",
                name="build_cosmos_query"
            )
            def build_cosmos_query(analysis: str, container: str = "financial_data") -> str:
                """Build Cosmos DB query from analysis"""
                prompt = f"""Build a Cosmos DB query for {self.bottler_name} based on this analysis:

Analysis: {analysis}
Container: {container}

Generate a SQL query that:
1. Filters by bottler_id = '{self.bottler_id}'
2. Applies all relevant filters from the analysis
3. Selects appropriate fields
4. Orders results appropriately

Return ONLY the SQL query string, no explanation."""
                
                return self._execute_prompt(prompt)
            
            # Create response formatter function
            @kernel_function(
                description="Format query results for sending back to TCCC Hub",
                name="format_response"
            )
            def format_response(analysis: str, results: str, record_count: str) -> str:
                """Format results for hub response"""
                prompt = f"""Format the query results for {self.bottler_name} to send back to TCCC Hub.

Query Analysis: {analysis}
Raw Results: {results}
Record Count: {record_count}

Create a professional summary that includes:
1. Direct answer to the query
2. Key metrics and totals
3. Relevant breakdowns (by product, period, etc.)
4. Data quality notes if applicable

Format as structured JSON response."""
                
                return self._execute_prompt(prompt)
            
            # Add functions to kernel
            self.kernel.add_function(plugin_name="QueryAnalysis", function=analyze_query)
            self.kernel.add_function(plugin_name="DataQuery", function=build_cosmos_query)
            self.kernel.add_function(plugin_name="ResponseFormat", function=format_response)
            
        except Exception as e:
            logger.error(f"Failed to setup functions: {str(e)}")
            
    def _execute_prompt(self, prompt: str) -> str:
        """Execute a prompt using the kernel"""
        if not SEMANTIC_KERNEL_AVAILABLE:
            return json.dumps({"error": "Semantic Kernel not available"})
            
        try:
            # Create chat history
            history = ChatHistory()
            history.add_user_message(prompt)
            
            # Get chat completion service
            chat_service = self.kernel.get_service("chat_completion")
            
            # Execute the prompt
            response = chat_service.get_chat_message_content(
                chat_history=history,
                settings={"max_tokens": 1000, "temperature": 0.1}
            )
            
            return str(response.content)
            
        except Exception as e:
            logger.error(f"Error executing prompt: {str(e)}")
            return json.dumps({"error": str(e)})
        
    async def initialize(self):
        """Initialize async components including MCP tool discovery"""
        if not SEMANTIC_KERNEL_AVAILABLE:
            logger.warning("Semantic Kernel not available - initialization skipped")
            return
            
        if self.mcp_bridge:
            try:
                # Discover available MCP tools
                tools = await self.mcp_bridge.list_tools()
                logger.info(f"Discovered {len(tools.get('tools', {}))} MCP tools for bottler {self.bottler_id}")
                
                # Register bottler-specific functions with MCP awareness
                await self._register_mcp_functions()
                
            except Exception as e:
                logger.error(f"Failed to initialize MCP tools: {str(e)}")
                
    async def _register_mcp_functions(self):
        """Register MCP-aware functions for the bottler"""
        if not SEMANTIC_KERNEL_AVAILABLE or not self.mcp_bridge:
            return
            
        try:
            # Create MCP-integrated functions
            @kernel_function(
                description="Query financial data from bottler's Cosmos DB",
                name="query_financial_data"
            )
            async def query_financial_data(query: str) -> str:
                """Query financial data using MCP"""
                result = await self.mcp_bridge.execute_tool(
                    server_name="cosmos",
                    tool_name="query_documents",
                    arguments={
                        "container": "financial_data",
                        "query": query,
                        "max_items": 1000
                    }
                )
                
                if result.get("success"):
                    return json.dumps(result.get("result", []))
                else:
                    return json.dumps({"error": result.get("error")})
                    
            # Add the function to kernel
            self.kernel.add_function(plugin_name="MCPDataAccess", function=query_financial_data)
            
        except Exception as e:
            logger.error(f"Failed to register MCP functions: {str(e)}")
        
    async def analyze_hub_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze a query from the hub using Semantic Kernel
        
        Args:
            query: Natural language query from hub
            context: Additional context (e.g., specific products, time ranges)
            
        Returns:
            Analysis results with query interpretation
        """
        try:
            if not SEMANTIC_KERNEL_AVAILABLE:
                # Fallback analysis without SK
                return self._fallback_analysis(query, context)
                
            logger.info(f"Analyzing hub query for bottler {self.bottler_id}: {query}")
            
            # Get the analyze function
            analyze_func = self.kernel.get_function("QueryAnalysis", "analyze_query")
            
            # Execute analysis
            result = await analyze_func.invoke(self.kernel, query=query)
            analysis_text = str(result.value)
            
            # Parse the analysis
            try:
                analysis = json.loads(analysis_text)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse analysis result: {analysis_text}")
                return self._fallback_analysis(query, context)
                
            # Add bottler context to analysis
            analysis["bottler_id"] = self.bottler_id
            analysis["bottler_name"] = self.bottler_name
            analysis["bottler_region"] = self.bottler_region
            analysis["analysis_timestamp"] = datetime.utcnow().isoformat()
            
            # Add any additional context
            if context:
                analysis["additional_context"] = context
            
            logger.info(f"Query analysis complete for bottler {self.bottler_id}: {analysis}")
            
            return {
                "success": True,
                "analysis": analysis,
                "original_query": query
            }
            
        except Exception as e:
            logger.error(f"Error analyzing hub query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "original_query": query,
                "fallback_analysis": self._fallback_analysis(query, context)
            }
            
    def _fallback_analysis(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Fallback analysis when SK is not available"""
        query_lower = query.lower()
        
        # Basic pattern matching
        analysis = {
            "query_type": "unknown",
            "products": [],
            "time_period": None,
            "metrics": [],
            "filters": {},
            "aggregation": "sum",
            "bottler_id": self.bottler_id,
            "bottler_name": self.bottler_name,
            "bottler_region": self.bottler_region,
            "analysis_method": "fallback_pattern_matching",
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
        
        # Detect query type
        if any(word in query_lower for word in ["sales", "revenue", "ingresos"]):
            analysis["query_type"] = "sales"
        elif any(word in query_lower for word in ["cost", "costo", "expense"]):
            analysis["query_type"] = "costs"
        elif any(word in query_lower for word in ["inventory", "stock"]):
            analysis["query_type"] = "inventory"
            
        # Detect products
        products = ["coca-cola", "sprite", "fanta", "powerade", "del valle"]
        for product in products:
            if product in query_lower:
                analysis["products"].append(product.title())
                
        # Detect time periods
        if any(word in query_lower for word in ["month", "monthly"]):
            analysis["time_period"] = "monthly"
        elif any(word in query_lower for word in ["year", "annual"]):
            analysis["time_period"] = "yearly"
            
        return {
            "success": True,
            "analysis": analysis,
            "original_query": query
        }
            
    async def build_data_query(self, analysis: Dict[str, Any], container: str = "financial_data") -> str:
        """
        Build a Cosmos DB query based on the analysis
        
        Args:
            analysis: Query analysis from analyze_hub_query
            container: Target container name
            
        Returns:
            SQL query string
        """
        try:
            if not SEMANTIC_KERNEL_AVAILABLE:
                return self._fallback_query_builder(analysis, container)
                
            # Get the query builder function
            build_func = self.kernel.get_function("DataQuery", "build_cosmos_query")
            
            # Execute query building
            result = await build_func.invoke(
                self.kernel, 
                analysis=json.dumps(analysis),
                container=container
            )
            
            query = str(result.value).strip()
            
            # Ensure bottler_id filter is always present
            if f"c.bottler_id = '{self.bottler_id}'" not in query:
                if "WHERE" in query:
                    query = query.replace("WHERE", f"WHERE c.bottler_id = '{self.bottler_id}' AND")
                else:
                    query += f" WHERE c.bottler_id = '{self.bottler_id}'"
                    
            logger.info(f"Built Cosmos query for bottler {self.bottler_id}: {query}")
            
            return query
            
        except Exception as e:
            logger.error(f"Error building data query: {str(e)}")
            return self._fallback_query_builder(analysis, container)
            
    def _fallback_query_builder(self, analysis: Dict[str, Any], container: str) -> str:
        """Fallback query builder when SK is not available"""
        query = f"SELECT * FROM c WHERE c.bottler_id = '{self.bottler_id}'"
        
        # Add basic filters based on analysis
        if analysis.get("query_type") in ["sales", "revenue"]:
            query += " AND c.type = 'financial_record'"
            
        # Add product filters
        products = analysis.get("products", [])
        if products:
            product_filter = " OR ".join([f"CONTAINS(c.product_name, '{p}')" for p in products])
            query += f" AND ({product_filter})"
            
        # Add time filter
        time_period = analysis.get("time_period")
        if time_period == "monthly":
            current_month = datetime.utcnow().strftime("%Y-%m")
            query += f" AND c.period = '{current_month}'"
        elif time_period == "yearly":
            current_year = datetime.utcnow().strftime("%Y")
            query += f" AND STARTSWITH(c.period, '{current_year}')"
            
        return query
            
    async def format_response(self, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Format query results for sending back to hub
        
        Args:
            analysis: Original query analysis
            results: Raw query results from Cosmos DB
            
        Returns:
            Formatted response for hub
        """
        try:
            if not SEMANTIC_KERNEL_AVAILABLE:
                return self._fallback_formatter(analysis, results)
                
            # Get the formatter function
            format_func = self.kernel.get_function("ResponseFormat", "format_response")
            
            # Execute formatting
            result = await format_func.invoke(
                self.kernel,
                analysis=json.dumps(analysis),
                results=json.dumps(results[:50]),  # Limit for token size
                record_count=str(len(results))
            )
            
            try:
                formatted_response = json.loads(str(result.value))
            except json.JSONDecodeError:
                # Fallback to basic formatting
                return self._fallback_formatter(analysis, results)
                
            # Add metadata
            formatted_response["bottler_id"] = self.bottler_id
            formatted_response["bottler_name"] = self.bottler_name
            formatted_response["query_analysis"] = analysis
            formatted_response["response_timestamp"] = datetime.utcnow().isoformat()
            
            return formatted_response
            
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return self._fallback_formatter(analysis, results)
            
    def _fallback_formatter(self, analysis: Dict[str, Any], results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback formatter when SK is not available"""
        # Basic response formatting
        total_records = len(results)
        
        # Calculate basic metrics if financial data
        total_revenue = 0
        total_costs = 0
        
        if analysis.get("query_type") in ["sales", "revenue"]:
            for record in results:
                total_revenue += record.get("revenue", 0)
                total_costs += record.get("costs", 0)
                
        response = {
            "bottler_id": self.bottler_id,
            "bottler_name": self.bottler_name,
            "summary": f"Found {total_records} records for {analysis.get('query_type', 'unknown')} query",
            "total_records": total_records,
            "data": results[:10],  # First 10 records
            "metrics": {
                "total_revenue": total_revenue,
                "total_costs": total_costs,
                "net_profit": total_revenue - total_costs
            } if total_revenue > 0 else {},
            "query_analysis": analysis,
            "response_timestamp": datetime.utcnow().isoformat(),
            "formatting_method": "fallback"
        }
        
        return response
            
    async def process_hub_query(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Complete pipeline: analyze query, fetch data, format response
        
        Args:
            query: Natural language query from hub
            context: Additional context
            
        Returns:
            Complete response ready for hub
        """
        try:
            # Step 1: Analyze the query
            analysis_result = await self.analyze_hub_query(query, context)
            
            if not analysis_result.get("success"):
                return analysis_result
                
            analysis = analysis_result["analysis"]
            
            # Step 2: Build data query
            cosmos_query = await self.build_data_query(analysis)
            
            # Step 3: Execute query using MCP
            if self.mcp_bridge:
                query_result = await self.mcp_bridge.execute_tool(
                    server_name="cosmos",
                    tool_name="query_documents",
                    arguments={
                        "container": "financial_data",
                        "query": cosmos_query,
                        "max_items": 1000
                    }
                )
                
                if query_result.get("success"):
                    results = query_result.get("result", [])
                else:
                    return {
                        "success": False,
                        "error": query_result.get("error"),
                        "bottler_id": self.bottler_id
                    }
            else:
                # Fallback for testing
                results = []
                
            # Step 4: Format response
            formatted_response = await self.format_response(analysis, results)
            
            return {
                "success": True,
                "response": formatted_response,
                "bottler_id": self.bottler_id,
                "processing_complete": True
            }
            
        except Exception as e:
            logger.error(f"Error processing hub query: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "bottler_id": self.bottler_id
            }
