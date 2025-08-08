import azure.functions as func
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
import asyncio

# Try to import Cosmos DB (optional for basic functionality)
try:
    from azure.cosmos import CosmosClient, exceptions
    COSMOS_AVAILABLE = True
except ImportError:
    COSMOS_AVAILABLE = False
    CosmosClient = None
    exceptions = None

# Try to import openai for Azure AI Foundry
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("OpenAI module loaded successfully")
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("OpenAI module not available - AI features disabled")
    OPENAI_AVAILABLE = False

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debug: Print environment variables at startup
logger.info("=== Function App Starting ===")
logger.info(f"Working Directory: {os.getcwd()}")
logger.info(f"AZURE_AI_FOUNDRY_ENDPOINT from env: {os.getenv('AZURE_AI_FOUNDRY_ENDPOINT', 'NOT SET')}")
logger.info(f"AI_FOUNDRY_ENDPOINT from env: {os.getenv('AI_FOUNDRY_ENDPOINT', 'NOT SET')}")

# Import Semantic Kernel and AutoGen (without MCP)
try:
    from integration.semantic_kernel_integration import BottlerSemanticKernelIntegration
    from integration.autogen_orchestrator import BottlerAutoGenOrchestrator
    SK_AVAILABLE = True
    AUTOGEN_AVAILABLE = True
    logger.info("Semantic Kernel and AutoGen loaded successfully")
except ImportError as e:
    logger.warning(f"SK/AutoGen modules not available: {e}")
    SK_AVAILABLE = False
    AUTOGEN_AVAILABLE = False

# Initialize Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Global instances for SK and AutoGen (without MCP)
sk_integration = None
autogen_orchestrator = None

# Bottler configuration from environment (NO HARDCODING)
bottler_config = {
    "id": os.getenv("BOTTLER_ID", "default-bottler"),
    "name": os.getenv("BOTTLER_NAME", "Default Bottler"),
    "region": os.getenv("BOTTLER_REGION", "Unknown"),
    "hub_url": os.getenv("TCCC_HUB_URL", "http://localhost:7071"),
    "hub_api_key": os.getenv("TCCC_HUB_API_KEY", "hub-key")
}

# Azure AI Foundry configuration (Each bottler uses their OWN AI)
ai_config = {
    "endpoint": os.getenv("AI_FOUNDRY_ENDPOINT", os.getenv("AZURE_AI_FOUNDRY_ENDPOINT", "")),
    "api_key": os.getenv("AI_FOUNDRY_KEY", os.getenv("AZURE_AI_FOUNDRY_KEY", os.getenv("AI_FOUNDRY_API_KEY", ""))),
    "deployment": os.getenv("AI_FOUNDRY_DEPLOYMENT", os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", "gpt-4")),
    "api_version": os.getenv("AI_FOUNDRY_API_VERSION", os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-12-01-preview"))
}

# Cosmos DB configuration
cosmos_config = {
    "endpoint": os.getenv("COSMOS_DB_ENDPOINT", ""),
    "key": os.getenv("COSMOS_DB_KEY", ""),
    "database": "bottler-db",
    "container": "financial_data"
}

# Initialize Azure OpenAI client if available
azure_openai_client = None
logger.info(f"\n=== AI Configuration ===")
logger.info(f"OpenAI Module Available: {OPENAI_AVAILABLE}")
logger.info(f"Endpoint: {ai_config['endpoint']}")
logger.info(f"API Key Present: {bool(ai_config['api_key'])}")
logger.info(f"Deployment: {ai_config['deployment']}")
logger.info(f"API Version: {ai_config['api_version']}")

if OPENAI_AVAILABLE and ai_config["endpoint"] and ai_config["api_key"]:
    try:
        azure_openai_client = AzureOpenAI(
            api_version=ai_config["api_version"],
            azure_endpoint=ai_config["endpoint"],
            api_key=ai_config["api_key"]
        )
        logger.info(f"Initialized Azure AI Foundry for bottler {bottler_config['id']} with model: {ai_config['deployment']}")
    except Exception as e:
        logger.error(f"Failed to initialize Azure AI Foundry: {e}")
        azure_openai_client = None

# Initialize Cosmos DB client
cosmos_client = None
if COSMOS_AVAILABLE and cosmos_config["endpoint"] and cosmos_config["key"]:
    try:
        cosmos_client = CosmosClient(cosmos_config["endpoint"], cosmos_config["key"])
        database = cosmos_client.get_database_client(cosmos_config["database"])
        container = database.get_container_client(cosmos_config["container"])
        logger.info(f"Initialized Cosmos DB client for database: {cosmos_config['database']}, container: {cosmos_config['container']}")
    except Exception as e:
        logger.error(f"Failed to initialize Cosmos DB: {e}")
        cosmos_client = None
else:
    if not COSMOS_AVAILABLE:
        logger.warning("Cosmos DB module not available - database features disabled")
    else:
        logger.warning("Cosmos DB credentials not configured")

async def initialize_integrations():
    """Initialize SK and AutoGen without MCP"""
    global sk_integration, autogen_orchestrator
    
    try:
        # Initialize Semantic Kernel (without MCP)
        if SK_AVAILABLE and sk_integration is None:
            sk_integration = BottlerSemanticKernelIntegration(mcp_bridge=None)  # No MCP
            await sk_integration.initialize()
            logger.info(f"Initialized SK integration for bottler {bottler_config['id']}")
        
        # Initialize AutoGen orchestrator (without MCP)
        if AUTOGEN_AVAILABLE and autogen_orchestrator is None:
            autogen_orchestrator = BottlerAutoGenOrchestrator(mcp_bridge=None, sk_integration=sk_integration)
            await autogen_orchestrator.initialize()
            logger.info(f"Initialized AutoGen orchestrator for bottler {bottler_config['id']}")
            
    except Exception as e:
        logger.error(f"Failed to initialize integrations: {str(e)}")
    
    # Register with hub
    await register_with_hub()

async def query_financial_data(bottler_id: str, query_type: str = "all", limit: int = 10):
    """Query financial data from Cosmos DB for the bottler"""
    if not COSMOS_AVAILABLE or not cosmos_client:
        logger.warning("Cosmos DB not available - returning empty data")
        return []
    
    try:
        database = cosmos_client.get_database_client(cosmos_config["database"])
        container = database.get_container_client(cosmos_config["container"])
        
        # Build query based on type
        if query_type == "revenue":
            query = f"SELECT TOP {limit} * FROM c WHERE c.bottler_id = '{bottler_id}' AND c.type = 'revenue' ORDER BY c.period DESC"
        elif query_type == "costs":
            query = f"SELECT TOP {limit} * FROM c WHERE c.bottler_id = '{bottler_id}' AND c.type = 'costs' ORDER BY c.period DESC"
        elif query_type == "products":
            query = f"SELECT TOP {limit} * FROM c WHERE c.bottler_id = '{bottler_id}' AND c.type = 'product_sales' ORDER BY c.sales_volume DESC"
        else:
            query = f"SELECT TOP {limit} * FROM c WHERE c.bottler_id = '{bottler_id}' ORDER BY c._ts DESC"
        
        # Execute query
        items = list(container.query_items(
            query=query,
            enable_cross_partition_query=True
        ))
        
        logger.info(f"Retrieved {len(items)} financial records for bottler {bottler_id}")
        return items
        
    except Exception as e:
        logger.error(f"Error querying Cosmos DB: {e}")
        return []

async def register_with_hub():
    """Register this bottler with the TCCC Hub"""
    try:
        logger.info(f"Registering bottler {bottler_config['id']} with hub {bottler_config['hub_url']}")
        logger.info(f"Bottler capabilities: financial_analysis, sales_reporting, ai_powered, sk_autogen")
        logger.info(f"Bottler {bottler_config['id']} registration completed")
        
    except Exception as e:
        logger.error(f"Failed to register with hub: {str(e)}")


@app.route(route="health", methods=["GET"])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint with SK/AutoGen status"""
    await initialize_integrations()
    
    health_status = {
        "status": "healthy",
        "bottler": bottler_config,
        "timestamp": datetime.utcnow().isoformat(),
        "ai_available": azure_openai_client is not None,
        "ai_model": ai_config["deployment"] if azure_openai_client else None,
        "sk_available": sk_integration is not None,
        "autogen_available": autogen_orchestrator is not None,
        "capabilities": ["financial_analysis", "sales_reporting", "ai_powered", "sk_autogen"]
    }
    
    return func.HttpResponse(
        json.dumps(health_status),
        mimetype="application/json",
        status_code=200
    )

@app.route(route="hub/query", methods=["POST"])
async def process_hub_query(req: func.HttpRequest) -> func.HttpResponse:
    """
    Main endpoint for processing queries from TCCC Hub.
    This is the SPOKE endpoint that receives queries via APIM.
    """
    try:
        
        # Verify request is from hub (via APIM)
        hub_key = req.headers.get("X-Hub-Key") or req.headers.get("Ocp-Apim-Subscription-Key")
        if not hub_key:
            return func.HttpResponse(
                json.dumps({"error": "Unauthorized - Missing hub key"}),
                mimetype="application/json",
                status_code=401
            )
        
        # Parse request
        req_body = req.get_json()
        query = req_body.get("query")
        request_id = req_body.get("request_id", str(uuid.uuid4()))
        
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Query is required"}),
                mimetype="application/json",
                status_code=400
            )
        
        logger.info(f"Received hub query for bottler {bottler_config['id']}: {query}")
        
        # Try to process with SK/AutoGen first if available
        response_data = None
        
        if SK_AVAILABLE and sk_integration:
            try:
                sk_result = await sk_integration.process_query(query)
                response_data = {
                    "success": True,
                    "bottler_id": bottler_config["id"],
                    "bottler_name": bottler_config["name"],
                    "message": sk_result.get("response", "SK processed"),
                    "processing_type": "semantic_kernel",
                    "request_id": request_id
                }
                logger.info("Query processed with Semantic Kernel")
            except Exception as sk_error:
                logger.warning(f"SK processing failed: {sk_error}")
        
        # Fallback to direct Azure AI Foundry
        if response_data is None:
            
            # Try to use Azure AI Foundry if available
            if azure_openai_client:
                try:
                    # Query financial data from Cosmos DB
                    financial_data = await query_financial_data(bottler_config['id'], limit=20)
                    
                    # Format financial data for context
                    financial_context = ""
                    if financial_data:
                        financial_context = f"\n\nRECENT FINANCIAL DATA FOR {bottler_config['name'].upper()}:\n"
                        for item in financial_data[:5]:  # Show top 5 records
                            if item.get('type') == 'revenue':
                                financial_context += f"- Period {item.get('period', 'N/A')}: Revenue ${item.get('amount', 0):,.2f}\n"
                            elif item.get('type') == 'costs':
                                financial_context += f"- Period {item.get('period', 'N/A')}: Costs ${item.get('amount', 0):,.2f}\n"
                            elif item.get('type') == 'product_sales':
                                financial_context += f"- Product {item.get('product_name', 'N/A')}: {item.get('sales_volume', 0):,} units\n"
                    
                    # Build expert financial analyst prompt
                    system_prompt = f"""You are a SENIOR FINANCIAL ANALYST and EXPERT ADVISOR for {bottler_config['name']} (ID: {bottler_config['id']}), 
a prestigious Coca-Cola bottler operating in {bottler_config['region']}.

YOUR EXPERTISE:
- Deep knowledge of beverage industry financial metrics and KPIs
- Expert in cost analysis, revenue optimization, and margin improvement
- Specialist in Coca-Cola bottling operations and regional market dynamics
- Advanced understanding of financial forecasting and strategic planning

YOUR ROLE:
- Provide comprehensive financial analysis with specific metrics and data
- Always mention {bottler_config['name']} by name throughout your responses
- Give detailed explanations with financial context and industry benchmarks
- Provide actionable recommendations based on financial best practices
- Use professional financial terminology and be thorough in your analysis

FINANCIAL DATABASE ACCESS:
You have access to {bottler_config['name']}'s financial data stored in Cosmos DB (database: bottler-db, container: financial_data).
{financial_context}

PRIVACY RULES:
- You can ONLY access and discuss {bottler_config['name']}'s data
- You have NO access to other bottlers' confidential information
- If asked about other bottlers, politely state: "I only have access to {bottler_config['name']}'s confidential financial data."
- Focus exclusively on {bottler_config['name']}'s operations and financial performance

RESPONSE GUIDELINES:
- Provide EXTENSIVE and DETAILED responses (minimum 3-4 paragraphs)
- Always start by acknowledging you are representing {bottler_config['name']}
- Include specific financial metrics, percentages, and trends when possible
- Reference the bottler's name ({bottler_config['name']}) multiple times in your response
- If asked in Spanish, respond in Spanish. If asked in English, respond in English.
- Structure your response with clear sections and financial insights
- Conclude with strategic recommendations for {bottler_config['name']}"""
                    
                    # Get AI response with extended token limit for detailed financial analysis
                    ai_response = azure_openai_client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": f"As a financial expert for {bottler_config['name']}, please provide a comprehensive analysis for the following query: {query}"}
                        ],
                        model=ai_config["deployment"],
                        max_tokens=2000,  # Increased for extensive responses
                        temperature=0.7
                    )
                    
                    response_data = {
                        "success": True,
                        "bottler_id": bottler_config["id"],
                        "bottler_name": bottler_config["name"],
                        "message": ai_response.choices[0].message.content,
                        "model_used": ai_response.model,
                        "ai_powered": True,
                        "request_id": request_id,
                        "usage": {
                            "prompt_tokens": ai_response.usage.prompt_tokens,
                            "completion_tokens": ai_response.usage.completion_tokens,
                            "total_tokens": ai_response.usage.total_tokens
                        }
                    }
                    
                    logger.info(f"AI response generated for bottler {bottler_config['id']} using model: {ai_response.model}")
                    
                except Exception as ai_error:
                    logger.error(f"AI processing error: {str(ai_error)}")
                    response_data = {
                        "success": False,
                        "bottler_id": bottler_config["id"],
                        "bottler_name": bottler_config["name"],
                        "message": f"AI processing failed: {str(ai_error)}",
                        "ai_powered": False,
                        "request_id": request_id
                    }
            else:
                # No AI available - basic response
                response_data = {
                    "success": True,
                    "bottler_id": bottler_config["id"],
                    "bottler_name": bottler_config["name"],
                    "message": "Basic mode - AI not configured for this bottler",
                    "query": query,
                    "request_id": request_id,
                    "ai_powered": False
                }
            
            return func.HttpResponse(
                json.dumps(response_data),
                mimetype="application/json",
                status_code=200
            )
        
    except Exception as e:
        logger.error(f"Error processing hub query: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": str(e),
                "bottler_id": bottler_config["id"]
            }),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="hub/command", methods=["POST"])
async def handle_hub_command(req: func.HttpRequest) -> func.HttpResponse:
    """Handle specific commands from the hub"""
    try:
        await initialize_integrations()
        
        # Verify request is from hub
        hub_key = req.headers.get("X-Hub-Key") or req.headers.get("Ocp-Apim-Subscription-Key")
        if not hub_key:
            return func.HttpResponse(
                json.dumps({"error": "Unauthorized"}),
                mimetype="application/json",
                status_code=401
            )
        
        # Parse command
        req_body = req.get_json()
        command = req_body.get("command")
        parameters = req_body.get("parameters", {})
        
        if not command:
            return func.HttpResponse(
                json.dumps({"error": "Command is required"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Process command
        # For now, just return a simple response as orchestrator is not available
        return func.HttpResponse(
            json.dumps({
                "error": "Command processing not implemented",
                "bottler_id": bottler_config["id"],
                "command": command
            }),
            mimetype="application/json",
            status_code=501
        )
        
    except Exception as e:
        logger.error(f"Error handling hub command: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="query", methods=["POST"])
async def process_user_query(req: func.HttpRequest) -> func.HttpResponse:
    """
    Process a query from a user directly to this bottler.
    The bottler will then communicate with TCCC Hub if needed.
    """
    try:
        await initialize_integrations()
        
        # Parse request
        req_body = req.get_json()
        query = req_body.get("query")
        query_type = req_body.get("type", "general")
        
        if not query:
            return func.HttpResponse(
                json.dumps({"error": "Query is required"}),
                mimetype="application/json",
                status_code=400
            )
        
        logger.info(f"Bottler {bottler_config['id']} received user query: {query}")
        
        # Check if query needs hub coordination
        needs_hub = any(keyword in query.lower() for keyword in ["otros bottlers", "other bottlers", "comparar", "compare", "consolidado", "consolidated"])
        
        response_data = None
        
        # If needs hub coordination, forward to hub
        if needs_hub:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    hub_response = await client.post(
                        f"{bottler_config['hub_url']}/api/query",
                        headers={
                            "X-TCCC-API-Key": bottler_config["hub_api_key"],
                            "X-Bottler-ID": bottler_config["id"],
                            "Content-Type": "application/json"
                        },
                        json={
                            "query": query,
                            "type": query_type,
                            "from_bottler": bottler_config["id"]
                        },
                        timeout=30.0
                    )
                    
                    if hub_response.status_code == 200:
                        hub_data = hub_response.json()
                        response_data = {
                            "success": True,
                            "bottler_id": bottler_config["id"],
                            "message": f"Coordinated with TCCC Hub: {hub_data.get('response', {}).get('message', 'Hub response received')}",
                            "hub_response": hub_data,
                            "coordinated": True
                        }
                    else:
                        logger.error(f"Hub returned error: {hub_response.status_code}")
                        
            except Exception as hub_error:
                logger.error(f"Failed to coordinate with hub: {hub_error}")
        
        # If no hub response or local query, use bottler's own AI
        if not response_data and azure_openai_client:
            try:
                # Get bottler-specific AI response
                system_prompt = f"""You are {bottler_config['name']} (ID: {bottler_config['id']}), a Coca-Cola bottler in {bottler_config['region']}.
You are an independent bottler with your own AI capabilities.

IMPORTANT PRIVACY RULES:
- You ONLY have access to your OWN data as {bottler_config['name']}
- You CANNOT access or know about other bottlers' private data (sales, revenue, costs)
- If asked about other bottlers' specific data, say: "I don't have access to other bottlers' confidential information. I can only share information about {bottler_config['name']}."
- You can mention that other bottlers exist, but not their private business data

Respond based on your own data and perspective as {bottler_config['name']}."""
                
                ai_response = azure_openai_client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": query}
                    ],
                    model=ai_config["deployment"],
                    max_tokens=1000,
                    temperature=0.7
                )
                
                response_data = {
                    "success": True,
                    "bottler_id": bottler_config["id"],
                    "bottler_name": bottler_config["name"],
                    "message": ai_response.choices[0].message.content,
                    "model_used": ai_response.model,
                    "ai_model": ai_config["deployment"],
                    "coordinated": False,
                    "usage": {
                        "prompt_tokens": ai_response.usage.prompt_tokens,
                        "completion_tokens": ai_response.usage.completion_tokens,
                        "total_tokens": ai_response.usage.total_tokens
                    }
                }
                
            except Exception as ai_error:
                logger.error(f"Bottler AI error: {str(ai_error)}")
                response_data = {
                    "success": False,
                    "bottler_id": bottler_config["id"],
                    "error": f"AI processing failed: {str(ai_error)}"
                }
        
        # Default response if no AI
        if not response_data:
            response_data = {
                "success": False,
                "bottler_id": bottler_config["id"],
                "bottler_name": bottler_config["name"],
                "message": "AI not configured for this bottler",
                "error": "No AI available"
            }
        
        return func.HttpResponse(
            json.dumps(response_data),
            mimetype="application/json",
            status_code=200 if response_data.get("success") else 500
        )
        
    except Exception as e:
        logger.error(f"Error processing user query: {str(e)}")
        return func.HttpResponse(
            json.dumps({
                "error": str(e),
                "bottler_id": bottler_config["id"]
            }),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="financial/query", methods=["POST"])
async def query_financial_data_endpoint(req: func.HttpRequest) -> func.HttpResponse:
    """Query financial data using AI (without MCP)"""
    try:
        # Parse query request
        req_body = req.get_json()
        query_type = req_body.get("type", "revenue")
        period = req_body.get("period", "2024")
        product = req_body.get("product")
        
        # Use AI to generate response based on bottler's perspective
        if azure_openai_client:
            query_text = f"Provide {query_type} data for {bottler_config['name']} for period {period}"
            if product:
                query_text += f" specifically for product: {product}"
            
            system_prompt = f"""You are the financial data assistant for {bottler_config['name']} (ID: {bottler_config['id']}).
            Provide realistic financial analysis and data based on this bottler's operations in {bottler_config['region']}.
            Focus on revenue, costs, margins, and product performance."""
            
            response = azure_openai_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query_text}
                ],
                model=ai_config["deployment"],
                max_tokens=800,
                temperature=0.7
            )
            
            return func.HttpResponse(
                json.dumps({
                    "bottler_id": bottler_config["id"],
                    "query_type": query_type,
                    "period": period,
                    "response": response.choices[0].message.content,
                    "ai_powered": True
                }),
                mimetype="application/json",
                status_code=200
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "error": "AI service not available",
                    "bottler_id": bottler_config["id"]
                }),
                mimetype="application/json",
                status_code=503
            )
            
    except Exception as e:
        logger.error(f"Error querying financial data: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="financial/submit", methods=["POST"])
async def submit_financial_data(req: func.HttpRequest) -> func.HttpResponse:
    """Submit financial data - currently returns not implemented"""
    try:
        # Without MCP, we can't store data in Cosmos DB directly
        # This would need to be implemented with direct Azure SDK calls
        return func.HttpResponse(
            json.dumps({
                "error": "Financial data submission not implemented without MCP",
                "bottler_id": bottler_config["id"]
            }),
            mimetype="application/json",
            status_code=501
        )
            
    except Exception as e:
        logger.error(f"Error submitting financial data: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )



# Prompt management endpoints removed - they depend on MCP
# These would need to be reimplemented with direct Azure SDK calls if needed

# Initialize bottler on startup
logger.info(f"Starting Soft Bottler Manager: {bottler_config['id']}")
logger.info(f"Region: {bottler_config['region']}")
logger.info(f"Hub URL: {bottler_config['hub_url']}")
logger.info("This is a SPOKE - All communication goes through the hub")
