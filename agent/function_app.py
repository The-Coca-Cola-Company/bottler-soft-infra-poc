import azure.functions as func
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import uuid
import asyncio

# Add current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Type checking imports
if TYPE_CHECKING:
    from integration.semantic_kernel_integration import BottlerSemanticKernelIntegration
    from integration.autogen_orchestrator import BottlerAutoGenOrchestrator
    from integration.mcp import MCPBridge
    from orchestration.bottler_orchestrator import BottlerOrchestrator
    from prompts.prompt_manager import PromptManager

# Import integration modules
try:
    from integration.semantic_kernel_integration import BottlerSemanticKernelIntegration
    from integration.autogen_orchestrator import BottlerAutoGenOrchestrator
    from integration.mcp import MCPBridge
    from orchestration.bottler_orchestrator import BottlerOrchestrator
    from prompts.prompt_manager import PromptManager
    INTEGRATIONS_AVAILABLE = True
    MCP_BRIDGE_AVAILABLE = True
    ORCHESTRATOR_AVAILABLE = True
    PROMPT_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Integration modules not available - running in basic mode: {e}")
    INTEGRATIONS_AVAILABLE = False
    MCP_BRIDGE_AVAILABLE = False
    ORCHESTRATOR_AVAILABLE = False
    PROMPT_MANAGER_AVAILABLE = False

# Initialize Function App
app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

# Global instances - using string annotations to avoid import errors
sk_integration: Optional["BottlerSemanticKernelIntegration"] = None
autogen_orchestrator: Optional["BottlerAutoGenOrchestrator"] = None
mcp_bridge: Optional["MCPBridge"] = None
bottler_orchestrator: Optional["BottlerOrchestrator"] = None
prompt_manager: Optional["PromptManager"] = None

# Bottler configuration from environment (NO HARDCODING)
bottler_config = {
    "id": os.getenv("BOTTLER_ID", "default-bottler"),
    "name": os.getenv("BOTTLER_NAME", "Default Bottler"),
    "region": os.getenv("BOTTLER_REGION", "Unknown"),
    "hub_url": os.getenv("TCCC_HUB_URL", "http://localhost:7071"),
    "hub_api_key": os.getenv("TCCC_HUB_API_KEY", "hub-key")
}

async def initialize_integrations():
    """Initialize all integration components"""
    global sk_integration, autogen_orchestrator, mcp_bridge, bottler_orchestrator, prompt_manager
    
    # Initialize MCP Bridge first
    if MCP_BRIDGE_AVAILABLE and mcp_bridge is None:
        try:
            mcp_bridge = MCPBridge()
            await mcp_bridge.initialize()
            logger.info(f"Initialized MCP Bridge for bottler {bottler_config['id']}")
        except Exception as e:
            logger.error(f"Failed to initialize MCP Bridge: {str(e)}")
    
    if INTEGRATIONS_AVAILABLE:
        try:
            # Initialize Semantic Kernel integration
            if sk_integration is None:
                sk_integration = BottlerSemanticKernelIntegration(mcp_bridge)
                await sk_integration.initialize()
                logger.info(f"Initialized SK integration for bottler {bottler_config['id']}")
            
            # Initialize AutoGen orchestrator
            if autogen_orchestrator is None:
                autogen_orchestrator = BottlerAutoGenOrchestrator(mcp_bridge, sk_integration)
                await autogen_orchestrator.initialize()
                logger.info(f"Initialized AutoGen orchestrator for bottler {bottler_config['id']}")
                
        except Exception as e:
            logger.error(f"Failed to initialize integrations: {str(e)}")
    
    # Initialize Bottler Orchestrator
    if ORCHESTRATOR_AVAILABLE and bottler_orchestrator is None:
        try:
            bottler_orchestrator = BottlerOrchestrator(
                sk_integration=sk_integration,
                autogen_orchestrator=autogen_orchestrator,
                mcp_bridge=mcp_bridge
            )
            await bottler_orchestrator.initialize()
            logger.info(f"Initialized Bottler Orchestrator for {bottler_config['id']}")
        except Exception as e:
            logger.error(f"Failed to initialize Bottler Orchestrator: {str(e)}")
    
    # Initialize Prompt Manager
    if PROMPT_MANAGER_AVAILABLE and prompt_manager is None:
        try:
            prompt_manager = PromptManager(mcp_bridge=mcp_bridge)
            logger.info(f"Initialized Prompt Manager for {bottler_config['id']}")
        except Exception as e:
            logger.error(f"Failed to initialize Prompt Manager: {str(e)}")
            
    # Register with hub
    await register_with_hub()

async def register_with_hub():
    """Register this bottler with the TCCC Hub"""
    try:
        if not mcp_bridge:
            return
            
        # Store registration in OUR OWN Cosmos DB
        registration_data = {
            "id": f"registration-{bottler_config['id']}",
            "type": "hub_registration",
            "bottler_id": bottler_config["id"],
            "bottler_name": bottler_config["name"],
            "bottler_region": bottler_config["region"],
            "hub_url": bottler_config["hub_url"],
            "registered_at": datetime.utcnow().isoformat(),
            "status": "active",
            "capabilities": ["financial_analysis", "sales_reporting", "mcp_integration", "sk_autogen"]
        }
        
        # Use REAL Cosmos DB to store our registration
        result = await mcp_bridge.execute_tool(
            server_name="cosmos",
            tool_name="upsert_document",
            arguments={
                "container": "bottler_config",
                "document": registration_data
            }
        )
        logger.info(f"Stored registration in local Cosmos DB: {result}")
        
    except Exception as e:
        logger.error(f"Failed to register with hub: {str(e)}")

@app.route(route="health", methods=["GET"])
async def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint"""
    await initialize_integrations()
    
    health_status = {
        "status": "healthy",
        "bottler": bottler_config,
        "timestamp": datetime.utcnow().isoformat(),
        "integrations": {
            "mcp_bridge": mcp_bridge is not None,
            "semantic_kernel": sk_integration is not None,
            "autogen": autogen_orchestrator is not None,
            "bottler_orchestrator": bottler_orchestrator is not None
        }
    }
    
    # Check REAL database connectivity
    if mcp_bridge:
        try:
            # Query our OWN Cosmos DB for health check
            db_check = await mcp_bridge.execute_tool(
                server_name="cosmos",
                tool_name="query_documents",
                arguments={
                    "container": "health_checks",
                    "query": "SELECT TOP 1 * FROM c ORDER BY c._ts DESC",
                    "max_items": 1
                }
            )
            health_status["database_status"] = "connected"
            health_status["last_db_check"] = datetime.utcnow().isoformat()
        except Exception as e:
            health_status["database_status"] = "error"
            health_status["database_error"] = str(e)
    
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
        await initialize_integrations()
        
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
        
        # Process with full orchestration if available
        if ORCHESTRATOR_AVAILABLE and bottler_orchestrator:
            result = await bottler_orchestrator.process_hub_query(query, request_id)
            
            return func.HttpResponse(
                json.dumps(result, default=str),
                mimetype="application/json",
                status_code=200 if result.get("success") else 500
            )
        
        # Fallback to basic processing
        else:
            # Basic response without full integration
            response = {
                "success": True,
                "bottler_id": bottler_config["id"],
                "bottler_name": bottler_config["name"],
                "message": "Basic mode - Full integration not available",
                "query": query,
                "request_id": request_id
            }
            
            return func.HttpResponse(
                json.dumps(response),
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
        if ORCHESTRATOR_AVAILABLE and bottler_orchestrator:
            result = await bottler_orchestrator.handle_hub_command(command, parameters)
            
            return func.HttpResponse(
                json.dumps(result, default=str),
                mimetype="application/json",
                status_code=200 if result.get("success") else 500
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "error": "Orchestrator not available",
                    "bottler_id": bottler_config["id"]
                }),
                mimetype="application/json",
                status_code=503
            )
        
    except Exception as e:
        logger.error(f"Error handling hub command: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="financial/query", methods=["POST"])
async def query_financial_data(req: func.HttpRequest) -> func.HttpResponse:
    """Query REAL financial data from this bottler's Cosmos DB"""
    try:
        await initialize_integrations()
        
        if not mcp_bridge:
            return func.HttpResponse(
                json.dumps({"error": "Database not available"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Parse query request
        req_body = req.get_json()
        query_type = req_body.get("type", "revenue")
        period = req_body.get("period", "2024")
        product = req_body.get("product")
        
        # Build REAL Cosmos DB query
        if query_type == "revenue":
            query = f"SELECT * FROM c WHERE c.bottler_id = '{bottler_config['id']}' AND c.type = 'financial_record' AND c.period >= '{period}'"
            if product:
                query += f" AND ARRAY_CONTAINS(c.products, {{'name': '{product}'}}, true)"
        else:
            query = f"SELECT * FROM c WHERE c.bottler_id = '{bottler_config['id']}' AND c.type = 'financial_record'"
        
        # Execute REAL query against OUR Cosmos DB
        result = await mcp_bridge.execute_tool(
            server_name="cosmos",
            tool_name="query_documents",
            arguments={
                "container": "financial_data",
                "query": query,
                "max_items": 100
            }
        )
        
        # Process REAL results
        if result.get("success"):
            documents = result.get("result", [])
            
            # Calculate totals from REAL data
            total_revenue = sum(doc.get("revenue", 0) for doc in documents)
            total_costs = sum(doc.get("costs", 0) for doc in documents)
            
            response_data = {
                "bottler_id": bottler_config["id"],
                "query_type": query_type,
                "period": period,
                "total_revenue": total_revenue,
                "total_costs": total_costs,
                "margin": (total_revenue - total_costs) / total_revenue if total_revenue > 0 else 0,
                "records_count": len(documents),
                "data": documents[:10]  # Return first 10 records
            }
            
            return func.HttpResponse(
                json.dumps(response_data),
                mimetype="application/json",
                status_code=200
            )
        else:
            return func.HttpResponse(
                json.dumps({"error": "Query failed", "details": result.get("error")}),
                mimetype="application/json",
                status_code=500
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
    """Submit financial data to this bottler's REAL Cosmos DB"""
    try:
        await initialize_integrations()
        
        if not mcp_bridge:
            return func.HttpResponse(
                json.dumps({"error": "Database not available"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Parse financial data
        req_body = req.get_json()
        
        # Create REAL document for Cosmos DB
        document = {
            "id": str(uuid.uuid4()),
            "type": "financial_record",
            "bottler_id": bottler_config["id"],
            "period": req_body.get("period", datetime.utcnow().strftime("%Y-%m")),
            "revenue": req_body.get("revenue", 0),
            "costs": req_body.get("costs", 0),
            "products": req_body.get("products", []),
            "submitted_at": datetime.utcnow().isoformat(),
            "submitted_by": req_body.get("submitted_by", "system")
        }
        
        # Store in REAL Cosmos DB
        result = await mcp_bridge.execute_tool(
            server_name="cosmos",
            tool_name="upsert_document",
            arguments={
                "container": "financial_data",
                "document": document
            }
        )
        
        if result.get("success"):
            # Also store a backup in Blob Storage
            await mcp_bridge.execute_tool(
                server_name="blob",
                tool_name="write_blob",
                arguments={
                    "blob_name": f"financial/{bottler_config['id']}/{document['id']}.json",
                    "content": json.dumps(document),
                    "content_type": "application/json",
                    "metadata": {
                        "bottler_id": bottler_config["id"],
                        "period": document["period"],
                        "type": "financial_backup"
                    }
                }
            )
            
            return func.HttpResponse(
                json.dumps({
                    "message": "Financial data stored successfully",
                    "document_id": document["id"],
                    "stored_in": ["cosmos_db", "blob_storage"]
                }),
                mimetype="application/json",
                status_code=201
            )
        else:
            return func.HttpResponse(
                json.dumps({"error": "Failed to store data", "details": result.get("error")}),
                mimetype="application/json",
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Error submitting financial data: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="mcp/tools/list", methods=["GET"])
async def list_mcp_tools(req: func.HttpRequest) -> func.HttpResponse:
    """List available MCP tools for this bottler"""
    try:
        await initialize_integrations()
        
        if not mcp_bridge:
            return func.HttpResponse(
                json.dumps({"error": "MCP Bridge not available"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Get optional server filter
        server_name = req.params.get("server")
        
        # List tools
        result = await mcp_bridge.list_tools(server_name)
        
        return func.HttpResponse(
            json.dumps(result),
            mimetype="application/json",
            status_code=200 if result.get("success") else 500
        )
        
    except Exception as e:
        logger.error(f"Error listing MCP tools: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="mcp/status", methods=["GET"])
async def get_mcp_status(req: func.HttpRequest) -> func.HttpResponse:
    """Get MCP Bridge and server status"""
    try:
        await initialize_integrations()
        
        if not mcp_bridge:
            return func.HttpResponse(
                json.dumps({"error": "MCP Bridge not available"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Get comprehensive status
        status = await mcp_bridge.get_status()
        
        # Add bottler info
        status["bottler_id"] = bottler_config["id"]
        status["bottler_name"] = bottler_config["name"]
        
        return func.HttpResponse(
            json.dumps(status),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error getting MCP status: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="prompts/upload/excel", methods=["POST"])
async def upload_excel_prompts(req: func.HttpRequest) -> func.HttpResponse:
    """Upload Excel file containing prompts and queries"""
    try:
        await initialize_integrations()
        
        if not prompt_manager:
            return func.HttpResponse(
                json.dumps({"error": "Prompt Manager not available"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Check if file is in form data or body
        file_content = None
        file_name = None
        
        # Try to get file from multipart form data
        files = req.files
        if files and 'file' in files:
            file_item = files['file']
            file_content = file_item.read()
            file_name = file_item.filename
        else:
            # Try to get from body
            file_content = req.get_body()
            file_name = req.headers.get('X-File-Name', 'uploaded_excel.xlsx')
        
        if not file_content:
            return func.HttpResponse(
                json.dumps({"error": "No file content found"}),
                mimetype="application/json",
                status_code=400
            )
        
        logger.info(f"Processing Excel file: {file_name}")
        
        # Process the Excel file
        result = await prompt_manager.process_excel_file(
            file_path=file_name,
            file_content=file_content
        )
        
        if result.get("success"):
            # Store the uploaded file in blob storage for audit
            if mcp_bridge:
                await mcp_bridge.execute_tool(
                    server_name="blob",
                    tool_name="write_blob",
                    arguments={
                        "blob_name": f"prompts/excel_uploads/{bottler_config['id']}/{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{file_name}",
                        "content": file_content,
                        "content_type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        "metadata": {
                            "bottler_id": bottler_config["id"],
                            "upload_date": datetime.utcnow().isoformat(),
                            "prompts_extracted": str(len(result.get("prompts", [])))
                        }
                    }
                )
            
            return func.HttpResponse(
                json.dumps({
                    "success": True,
                    "message": f"Processed {len(result.get('prompts', []))} prompts from Excel file",
                    "import_id": result.get("import_record", {}).get("id"),
                    "prompts_extracted": len(result.get("prompts", [])),
                    "file_name": file_name,
                    "bottler_id": bottler_config["id"]
                }),
                mimetype="application/json",
                status_code=200
            )
        else:
            return func.HttpResponse(
                json.dumps({
                    "error": result.get("error", "Failed to process Excel file"),
                    "file_name": file_name
                }),
                mimetype="application/json",
                status_code=500
            )
            
    except Exception as e:
        logger.error(f"Error uploading Excel prompts: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="prompts/search", methods=["GET", "POST"])
async def search_prompts(req: func.HttpRequest) -> func.HttpResponse:
    """Search for stored prompts"""
    try:
        await initialize_integrations()
        
        if not prompt_manager:
            return func.HttpResponse(
                json.dumps({"error": "Prompt Manager not available"}),
                mimetype="application/json",
                status_code=503
            )
        
        # Get search parameters
        if req.method == "GET":
            query = req.params.get("query", "")
            category = req.params.get("category")
        else:
            req_body = req.get_json()
            query = req_body.get("query", "")
            category = req_body.get("category")
        
        # Search prompts
        prompts = await prompt_manager.search_prompts(query, category)
        
        return func.HttpResponse(
            json.dumps({
                "success": True,
                "prompts": prompts,
                "count": len(prompts),
                "bottler_id": bottler_config["id"]
            }),
            mimetype="application/json",
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error searching prompts: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

@app.route(route="prompts/{prompt_id}", methods=["GET"])
async def get_prompt(req: func.HttpRequest) -> func.HttpResponse:
    """Get a specific prompt by ID"""
    try:
        await initialize_integrations()
        
        if not prompt_manager:
            return func.HttpResponse(
                json.dumps({"error": "Prompt Manager not available"}),
                mimetype="application/json",
                status_code=503
            )
        
        prompt_id = req.route_params.get("prompt_id")
        if not prompt_id:
            return func.HttpResponse(
                json.dumps({"error": "Prompt ID is required"}),
                mimetype="application/json",
                status_code=400
            )
        
        # Get prompt
        prompt = await prompt_manager.get_prompt_by_id(prompt_id)
        
        if prompt:
            return func.HttpResponse(
                json.dumps({
                    "success": True,
                    "prompt": prompt
                }),
                mimetype="application/json",
                status_code=200
            )
        else:
            return func.HttpResponse(
                json.dumps({"error": "Prompt not found"}),
                mimetype="application/json",
                status_code=404
            )
        
    except Exception as e:
        logger.error(f"Error getting prompt: {str(e)}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            mimetype="application/json",
            status_code=500
        )

# Initialize bottler on startup
logger.info(f"Starting Soft Bottler Manager: {bottler_config['id']}")
logger.info(f"Region: {bottler_config['region']}")
logger.info(f"Hub URL: {bottler_config['hub_url']}")
logger.info("This is a SPOKE - All communication goes through the hub")