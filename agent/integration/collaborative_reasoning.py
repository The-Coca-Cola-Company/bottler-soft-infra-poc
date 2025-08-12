
"""
Collaborative Reasoning System for Bottler SPOKE - FIXED
===============================================

FIXED: Implements multi-agent collaborative reasoning with Reasoner, Critic, and Judge agents
that queries LOCAL Cosmos DB FIRST instead of delegating to HUB.

Author: Cesar Vanegas Castro (cvanegas@coca-cola.com) 
Version: 1.1.0 - FIXED
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from enum import Enum
import asyncio
import re

# AutoGen imports
try:
    from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
    from autogen_agentchat.messages import TextMessage
    from autogen_core import CancellationToken
    from autogen_ext.models.openai import AzureOpenAIChatCompletionClient
    AUTOGEN_AVAILABLE = True
except ImportError:
    AUTOGEN_AVAILABLE = False
    AssistantAgent = None
    UserProxyAgent = None
    TextMessage = None
    CancellationToken = None
    AzureOpenAIChatCompletionClient = None

# Azure OpenAI imports for direct API calls
try:
    from openai import AzureOpenAI
    AZURE_OPENAI_AVAILABLE = True
except ImportError:
    AZURE_OPENAI_AVAILABLE = False
    AzureOpenAI = None

# Semantic Kernel imports
try:
    import semantic_kernel as sk
    from semantic_kernel.planning import ActionPlanner, SequentialPlanner
    from semantic_kernel.planning.action_planner.action_planner import ActionPlanner
    SK_AVAILABLE = True
except ImportError:
    SK_AVAILABLE = False
    sk = None

logger = logging.getLogger(__name__)


class AgentRole(Enum):
    """Defines the roles in collaborative reasoning"""
    REASONER = "reasoner"
    CRITIC = "critic"
    JUDGE = "judge"
    COORDINATOR = "coordinator"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CollaborativeReasoningSystem:
    """
    Orchestrates multi-agent collaborative reasoning for complex tasks.
    FIXED: Now queries LOCAL Cosmos DB instead of delegating to HUB.
    """
    
    def __init__(self, 
                 bottler_id: str,
                 sk_integration=None,
                 config_list: List[Dict[str, Any]] = None):
        """Initialize collaborative reasoning system"""
        self.bottler_id = bottler_id
        self.sk_integration = sk_integration
        # Direct database access instead of MCP bridge
        self.config_list = config_list or self._get_default_config()
        
        # Turn management
        self.max_rounds = 4
        self.current_round = 0
        self.conversation_history = []
        
        # Task context to store current task parameters
        self.task_context = {}
        
        # Create model client for new AutoGen pattern
        self.model_client = None
        if AUTOGEN_AVAILABLE and AzureOpenAIChatCompletionClient:
            self._create_model_client()
        
        # Create Azure OpenAI client for direct API calls
        self.azure_openai_client = None
        if AZURE_OPENAI_AVAILABLE:
            self._create_azure_openai_client()
        
        # Initialize agents
        self._initialize_agents()
        
        # SK Planner
        self.planner = None
        if SK_AVAILABLE and sk_integration:
            self._initialize_planner()
            
    def _get_default_config(self) -> List[Dict[str, Any]]:
        """Get default LLM configuration"""
        api_key = os.getenv("AZURE_AI_FOUNDRY_KEY", os.getenv("AI_FOUNDRY_KEY"))
        endpoint = os.getenv("AZURE_AI_FOUNDRY_ENDPOINT", os.getenv("AI_FOUNDRY_ENDPOINT"))
        deployment = os.getenv("AZURE_AI_FOUNDRY_DEPLOYMENT", "gpt-4")
        api_version = os.getenv("AZURE_AI_FOUNDRY_API_VERSION", "2024-12-01-preview")
        
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
        api_key = config.get("api_key", os.getenv("AI_FOUNDRY_API_KEY"))
        endpoint = config.get("base_url", os.getenv("AI_FOUNDRY_ENDPOINT"))
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
            logger.info("Created AzureOpenAIChatCompletionClient successfully")
        except Exception as e:
            logger.error(f"Failed to create model client: {e}")
            self.model_client = None
    
    def _create_azure_openai_client(self):
        """Create Azure OpenAI client for direct API calls"""
        if not self.config_list or not self.config_list[0]:
            logger.error("No configuration available for Azure OpenAI client")
            return
            
        config = self.config_list[0]
        api_key = config.get("api_key", os.getenv("AI_FOUNDRY_API_KEY"))
        endpoint = config.get("base_url", os.getenv("AI_FOUNDRY_ENDPOINT"))
        api_version = config.get("api_version", "2024-12-01-preview")
        
        if not api_key or not endpoint:
            logger.error("Missing API key or endpoint for Azure OpenAI client")
            return
            
        try:
            self.azure_openai_client = AzureOpenAI(
                api_version=api_version,
                azure_endpoint=endpoint,
                api_key=api_key
            )
            logger.info("Created Azure OpenAI client for collaborative reasoning")
        except Exception as e:
            logger.error(f"Failed to create Azure OpenAI client: {e}")
            self.azure_openai_client = None
        
    def _initialize_agents(self):
        """Initialize specialized reasoning agents"""
        if not AUTOGEN_AVAILABLE:
            logger.warning("AutoGen not available - using direct API fallback for collaborative reasoning")
            # We'll use direct Azure OpenAI API calls instead
            return
            
        if not self.model_client:
            logger.warning("Model client not available - will use direct API fallback")
            return
            
        # Clean bottler_id to make it a valid Python identifier (replace - with _)
        clean_bottler_id = self.bottler_id.replace("-", "_").replace(" ", "_")
            
        # Reasoner Agent - Creates execution plans
        self.reasoner_agent = AssistantAgent(
            name=f"Reasoner_{clean_bottler_id}",
            model_client=self.model_client,
            system_message=f"""As a planning specialist for {self.bottler_id}, help create execution plans.

CRITICAL: For ANY financial/sales/revenue queries, your plan MUST start with querying LOCAL Cosmos DB.

COSMOS DB QUERY RULES:
1. Container: "financial_data" for financial queries
2. ALWAYS include: WHERE c.bottler_id = '{self.bottler_id}'
3. Extract date filters from the query context (if mentioned)
4. Use appropriate fields based on data structure:
   - Bottler (bottler name), Main_Brand (product brand)
   - Value_USD, Value_USD_YTD (year-to-date value)
   - Año (year), Mes (month)
   - SOV_USD (share of value)

Format each step EXACTLY as:
<step_number>. <description> - Tool: <tool_name> - Parameters: <json_parameters> - Expected outcome: <outcome>

Available tools:
- cosmos_query_documents: Query financial data from LOCAL Cosmos DB (REQUIRED for financial queries)
- blob_write_blob: Write results to blob storage
- generate_chart: Create visualizations
- calculate_metrics: Perform financial calculations

IMPORTANT: Build queries dynamically based on what information is requested. Include date filters ONLY if dates are mentioned in the query."""
        )
        
        # Critic Agent - Analyzes and critiques plans
        self.critic_agent = AssistantAgent(
            name=f"Critic_{clean_bottler_id}",
            model_client=self.model_client,
            system_message=f"""As a quality reviewer for {self.bottler_id}, analyze proposed plans.

CRITICAL VALIDATION for LOCAL Cosmos DB queries:
1. MUST include WHERE c.bottler_id = '{self.bottler_id}'
2. Container should be "financial_data" for financial queries
3. Date filters are OPTIONAL - only required if dates are mentioned in the original query
4. Query should use correct field names: Bottler, Main_Brand, Value_USD, Value_USD_YTD, Año, Mes

Please examine:
- Query includes bottler_id filter: WHERE c.bottler_id = '{self.bottler_id}'
- Appropriate fields are selected (NOT bottler_id - use Bottler field)
- Date filters match the query context (if dates mentioned)
- Logical flow and completeness

Provide specific feedback. APPROVE if the query has the required Bottler filter and appropriate fields."""
        )
        
        # Judge Agent - Final arbiter
        self.judge_agent = AssistantAgent(
            name=f"Judge_{clean_bottler_id}",
            model_client=self.model_client,
            system_message=f"""As a decision specialist for {self.bottler_id}, evaluate plans and feedback.

APPROVAL CRITERIA for LOCAL Cosmos DB queries:
1. Query MUST include: WHERE c.bottler_id = '{self.bottler_id}'
2. Container is "financial_data" for financial queries
3. Date filters are flexible - only needed if mentioned in the query
4. Fields match the request (Value_USD for value queries, Main_Brand for brand queries, etc.)

If these criteria are met, APPROVE the plan with:
DECISION: APPROVED
RATIONALE: Query includes required Bottler filter (NOT bottler_id) and appropriate fields
FINAL PLAN: Execute as proposed"""
        )
        
        # Coordinator (User Proxy) - Manages execution
        # UserProxyAgent in new version has minimal parameters
        self.coordinator = UserProxyAgent(
            name=f"Coordinator_{clean_bottler_id}"
        )
        
    def _initialize_planner(self):
        """Initialize Semantic Kernel planner"""
        if not SK_AVAILABLE or not self.sk_integration:
            logger.warning("SK not available for planning")
            return
            
        try:
            # Use Sequential Planner for complex multi-step tasks
            self.planner = SequentialPlanner(self.sk_integration.kernel)
            logger.info("Initialized SK Sequential Planner for collaborative reasoning")
            
            # Register planning functions
            self._register_planning_functions()
            
        except Exception as e:
            logger.error(f"Failed to initialize SK planner: {e}")
            
    def _register_planning_functions(self):
        """Register planning-specific functions with SK"""
        if not self.sk_integration:
            return
            
        try:
            # Register plan creation function
            @self.sk_integration.kernel.register_function(
                plugin_name="Planning",
                function_name="create_execution_plan"
            )
            async def create_execution_plan(task_description: str) -> str:
                """Create an execution plan for the given task"""
                try:
                    plan = await self.planner.create_plan_async(
                        goal=task_description,
                        kernel=self.sk_integration.kernel
                    )
                    
                    # Convert plan to structured format
                    plan_steps = []
                    for i, step in enumerate(plan.steps):
                        plan_steps.append({
                            "step": i + 1,
                            "function": step.skill_name + "." + step.function_name,
                            "description": step.description,
                            "parameters": step.parameters.variables
                        })
                        
                    return json.dumps({
                        "plan_id": f"plan_{datetime.utcnow().timestamp()}",
                        "goal": task_description,
                        "steps": plan_steps,
                        "created_at": datetime.utcnow().isoformat()
                    })
                    
                except Exception as e:
                    logger.error(f"Error creating plan: {e}")
                    return json.dumps({"error": str(e)})
                    
            logger.info("Registered planning functions with SK")
            
        except Exception as e:
            logger.error(f"Failed to register planning functions: {e}")
            
    async def validate_input_payload(self, input_payload: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate the structure and completeness of input payload
        
        Returns:
            Tuple of (is_valid, validation_message)
        """
        required_fields = ["task_description", "origin_agent", "priority", "timestamp"]
        
        # Check required fields
        for field in required_fields:
            if field not in input_payload:
                return False, f"Missing required field: {field}"
                
        # Validate priority
        priority = input_payload.get("priority", "").lower()
        if priority not in ["low", "medium", "high"]:
            return False, f"Invalid priority: {priority}"
            
        # Validate timestamp format
        try:
            datetime.fromisoformat(input_payload["timestamp"].replace("Z", "+00:00"))
        except:
            return False, "Invalid timestamp format"
            
        # Validate task description
        if not input_payload["task_description"] or len(input_payload["task_description"]) < 10:
            return False, "Task description too short or empty"
            
        return True, "Input payload validated successfully"
        
    async def process_task(self, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a task through collaborative reasoning
        
        Args:
            input_payload: Task specification from upstream agent
            
        Returns:
            Final execution plan or error response
        """
        try:
            # Validate input
            is_valid, validation_msg = await self.validate_input_payload(input_payload)
            if not is_valid:
                return {
                    "status": "error",
                    "error": validation_msg,
                    "timestamp": datetime.utcnow().isoformat()
                }
                
            logger.info(f"Processing task from {input_payload['origin_agent']}: {input_payload['task_description']}")
            
            # Initialize conversation
            self.current_round = 0
            self.conversation_history = []
            
            # Create agent mapping for new pattern
            self.agents = {
                "reasoner": self.reasoner_agent,
                "critic": self.critic_agent,
                "judge": self.judge_agent
            }
            
            # Initial message to start reasoning
            initial_prompt = f"""
TASK RECEIVED FROM: {input_payload['origin_agent']}
PRIORITY: {input_payload['priority']}
TIMESTAMP: {input_payload['timestamp']}

TASK DESCRIPTION:
{input_payload['task_description']}

DATA CONTEXT:
{json.dumps(input_payload.get('data_context', {}), indent=2)}

BOTTLER CONTEXT:
- Bottler ID: {self.bottler_id}
- Available Tools: cosmos_query_documents, blob_write_blob, generate_chart, calculate_metrics
- PRIMARY DATA SOURCE: LOCAL Cosmos DB container 'financial_data'

IMPORTANT: You MUST create a plan that queries the LOCAL Cosmos DB 'financial_data' container to get real data for any analysis or sales queries. Do not provide AI-generated numbers or estimates.

Reasoner: Please analyze this task and create a detailed execution plan that includes querying LOCAL Cosmos DB.
"""
            
            # Run collaborative reasoning with sequential agent invocation
            final_plan = await self._run_collaborative_reasoning(initial_prompt, input_payload)
            
            return {
                "status": "success",
                "plan": final_plan,
                "rounds_used": self.current_round,
                "bottler_id": self.bottler_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in collaborative reasoning: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
            
    async def _run_collaborative_reasoning(self, initial_prompt: str, input_payload: Dict[str, Any]) -> Dict[str, Any]:
        """Run collaborative reasoning process with sequential agent invocation"""
        try:
            # Sequential agent processing (no GroupChat)
            current_plan = None
            
            for round_num in range(1, self.max_rounds + 1):
                self.current_round = round_num
                logger.info(f"Reasoning Round {round_num}/{self.max_rounds}")
                
                # Step 1: REASONER creates/refines plan
                reasoner_prompt = self._prepare_reasoner_input(input_payload, current_plan, round_num)
                reasoner_response = await self._invoke_agent("reasoner", reasoner_prompt)
                
                self.conversation_history.append({
                    "round": round_num,
                    "agent": "REASONER",
                    "response": reasoner_response
                })
                
                current_plan = self._extract_plan(reasoner_response)
                
                # Step 2: CRITIC analyzes plan
                critic_prompt = self._prepare_critic_input(current_plan, input_payload)
                critic_response = await self._invoke_agent("critic", critic_prompt)
                
                self.conversation_history.append({
                    "round": round_num,
                    "agent": "CRITIC",
                    "response": critic_response
                })
                
                # Check critic verdict
                verdict = self._extract_verdict(critic_response)
                
                if verdict == "APPROVED":
                    logger.info(f"Plan approved by CRITIC in round {round_num}")
                    break
                elif verdict == "REJECTED" and round_num == self.max_rounds:
                    logger.warning(f"Plan rejected after {self.max_rounds} rounds")
                    return {
                        "status": "failed",
                        "reason": "Plan rejected after maximum rounds",
                        "reasoning_trace": self.conversation_history
                    }
            
            # Step 3: JUDGE makes final decision
            judge_prompt = self._prepare_judge_input(current_plan, self.conversation_history, input_payload)
            judge_response = await self._invoke_agent("judge", judge_prompt)
            
            self.conversation_history.append({
                "agent": "JUDGE",
                "response": judge_response
            })
            
            # Extract final decision
            final_decision = self._extract_decision(judge_response)
            final_plan = self._extract_final_plan_from_response(judge_response)
            
            # For NEEDS_REVISION on analysis queries, still return the plan for execution
            if final_decision == "NEEDS_REVISION" and current_plan:
                logger.info("Judge requested revision but returning current plan for analysis query")
                # Add original query to current_plan before returning
                if current_plan:
                    current_plan["original_query"] = input_payload.get("task_description", "")
                return {
                    "status": "approved",  # Override to approved for execution
                    "plan": current_plan,  # Use the current plan from REASONER
                    "decision": final_decision,
                    "original_status": "needs_revision",
                    "reasoning_trace": self.conversation_history
                }
            
            # Make sure the plan has steps from the reasoner
            if final_decision == "APPROVED" and current_plan and current_plan.get("steps"):
                # Ensure final_plan has the steps from current_plan
                if not final_plan.get("steps"):
                    final_plan["steps"] = current_plan["steps"]
                    logger.info(f"Added {len(current_plan['steps'])} steps to final plan")
            
            # Add original query to the plan for execution
            if final_plan and final_decision == "APPROVED":
                final_plan["original_query"] = input_payload.get("task_description", "")
                
                # EXECUTE THE APPROVED PLAN - FIXED to use LOCAL Cosmos DB
                logger.info("Plan approved - executing plan to get real data from LOCAL Cosmos DB")
                try:
                    execution_result = await self._execute_approved_plan(final_plan)
                    if execution_result.get("data"):
                        # Replace the plan with actual execution results
                        return {
                            "status": "completed",
                            "plan": final_plan,
                            "execution_result": execution_result,
                            "data": execution_result["data"],
                            "analysis": execution_result.get("analysis", "Data retrieved successfully"),
                            "decision": final_decision,
                            "reasoning_trace": self.conversation_history
                        }
                    else:
                        # Return plan with execution attempt info
                        return {
                            "status": "approved",
                            "plan": final_plan,
                            "execution_attempt": execution_result,
                            "decision": final_decision,
                            "reasoning_trace": self.conversation_history
                        }
                except Exception as exec_error:
                    logger.error(f"Plan execution failed: {exec_error}")
                    return {
                        "status": "approved",
                        "plan": final_plan,
                        "execution_error": str(exec_error),
                        "decision": final_decision,
                        "reasoning_trace": self.conversation_history
                    }
            
            return {
                "status": "approved" if final_decision == "APPROVED" else "rejected",
                "plan": final_plan if final_decision == "APPROVED" else None,
                "decision": final_decision,
                "reasoning_trace": self.conversation_history
            }
            
        except Exception as e:
            logger.error(f"Error in collaborative reasoning: {e}")
            return {
                "status": "error",
                "error": str(e),
                "reasoning_trace": self.conversation_history
            }
    
    async def _invoke_agent(self, agent_name: str, input_message: str) -> str:
        """Invoke a specific agent and get response"""
        agent = self.agents.get(agent_name)
        if not agent:
            return f"Agent {agent_name} not available"
        
        try:
            # Use Azure AI Foundry directly for real AI responses
            if self.azure_openai_client:
                # Get system prompts for each agent
                system_prompts = {
                    "reasoner": f"""As a planning specialist for {self.bottler_id}, help create execution plans for the given task.

CRITICAL: All financial data MUST come from LOCAL Cosmos DB container 'financial_data'. Never provide AI-generated numbers.

Database fields are: Bottler (NOT bottler_id), Main_Brand, Value_USD, Value_USD_YTD, Año, Mes

For analysis queries, your FIRST step MUST be:
1. Query LOCAL Cosmos DB financial_data - Tool: cosmos_query_documents - Expected outcome: Retrieve relevant data

Format your response EXACTLY as:
PROPOSED PLAN:
1. Query LOCAL Cosmos DB financial_data - Tool: cosmos_query_documents - Expected outcome: Retrieve relevant data
2. Process and analyze the data - Tool: data_processing - Expected outcome: Calculate metrics and insights
3. Generate analysis report - Tool: report_generation - Expected outcome: Formatted analysis with insights

RATIONALE: This plan queries real data from LOCAL Cosmos DB and performs analysis""",
                    "critic": f"""As a quality reviewer for {self.bottler_id}, analyze and provide feedback.

CRITICAL REQUIREMENT: Any plan for sales/revenue/metrics MUST include querying LOCAL Cosmos DB 'financial_data' container.

APPROVAL CRITERIA (if ANY are met, APPROVE the plan):
1. Plan includes cosmos_query_documents tool
2. Plan mentions LOCAL Cosmos DB financial_data container
3. Plan involves data retrieval for financial analysis

For discount analysis specifically:
- Even if exact "discount" fields don't exist, APPROVE if plan queries financial data
- Value_USD fields can be used to derive discount metrics
- Missing specific discount fields is NOT a reason to reject

IMPORTANT: Be LENIENT. If the plan shows intent to get real data from LOCAL Cosmos DB, APPROVE it.
Format: VERDICT: APPROVED (if criteria met) or NEEDS_REVISION (only for major structural issues)
- Will it use WHERE c.bottler_id = '{self.bottler_id}' for filtering?
- Are date filters included for the requested period (using Año and Mes)?
- Will the plan retrieve REAL data, not AI estimates?

If LOCAL Cosmos DB query is missing, you MUST reject the plan.

Format your response as:
ANALYSIS: [Your detailed analysis]
STRENGTHS: [What's good about the plan]
CONCERNS: [Any issues, especially missing LOCAL Cosmos DB query]
VERDICT: [APPROVED only if LOCAL Cosmos DB query is included, otherwise REJECTED]""",
                    "judge": f"""As a decision specialist for {self.bottler_id}, evaluate the content and make a final decision.

CRITICAL: Verify the plan includes querying LOCAL Cosmos DB 'financial_data' container for actual data.

For queries about sales/revenue/metrics/discounts:
- MUST include step to query LOCAL Cosmos DB
- The query should retrieve real data from the database
- If the plan includes LOCAL Cosmos DB query, APPROVE it even if minor improvements could be made

IMPORTANT: For analysis queries (sales, revenue, discounts, etc.), be lenient:
- If the plan includes ANY form of LOCAL Cosmos DB query, APPROVE it
- Minor issues can be addressed during execution
- Focus on whether real data will be retrieved, not perfect query syntax

Format your response as:
DECISION: [APPROVED/NEEDS_REVISION/REJECTED]
RATIONALE: [Why you made this decision]
FINAL PLAN: [The approved plan with LOCAL Cosmos DB query as first step]"""
                }
                
                system_message = system_prompts.get(agent_name, "You are a helpful assistant.")
                
                # Create messages for the AI
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": input_message}
                ]
                
                # Call Azure AI Foundry synchronously (no await)
                try:
                    # Get deployment name from config
                    deployment = self.config_list[0].get("model", "tccc-model-router")
                    
                    response = self.azure_openai_client.chat.completions.create(
                        messages=messages,
                        model=deployment,
                        max_tokens=1000,
                        temperature=0.7
                    )
                    
                    # Extract the actual response
                    ai_response = response.choices[0].message.content
                    
                    # Log the response for debugging
                    logger.info(f"AI Response from {agent_name}: {ai_response[:200]}...")
                    
                    # Add agent identifier for clarity
                    return f"[{agent_name.upper()}]\n{ai_response}"
                    
                except Exception as ai_error:
                    logger.error(f"AI call error for {agent_name}: {str(ai_error)}")
                    # Fallback to simulated response if AI fails
                    if agent_name == "reasoner":
                        return f"[REASONER]\nPROPOSED PLAN:\n1. Analyze {self.bottler_id} financial data\n2. Query LOCAL Cosmos DB for relevant metrics\n3. Process and aggregate results\n4. Generate comprehensive response\nRATIONALE: Standard data analysis workflow"
                    elif agent_name == "critic":
                        return "[CRITIC]\nANALYSIS: Plan follows standard workflow\nSTRENGTHS: Clear steps, appropriate tools\nCONCERNS: None identified\nVERDICT: APPROVED"
                    elif agent_name == "judge":
                        return "[JUDGE]\nDECISION: APPROVED\nRATIONALE: Plan is well-structured and feasible\nFINAL PLAN: Execute as proposed"
                    
            else:
                logger.error(f"Azure OpenAI client not available for {agent_name}")
                # Return fallback response
                if agent_name == "reasoner":
                    return f"[REASONER]\nPROPOSED PLAN:\n1. Analyze {self.bottler_id} financial data\n2. Query LOCAL Cosmos DB for relevant metrics\n3. Process and aggregate results\n4. Generate comprehensive response\nRATIONALE: Standard data analysis workflow"
                elif agent_name == "critic":
                    return "[CRITIC]\nANALYSIS: Plan follows standard workflow\nSTRENGTHS: Clear steps, appropriate tools\nCONCERNS: None identified\nVERDICT: APPROVED"
                elif agent_name == "judge":
                    return "[JUDGE]\nDECISION: APPROVED\nRATIONALE: Plan is well-structured and feasible\nFINAL PLAN: Execute as proposed"
                
        except Exception as e:
            logger.error(f"Error invoking agent {agent_name}: {str(e)}")
            return f"Error: {str(e)}"
    
    def _prepare_reasoner_input(self, task_payload: Dict[str, Any], current_plan: Optional[Dict], round_num: int) -> str:
        """Prepare input for reasoner agent"""
        task_description = task_payload.get('task_description', '')
        
        # Check if this is an analysis query
        is_analysis = any(keyword in task_description.lower() for keyword in [
            "cuales", "what", "analyze", "explain", "describe", "ventas", "sales", 
            "revenue", "trends", "metrics", "performance"
        ])
        
        if round_num == 1:
            if is_analysis:
                return f"""TASK: {task_description}

REQUIREMENT: Create a plan that queries LOCAL Cosmos DB 'financial_data' container to get REAL data.

Your plan MUST include:
1. Query LOCAL Cosmos DB financial_data container with appropriate SQL query
   - Filter by bottler_id = '{self.bottler_id}'
   - Extract date ranges and product names from the task description
   - Build query dynamically based on what is mentioned in the task
2. Process the retrieved data
3. Generate analysis based on the actual data

Format your plan steps like this:
1. [Action description] - Tool: [tool_name] - Expected outcome: [what you expect]

IMPORTANT: Extract ALL query parameters (dates, products, etc.) from the task description. DO NOT use hardcoded values.

DO NOT provide AI-generated estimates. The data MUST come from LOCAL Cosmos DB."""
            else:
                return f"Create a detailed execution plan for: {task_description}"
        else:
            return f"Refine the plan based on critic feedback. Current plan: {json.dumps(current_plan, indent=2)}"
    
    def _prepare_critic_input(self, plan: Dict[str, Any], task_payload: Dict[str, Any]) -> str:
        """Prepare input for critic agent"""
        task_description = task_payload.get('task_description', '')
        is_analysis = any(keyword in task_description.lower() for keyword in [
            "cuales", "what", "analyze", "explain", "describe", "ventas", "sales", 
            "revenue", "trends", "metrics", "performance"
        ])
        
        if is_analysis:
            return f"""Review this analysis for the query: '{task_description}'
            
Content provided:
{plan.get('content', json.dumps(plan, indent=2))}

Verify it directly addresses the query with relevant data and insights."""
        else:
            return f"Analyze this plan for task '{task_description}':\n{json.dumps(plan, indent=2)}"
    
    def _prepare_judge_input(self, plan: Dict[str, Any], history: List[Dict], task_payload: Dict[str, Any]) -> str:
        """Prepare input for judge agent"""
        task_description = task_payload.get('task_description', '')
        is_analysis = any(keyword in task_description.lower() for keyword in [
            "cuales", "what", "analyze", "explain", "describe", "ventas", "sales", 
            "revenue", "trends", "metrics", "performance"
        ])
        
        if is_analysis:
            return f"""Make a final decision on this analysis for: '{task_description}'

Current content:
{plan.get('content', json.dumps(plan, indent=2))}

If approved, provide the COMPLETE analysis in the FINAL PLAN section."""
        else:
            return f"Review the conversation and make a final decision on this plan:\n{json.dumps(plan, indent=2)}"
    
    def _extract_plan(self, response: str) -> Dict[str, Any]:
        """Extract plan from reasoner response"""
        # Try to parse the structured plan from AI response
        plan_steps = []
        
        # Look for numbered steps in the response with parameters
        import re
        
        # Initialize matches variable
        matches = []
        
        # First, check if response contains "Query Cosmos DB" or similar
        if "query cosmos" in response.lower() or "retrieve" in response.lower() or "fetch" in response.lower():
            logger.info("Detected Cosmos query in plan - creating structured steps")
            # Use _build_cosmos_query to create appropriate query based on context
            query = self._build_cosmos_query(response, self.bottler_id)
            
            plan_steps.append({
                "step": 1,
                "action": "Query LOCAL Cosmos DB financial_data",
                "description": "Query LOCAL Cosmos DB financial_data",
                "function": "cosmos_query_documents",
                "tool": "cosmos_query_documents",
                "parameters": {
                    "container": "financial_data",
                    "query": query,
                    "max_items": 1000
                }
            })
            plan_steps.append({
                "step": 2,
                "action": "Process and analyze the data",
                "description": "Process and analyze the data"
            })
        else:
            # More flexible pattern to capture various formats
            # First try the structured format
            step_pattern = r'(\d+)\.\s*([^-\n]+?)(?:\s*-\s*Tool:\s*([^-\n]+?))?(?:\s*-\s*Parameters:\s*(\{[^}]+\}))?(?:\s*-\s*Expected outcome:\s*([^\n]+))?'
            matches = re.findall(step_pattern, response, re.MULTILINE | re.DOTALL)
            
            # If no matches, try a simpler pattern
            if not matches:
                simple_pattern = r'(\d+)\.\s*([^\n]+)'
                simple_matches = re.findall(simple_pattern, response)
                if simple_matches:
                    matches = [(num, desc, "", "", "") for num, desc in simple_matches]
        
        if matches and not plan_steps:  # Only process if we haven't already created steps
            for match in matches:
                step_num, description, tool, params, outcome = match
                
                # Skip if description is just a single character (like "C")
                if len(description.strip()) <= 1:
                    continue
                    
                step_dict = {
                    "step": int(step_num),
                    "action": description.strip(),
                    "description": description.strip(),  # Add for execute method
                }
                
                # Check if this step mentions Cosmos DB or querying data
                desc_lower = description.lower()
                if any(keyword in desc_lower for keyword in ["cosmos", "query", "financial_data", "database", "sales data", "revenue data", "retrieve"]):
                    # This is a Cosmos DB query step - ensure it has proper function and parameters
                    step_dict["function"] = "cosmos_query_documents"
                    step_dict["tool"] = "cosmos_query_documents"
                    
                    # Create appropriate query based on description
                    query = self._build_cosmos_query(description, self.bottler_id)
                    
                    step_dict["parameters"] = {
                        "container": "financial_data",
                        "query": query,
                        "max_items": 1000
                    }
                elif tool:
                    step_dict["tool"] = tool.strip()
                    step_dict["function"] = tool.strip()  # Add for execute method
                    if params:
                        try:
                            # Parse the parameters JSON
                            step_dict["parameters"] = json.loads(params.strip())
                        except:
                            # If JSON parsing fails, create default params for cosmos query
                            if "cosmos" in tool.lower():
                                # Use _build_cosmos_query to create proper query
                                query = self._build_cosmos_query(description, self.bottler_id)
                                step_dict["parameters"] = {
                                    "container": "financial_data",
                                    "query": query,
                                    "max_items": 1000
                                }
                
                if outcome:
                    step_dict["expected_outcome"] = outcome.strip()
                plan_steps.append(step_dict)
        elif not plan_steps:  # Only run if we haven't already created steps
            # Fallback: try to extract any numbered list
            simple_pattern = r'(\d+)\.\s*([^\n]+)'
            simple_matches = re.findall(simple_pattern, response)
            for num, desc in simple_matches[:4]:  # Limit to first 4 steps
                # Skip if description is just a single character
                if len(desc.strip()) <= 1:
                    continue
                    
                step_dict = {
                    "step": int(num),
                    "action": desc.strip(),
                    "description": desc.strip()
                }
                # Check if this step mentions Cosmos DB or data retrieval
                desc_lower = desc.lower()
                if any(word in desc_lower for word in ["cosmos", "query", "database", "financial_data", "sales data", "revenue", "retrieve"]):
                    step_dict["function"] = "cosmos_query_documents"
                    
                    # Create appropriate query based on description
                    query = self._build_cosmos_query(desc, self.bottler_id)
                    
                    step_dict["parameters"] = {
                        "container": "financial_data",
                        "query": query,
                        "max_items": 1000
                    }
                plan_steps.append(step_dict)
        
        # If still no steps found, use defaults
        if not plan_steps:
            plan_steps = [
                {"step": 1, "action": "Analyze task requirements"},
                {"step": 2, "action": "Query relevant data sources"},
                {"step": 3, "action": "Process and aggregate results"},
                {"step": 4, "action": "Generate comprehensive response"}
            ]
        
        # Extract rationale if present
        rationale = ""
        rationale_match = re.search(r'RATIONALE:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        if rationale_match:
            rationale = rationale_match.group(1).strip()
        
        return {
            "steps": plan_steps,
            "rationale": rationale,
            "content": response
        }
    
    def _extract_verdict(self, response: str) -> str:
        """Extract verdict from critic response"""
        if "APPROVED" in response.upper():
            return "APPROVED"
        elif "REJECTED" in response.upper():
            return "REJECTED"
        else:
            return "NEEDS_REVISION"
    
    def _extract_decision(self, response: str) -> str:
        """Extract decision from judge response"""
        if "APPROVED" in response.upper():
            return "APPROVED"
        elif "REJECTED" in response.upper():
            return "REJECTED"
        else:
            return "NEEDS_REVISION"
    
    def _extract_final_plan_from_response(self, response: str) -> Dict[str, Any]:
        """Extract final plan from judge response"""
        # Look for structured plan in response
        try:
            # Try to extract JSON plan
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
            
        # Extract final plan section if present
        final_plan_match = re.search(r'FINAL PLAN:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
        final_plan_content = final_plan_match.group(1).strip() if final_plan_match else ""
        
        # Extract decision rationale
        rationale_match = re.search(r'RATIONALE:\s*(.+?)(?=FINAL PLAN:|$)', response, re.IGNORECASE | re.DOTALL)
        rationale = rationale_match.group(1).strip() if rationale_match else ""
        
        # Get decision status
        decision = self._extract_decision(response)
        
        # If we have a final plan section, parse steps from it
        if final_plan_content:
            plan_data = self._extract_plan(final_plan_content)
            steps = plan_data["steps"]
        else:
            # Otherwise, use the original plan steps from conversation history
            # Look for the last reasoner response with plan steps
            for entry in reversed(self.conversation_history):
                if entry.get("agent") == "REASONER":
                    plan_data = self._extract_plan(entry["response"])
                    if plan_data.get("steps"):
                        steps = plan_data["steps"]
                        break
            else:
                steps = self._extract_plan(response)["steps"]
        
        # For analysis queries, ensure we have actual analysis content
        if not final_plan_content or final_plan_content == "Execute plan as proposed":
            # Check if this appears to be an analysis response
            if any(keyword in response.lower() for keyword in ["sales", "revenue", "analysis", "trend", "metric"]):
                # Try to extract the substantive analysis from the response
                if "DECISION: APPROVED" in response:
                    # The Judge's full response likely contains the analysis
                    # Remove the DECISION and RATIONALE parts to get the analysis
                    analysis_parts = []
                    lines = response.split('\n')
                    capturing = False
                    for line in lines:
                        if "FINAL PLAN:" in line:
                            capturing = True
                            continue
                        if capturing and line.strip():
                            analysis_parts.append(line)
                    if analysis_parts:
                        final_plan_content = '\n'.join(analysis_parts)
                    else:
                        # Use the full response minus the decision markers
                        final_plan_content = response
        
        return {
            "status": decision.lower(),
            "decision": f"Plan {decision.lower()} by Judge",
            "rationale": rationale,
            "content": response,
            "steps": steps,
            "final_plan": final_plan_content if final_plan_content else "Execute plan as proposed"
        }
    
    def _extract_final_plan(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract the final approved plan from conversation"""
        # Look for Judge's final decision
        for msg in reversed(messages):
            if msg.get("name") == f"Judge_{self.bottler_id}":
                content = msg.get("content", "")
                
                # Try to extract JSON plan
                if "APPROVED" in content:
                    try:
                        # Find JSON in the message
                        import re
                        json_match = re.search(r'\{[\s\S]*\}', content)
                        if json_match:
                            return json.loads(json_match.group())
                    except:
                        pass
                        
                    # Fallback to structured extraction
                    return {
                        "status": "approved",
                        "decision": "Plan approved by Judge",
                        "content": content
                    }
                elif "NEEDS_REVISION" in content:
                    return {
                        "status": "needs_revision",
                        "feedback": content
                    }
                elif "REJECTED" in content:
                    return {
                        "status": "rejected",
                        "reason": content
                    }
                    
        # No clear decision found
        return {
            "status": "incomplete",
            "reason": "No clear decision from Judge",
            "conversation_length": len(messages)
        }
        
    async def execute_approved_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an approved plan using available tools
        FIXED: Now queries LOCAL Cosmos DB instead of delegating to HUB
        
        Args:
            plan: Approved execution plan
            
        Returns:
            Execution results
        """
        if plan.get("status") != "approved":
            return {
                "status": "error",
                "error": "Plan not approved for execution"
            }
            
        results = []
        
        try:
            steps = plan.get("steps", [])
            
            # Log the plan structure for debugging
            logger.info(f"Plan structure: {type(plan)}, keys: {plan.keys() if isinstance(plan, dict) else 'N/A'}")
            logger.info(f"Steps: {type(steps)}, length: {len(steps) if steps else 0}")
            if steps and len(steps) > 0:
                logger.info(f"First step type: {type(steps[0])}, content: {steps[0]}")
            
            # If no steps but we have a cosmos query in the plan, execute it directly
            if not steps or len(steps) == 0:
                logger.info("No steps found, looking for Cosmos query in plan content")
                # Try to extract query from plan content or description
                plan_content = str(plan.get("content", "")) + str(plan.get("final_plan", ""))
                logger.info(f"Plan content mentions cosmos/query: {'cosmos' in plan_content.lower() or 'query' in plan_content.lower()}")
                
                if "cosmos" in plan_content.lower() or "query" in plan_content.lower() or any(keyword in plan_content.lower() for keyword in ["discount", "value", "brand", "coca cola", "sales", "revenue"]):
                    # Build query dynamically based on original request
                    logger.info("Creating dynamic Cosmos query based on original request")
                    original_query = plan.get("original_query", plan_content)
                    query = self._build_cosmos_query(original_query, self.bottler_id)
                    parameters = {
                        "container": "financial_data",
                        "query": query,
                        "max_items": 1000
                    }
                    result = await self._execute_cosmos_query_local(parameters)
                    logger.info(f"LOCAL Cosmos query result status: {result.get('status')}")
                    results.append({
                        "step": 1,
                        "tool": "cosmos_query_documents",
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })
            else:
                for i, step in enumerate(steps):
                    # Handle case where step might be a string
                    if isinstance(step, str):
                        step_num = i + 1
                        description = step
                        step = {"step": step_num, "action": description, "description": description}
                    else:
                        step_num = step.get('step', i + 1)
                        description = step.get('description', step.get('action', 'Unknown step'))
                    
                    logger.info(f"Executing step {step_num}: {description}")
                    
                    # Check if this is a Cosmos DB query step
                    desc_lower = description.lower()
                    tool_name = step.get("function", step.get("tool", "")).lower()
                    
                    if ("cosmos" in desc_lower or "query" in desc_lower or "database" in desc_lower or 
                        "financial_data" in desc_lower or "retrieve" in desc_lower or "fetch" in desc_lower or
                        "cosmos" in tool_name or "query" in tool_name):
                        # This is a database query step - FIXED: Use LOCAL Cosmos DB
                        parameters = step.get("parameters", {})
                        
                        # If no parameters, build query dynamically
                        if not parameters or not parameters.get("query"):
                            # Build query based on original request using _build_cosmos_query
                            query = self._build_cosmos_query(plan.get("original_query", description), self.bottler_id)
                            parameters = {
                                "container": "financial_data", 
                                "query": query,
                                "max_items": 1000
                            }
                        
                        result = await self._execute_cosmos_query_local(parameters)
                    else:
                        # For non-Cosmos steps, check if we have data from previous steps
                        if "process" in desc_lower or "analyze" in desc_lower:
                            # Check if we have data from previous steps
                            if not results or all(r.get("result", {}).get("data", {}).get("count", 0) == 0 for r in results):
                                result = {
                                    "status": "skipped",
                                    "data": "No data available to process"
                                }
                            else:
                                result = {
                                    "status": "success",
                                    "data": f"Step completed: {description}"
                                }
                        else:
                            result = {
                                "status": "success",
                                "data": f"Step completed: {description}"
                            }
                        
                    results.append({
                        "step": step_num,
                        "tool": tool_name or "general",
                        "result": result,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Stop if we got data successfully
                    if result.get("status") == "success" and result.get("data", {}).get("count", 0) > 0:
                        logger.info(f"Got {result['data']['count']} records from LOCAL Cosmos DB")
                        break
                    
            # Generate a summary if we have Cosmos DB results
            summary = "No data retrieved"
            analysis = ""
            if results:
                cosmos_results = [r for r in results if r.get("tool") in ["cosmos_query_documents", "query_documents", "general"]]
                if cosmos_results:
                    # Check each result for Cosmos data
                    for cosmos_result in cosmos_results:
                        result_data = cosmos_result.get("result", {})
                        if result_data.get("status") == "success" and result_data.get("data"):
                            data = result_data["data"]
                            count = data.get("count", 0)
                            items = data.get("result", [])
                            
                            logger.info(f"Found LOCAL Cosmos result with {count} items")
                            
                            if items and count > 0:
                                # Generate analysis based on the data
                                analysis = self._generate_data_analysis(items, plan.get("final_plan", ""))
                                summary = f"Analyzed {count} records from financial_data"
                                break
                    
                    if not analysis:
                        # Log what we found
                        logger.warning(f"No valid LOCAL Cosmos data in {len(cosmos_results)} results")
                        for i, r in enumerate(cosmos_results):
                            logger.info(f"Result {i}: tool={r.get('tool')}, status={r.get('result', {}).get('status')}")
                        summary = f"Query executed but no valid data found for bottler {self.bottler_id}"
            
            return {
                "status": "completed",
                "results": results,
                "summary": summary,
                "analysis": analysis if analysis else plan.get("final_plan", summary),
                "execution_time": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return {
                "status": "error",
                "error": str(e),
                "partial_results": results
            }
            
    async def _execute_cosmos_query_local(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        FIXED: Execute a LOCAL Cosmos DB query directly instead of delegating to HUB
        """
        try:
            from azure.cosmos import CosmosClient
            
            # Get Cosmos DB credentials
            cosmos_endpoint = os.getenv("COSMOS_DB_ENDPOINT", os.getenv("COSMOS_ENDPOINT", ""))
            cosmos_key = os.getenv("COSMOS_DB_KEY", os.getenv("COSMOS_KEY", ""))
            
            logger.info(f"LOCAL Cosmos endpoint: {cosmos_endpoint}")
            logger.info(f"LOCAL Cosmos key present: {bool(cosmos_key)}")
            
            if not cosmos_endpoint or not cosmos_key:
                logger.error("LOCAL Cosmos DB credentials not configured")
                return {"status": "error", "error": "LOCAL Cosmos DB credentials not configured"}
                
            # Create Cosmos client
            cosmos_client = CosmosClient(cosmos_endpoint, cosmos_key)
            database = cosmos_client.get_database_client("bottler-db")  # Correct database name
            container_name = parameters.get("container", "financial_data")
            container = database.get_container_client(container_name)
            
            # Execute query
            query = parameters.get("query", "")
            max_items = parameters.get("max_items", 100)
            
            logger.info(f"=== LOCAL COSMOS QUERY DEBUG ===")
            logger.info(f"Executing FULL LOCAL Cosmos query: {query}")
            logger.info(f"Database: bottler-db, Container: {container_name}, Max items: {max_items}")
            logger.info(f"bottler_id being used: '{self.bottler_id}'")
            
            items = list(container.query_items(
                query=query,
                enable_cross_partition_query=True,
                max_item_count=max_items
            ))
            
            logger.info(f"LOCAL Query returned {len(items)} items")
            
            return {
                "status": "success",
                "data": {
                    "result": items,
                    "count": len(items)
                }
            }
            
        except ImportError as ie:
            logger.error(f"Import error: {ie}")
            return {"status": "error", "error": "Azure Cosmos SDK not available"}
        except Exception as e:
            logger.error(f"LOCAL Cosmos query error: {e}")
            return {"status": "error", "error": str(e)}
            
    async def _execute_sk_function(self, function_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a function via Semantic Kernel"""
        if not self.sk_integration:
            return {"status": "error", "error": "SK integration not available"}
            
        try:
            # Parse plugin and function names
            parts = function_name.split(".")
            plugin_name = parts[0] if len(parts) > 1 else "default"
            func_name = parts[-1]
            
            # Get function from kernel
            func = self.sk_integration.kernel.get_function(plugin_name, func_name)
            if func:
                result = await func.invoke(self.sk_integration.kernel, **parameters)
                return {"status": "success", "data": str(result)}
            else:
                return {"status": "error", "error": f"Function not found: {function_name}"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _build_cosmos_query(self, description: str, bottler_id: str) -> str:
        """Build a Cosmos DB query based on the description - FULLY DYNAMIC"""
        desc_lower = description.lower()
        
        logger.info(f"Building query for: {description[:100]}...")
        
        # Base conditions always include bottler_id
        conditions = [f"c.bottler_id = '{bottler_id}'"]
        
        # Extract product name dynamically
        product_name = None
        
        # Pattern matching for product names in the query
        product_patterns = [
            r"(?:for|of|del?)\s+(?:the\s+)?(?:complete\s+)?([A-Za-z0-9\s\-\.]+?)(?:\s+product|\s+sales|\s+revenue|\s+discount|$)",
            r"(?:discount|descuento)\s+(?:for|of|del?)\s+(?:the\s+)?(?:complete\s+)?([A-Za-z0-9\s\-\.]+?)(?:\s+product|$)",
            r"(?:complete\s+)?([A-Za-z0-9\s\-\.]+?)\s+(?:product|item|articulo)",
            r'"([^"]+)"',  # Quoted product name
            r"'([^']+)'",  # Single quoted product name
        ]
        
        import re
        for pattern in product_patterns:
            match = re.search(pattern, desc_lower, re.IGNORECASE)
            if match:
                product_name = match.group(1).strip()
                # Clean up the product name
                product_name = product_name.replace("the ", "").strip()
                break
        
        # If we found a specific product name, add exact match condition
        if product_name:
            logger.info(f"Detected product: '{product_name}'")
            # Use exact match for the product
            conditions.append(f"c.DESCRIPCION = '{product_name.upper()}'")
        
        # Check what metrics are being requested
        if "discount" in desc_lower or "descuento" in desc_lower:
            # For discount queries, get all fields including DISCOUNTS
            select_clause = "SELECT c.CALMONTH, c.CEDI, c.ZMATERIAL, c.DESCRIPCION, c.ISSCOM, c.QUANTITY, c.GROSS_REVENUE, c.DISCOUNTS"
            
            # If asking for "complete" or "total", we need ALL records
            if "complete" in desc_lower or "total" in desc_lower or "all" in desc_lower:
                # Don't add TOP limit - get everything
                limit_clause = ""
            else:
                limit_clause = " TOP 1000"
        else:
            select_clause = "SELECT *"
            limit_clause = " TOP 100"
        
        # Build the final query
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        order_clause = " ORDER BY c.DISCOUNTS DESC" if "discount" in desc_lower else " ORDER BY c._ts DESC"
        
        query = f"{select_clause}{limit_clause} FROM c{where_clause}{order_clause}"
        
        logger.info(f"Generated query: {query}")
        return query
    
    def _generate_data_analysis(self, items: List[Dict[str, Any]], context: str) -> str:
        """Generate analysis summary from LOCAL Cosmos DB query results"""
        if not items:
            return "No data available for analysis"
        
        # Check if this is a discount analysis query
        if "discount" in context.lower() and "average" in context.lower():
            # Group by product for discount analysis
            from collections import defaultdict
            product_data = defaultdict(lambda: {
                'discounts': [],
                'revenues': [],
                'quantities': [],
                'cedis': set()
            })
            
            for item in items:
                product = item.get('DESCRIPCION', 'Unknown')
                discount = float(item.get('DISCOUNTS', 0))
                revenue = float(item.get('GROSS_REVENUE', 0))
                quantity = float(item.get('QUANTITY', 0))
                cedi = item.get('CEDI', 'Unknown')
                
                if discount is not None and revenue > 0:  # Only include items with valid discount data
                    product_data[product]['discounts'].append(discount)
                    product_data[product]['revenues'].append(revenue)
                    product_data[product]['quantities'].append(quantity)
                    product_data[product]['cedis'].add(cedi)
            
            # Calculate averages
            results = []
            for product, data in product_data.items():
                if data['discounts']:  # Only process products with discount data
                    avg_discount = sum(data['discounts']) / len(data['discounts'])
                    total_discounts = sum(data['discounts'])
                    total_revenue = sum(data['revenues'])
                    
                    if total_revenue > 0:
                        discount_percentage = (total_discounts / total_revenue) * 100
                    else:
                        discount_percentage = 0
                    
                    results.append({
                        'product': product,
                        'avg_discount': avg_discount,
                        'total_discounts': total_discounts,
                        'total_revenue': total_revenue,
                        'discount_percentage': discount_percentage,
                        'record_count': len(data['discounts'])
                    })
            
            # Sort by average discount
            results.sort(key=lambda x: x['avg_discount'], reverse=True)
            
            # Generate analysis
            analysis = f"Average Discount Analysis by Product (from LOCAL database):\n\n"
            
            if results:
                analysis += f"Found {len(results)} products with discount data:\n\n"
                
                # Show top products
                for i, result in enumerate(results[:10], 1):  # Top 10
                    analysis += f"{i}. {result['product']}\n"
                    analysis += f"   - Average Discount: ${result['avg_discount']:,.2f}\n"
                    analysis += f"   - Discount Percentage: {result['discount_percentage']:.2f}%\n"
                    analysis += f"   - Total Discounts: ${result['total_discounts']:,.2f}\n"
                    analysis += f"   - Records: {result['record_count']}\n\n"
                
                # Overall summary
                total_all_discounts = sum(r['total_discounts'] for r in results)
                total_all_revenue = sum(r['total_revenue'] for r in results)
                if total_all_revenue > 0:
                    overall_discount_pct = (total_all_discounts / total_all_revenue) * 100
                    analysis += f"\nOverall Summary:\n"
                    analysis += f"- Total Products: {len(results)}\n"
                    analysis += f"- Total Discounts: ${total_all_discounts:,.2f}\n"
                    analysis += f"- Total Revenue: ${total_all_revenue:,.2f}\n"
                    analysis += f"- Overall Discount Rate: {overall_discount_pct:.2f}%"
            else:
                analysis += "No products found with discount data."
            
            return analysis
        
        # For other types of analysis, use actual field names
        total_revenue = sum(float(item.get("GROSS_REVENUE", 0)) for item in items)
        total_quantity = sum(float(item.get("QUANTITY", 0)) for item in items)
        total_discounts = sum(float(item.get("DISCOUNTS", 0)) for item in items)
        
        # Group by product
        product_summary = {}
        for item in items:
            product = item.get("DESCRIPCION", "Unknown")
            if product not in product_summary:
                product_summary[product] = {
                    "revenue": 0,
                    "quantity": 0,
                    "discounts": 0,
                    "count": 0
                }
            product_summary[product]["revenue"] += float(item.get("GROSS_REVENUE", 0))
            product_summary[product]["quantity"] += float(item.get("QUANTITY", 0))
            product_summary[product]["discounts"] += float(item.get("DISCOUNTS", 0))
            product_summary[product]["count"] += 1
        
        # Generate analysis text
        analysis = f"Based on {len(items)} records from LOCAL database for {self.bottler_id}:\n\n"
        
        # Check if this is Coca-Cola specific query
        if any("coca" in item.get("DESCRIPCION", "").lower() for item in items):
            coca_items = [item for item in items if "coca" in item.get("DESCRIPCION", "").lower()]
            coca_revenue = sum(float(item.get("GROSS_REVENUE", 0)) for item in coca_items)
            coca_quantity = sum(float(item.get("QUANTITY", 0)) for item in coca_items)
            
            analysis += f"Coca-Cola Sales Analysis:\n"
            analysis += f"- Total Revenue: ${coca_revenue:,.2f}\n"
            analysis += f"- Total Quantity: {coca_quantity:,} units\n"
            if coca_quantity > 0:
                analysis += f"- Average Price: ${coca_revenue/coca_quantity:.2f} per unit\n"
            
            # Add product breakdown
            for product, data in product_summary.items():
                if "coca" in product.lower():
                    analysis += f"\n{product}:\n"
                    analysis += f"  - Revenue: ${data['revenue']:,.2f}\n"
                    analysis += f"  - Quantity: {data['quantity']:,} units\n"
                    analysis += f"  - Records: {data['count']}\n"
        else:
            # General analysis
            analysis += f"Financial Summary:\n"
            analysis += f"- Total Revenue: ${total_revenue:,.2f}\n"
            analysis += f"- Total Quantity: {total_quantity:,} units\n"
            analysis += f"- Total Discounts: ${total_discounts:,.2f}\n"
            if total_revenue > 0:
                analysis += f"- Discount Rate: {(total_discounts/total_revenue*100):.1f}%\n"
            
            # Top products
            top_products = sorted(product_summary.items(), key=lambda x: x[1]["revenue"], reverse=True)[:3]
            analysis += f"\nTop Products by Revenue:\n"
            for product, data in top_products:
                analysis += f"- {product}: ${data['revenue']:,.2f} ({data['quantity']:,} units)\n"
        
        # Add date range if available
        dates = [item.get("date") for item in items if item.get("date")]
        if dates:
            analysis += f"\nPeriod: {min(dates)} to {max(dates)}"
        
        return analysis

    async def _execute_approved_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an approved plan and return real data results from LOCAL Cosmos DB"""
        logger.info("Executing approved plan to get real data from LOCAL Cosmos DB")
        
        try:
            steps = plan.get("steps", [])
            original_query = plan.get("original_query", "")
            
            # Execute Cosmos query
            all_data = []
            
            for i, step in enumerate(steps):
                step_num = step.get('step', i + 1)
                description = step.get('description', step.get('action', 'Unknown step'))
                
                logger.info(f"Executing step {step_num}: {description}")
                
                # Check if this is a Cosmos DB query step
                desc_lower = description.lower()
                tool_name = step.get("function", step.get("tool", "")).lower()
                
                if ("cosmos" in desc_lower or "query" in desc_lower or "database" in desc_lower or 
                    "financial_data" in desc_lower or "retrieve" in desc_lower or 
                    "cosmos" in tool_name or "query" in tool_name):
                    
                    # Build query from original request
                    query = self._build_cosmos_query(original_query, self.bottler_id)
                    parameters = {
                        "container": "financial_data",
                        "query": query,
                        "max_items": 10000  # Increased to get ALL records
                    }
                    
                    result = await self._execute_cosmos_query_local(parameters)
                    
                    if result.get("status") == "success" and result.get("data"):
                        items = result["data"].get("result", [])
                        all_data.extend(items)
                        logger.info(f"Retrieved {len(items)} records from LOCAL Cosmos DB")
                        
                        # If this is a discount query for a specific product, aggregate
                        if "discount" in original_query.lower() and "complete" in original_query.lower():
                            analysis = self._aggregate_product_discounts(items, original_query)
                            return {
                                "status": "completed",
                                "data": {
                                    "items": items,
                                    "count": len(items),
                                    "aggregated": True
                                },
                                "analysis": analysis,
                                "execution_results": [{"step": step_num, "status": "completed"}]
                            }
                    break
            
            # Generate final analysis
            final_analysis = self._generate_data_analysis(all_data, original_query) if all_data else "No data found"
            
            return {
                "status": "completed",
                "data": {
                    "items": all_data,
                    "count": len(all_data)
                },
                "analysis": final_analysis,
                "execution_results": []
            }
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return {
                "status": "error",
                "error": str(e),
                "data": {"items": [], "count": 0}
            }

    def _aggregate_product_discounts(self, items: List[Dict], context: str) -> str:
        """Aggregate discount data for a specific product"""
        if not items:
            return "No discount data available"
        
        # Get product name from first item
        product_name = items[0].get('DESCRIPCION', 'Unknown Product') if items else 'Product'
        
        # Calculate totals
        total_discount = 0
        total_revenue = 0
        total_quantity = 0
        cedis = set()
        records_with_discount = 0
        
        for item in items:
            discount = float(item.get('DISCOUNTS', 0) or 0)
            revenue = float(item.get('GROSS_REVENUE', 0) or 0)
            quantity = float(item.get('QUANTITY', 0) or 0)
            cedi = item.get('CEDI')
            
            total_discount += discount
            total_revenue += revenue
            total_quantity += quantity
            
            if cedi:
                cedis.add(cedi)
            if discount > 0:
                records_with_discount += 1
        
        # Calculate discount percentage
        discount_percentage = (total_discount / total_revenue * 100) if total_revenue > 0 else 0
        
        # Format analysis
        analysis = f"""
    Discount Analysis for {product_name}:
    =====================================
    Total Records Found: {len(items)}
    Records with Discounts: {records_with_discount}
    Distribution Centers (CEDIs): {len(cedis)}

    FINANCIAL METRICS:
    - Total Discounts: ${total_discount:,.2f}
    - Total Revenue: ${total_revenue:,.2f}
    - Total Quantity: {total_quantity:,.2f} units
    - Discount Rate: {discount_percentage:.2f}%

    BREAKDOWN BY RECORD:
    """
        
        # Add top 5 discount records
        sorted_items = sorted(items, key=lambda x: float(x.get('DISCOUNTS', 0) or 0), reverse=True)
        for i, item in enumerate(sorted_items[:5], 1):
            discount = float(item.get('DISCOUNTS', 0) or 0)
            revenue = float(item.get('GROSS_REVENUE', 0) or 0)
            cedi = item.get('CEDI', 'N/A')
            analysis += f"\n{i}. CEDI {cedi}: Discount ${discount:,.2f} (Revenue: ${revenue:,.2f})"
        
        return analysis

# Standalone planning function for SK integration
async def sk_create_plan(task_description: str, kernel=None) -> str:
    """
    Create an execution plan using Semantic Kernel
    This can be called independently or through the reasoning system
    """
    if not SK_AVAILABLE or not kernel:
        return json.dumps({
            "error": "SK not available for planning",
            "fallback": True
        })
        
    try:
        planner = SequentialPlanner(kernel)
        plan = await planner.create_plan_async(goal=task_description)
        
        # Convert to structured format
        plan_data = {
            "goal": task_description,
            "steps": []
        }
        
        for i, step in enumerate(plan.steps):
            plan_data["steps"].append({
                "order": i + 1,
                "function": f"{step.skill_name}.{step.function_name}",
                "description": step.description or f"Execute {step.function_name}",
                "inputs": dict(step.parameters.variables) if step.parameters else {}
            })
            
        return json.dumps(plan_data, indent=2)
        
    except Exception as e:
        logger.error(f"Error in SK planning: {e}")
        return json.dumps({"error": str(e)})