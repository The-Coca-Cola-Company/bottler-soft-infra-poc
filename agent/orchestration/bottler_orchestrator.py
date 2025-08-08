"""
Prompt Manager for Bottler SPOKE
================================

Manages prompts from Excel files and other sources.
Stores prompts in structured format for reuse.

Author: TCCC Emerging Technology
Version: 1.0.0
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime
import hashlib
from pathlib import Path
import io

# Import pandas for Excel processing
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    pd = None

logger = logging.getLogger(__name__)


class PromptManager:
    """Manages prompts for the bottler system"""
    
    def __init__(self, mcp_bridge: Any = None):
        """Initialize prompt manager"""
        self.mcp_bridge = mcp_bridge
        self.prompts_dir = Path(__file__).parent
        self.bottler_id = os.getenv("BOTTLER_ID", "unknown")
        
        # Ensure prompt directories exist
        self.ensure_directories()
        
    def ensure_directories(self):
        """Ensure all required directories exist"""
        directories = [
            self.prompts_dir / "system",
            self.prompts_dir / "user",
            self.prompts_dir / "templates",
            self.prompts_dir / "excel_imports"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            
    async def process_excel_file(self, file_path: str, file_content: bytes = None) -> Dict[str, Any]:
        """
        Process an Excel file containing prompts/queries
        
        Args:
            file_path: Path to Excel file or blob name
            file_content: File content if already loaded
            
        Returns:
            Processing result with extracted prompts
        """
        try:
            if not PANDAS_AVAILABLE:
                return {
                    "success": False,
                    "error": "Pandas not available - cannot process Excel files"
                }
                
            logger.info(f"Processing Excel file: {file_path}")
            
            # Load Excel file
            if file_content:
                # Convert bytes to BytesIO for pandas
                file_stream = io.BytesIO(file_content)
                df = pd.read_excel(file_stream, engine='openpyxl')
            else:
                # Try to read from blob storage via MCP
                if self.mcp_bridge and file_path.startswith("blob://"):
                    blob_name = file_path.replace("blob://", "")
                    result = await self.mcp_bridge.execute_tool(
                        server_name="blob",
                        tool_name="read_blob",
                        arguments={"blob_name": blob_name}
                    )
                    
                    if result.get("success"):
                        content = result.get("result", {}).get("content", "")
                        # Assuming content is base64 encoded or raw bytes
                        if isinstance(content, str):
                            import base64
                            content = base64.b64decode(content)
                        file_stream = io.BytesIO(content)
                        df = pd.read_excel(file_stream, engine='openpyxl')
                    else:
                        raise Exception(f"Failed to read blob: {result.get('error')}")
                else:
                    df = pd.read_excel(file_path, engine='openpyxl')
                    
            # Extract prompts based on common column patterns
            prompts = self._extract_prompts_from_dataframe(df)
            
            # Store prompts
            stored_prompts = []
            for prompt in prompts:
                stored = await self.store_prompt(prompt)
                stored_prompts.append(stored)
                
            # Save the import record
            import_record = {
                "id": f"import-{datetime.utcnow().timestamp()}",
                "file_name": os.path.basename(file_path),
                "import_date": datetime.utcnow().isoformat(),
                "bottler_id": self.bottler_id,
                "prompts_extracted": len(prompts),
                "status": "completed"
            }
            
            # Store in blob
            if self.mcp_bridge:
                await self.mcp_bridge.execute_tool(
                    server_name="blob",
                    tool_name="write_blob",
                    arguments={
                        "blob_name": f"prompts/imports/{import_record['id']}.json",
                        "content": json.dumps(import_record),
                        "content_type": "application/json",
                        "metadata": {
                            "bottler_id": self.bottler_id,
                            "type": "excel_import_record"
                        }
                    }
                )
                
            return {
                "success": True,
                "import_record": import_record,
                "prompts": stored_prompts
            }
            
        except Exception as e:
            logger.error(f"Error processing Excel file: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
            
    def _extract_prompts_from_dataframe(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Extract prompts from a pandas DataFrame"""
        if not PANDAS_AVAILABLE:
            return []
            
        prompts = []
        
        # Common column patterns for prompts
        prompt_columns = [
            "prompt", "query", "question", "pregunta", "consulta",
            "user_input", "text", "description", "request"
        ]
        
        category_columns = [
            "category", "type", "categoria", "tipo", "classification"
        ]
        
        # Find relevant columns
        prompt_col = None
        category_col = None
        
        for col in df.columns:
            col_lower = col.lower().strip()
            if not prompt_col and any(pc in col_lower for pc in prompt_columns):
                prompt_col = col
            if not category_col and any(cc in col_lower for cc in category_columns):
                category_col = col
                
        if not prompt_col:
            # If no prompt column found, try first text column
            for col in df.columns:
                if df[col].dtype == 'object':
                    prompt_col = col
                    break
                    
        if prompt_col:
            for index, row in df.iterrows():
                prompt_text = str(row[prompt_col]).strip()
                
                if prompt_text and prompt_text.lower() not in ['nan', 'none', '']:
                    prompt = {
                        "text": prompt_text,
                        "category": str(row[category_col]) if category_col and pd.notna(row[category_col]) else "general",
                        "source": "excel_import",
                        "row_index": index,
                        "metadata": {}
                    }
                    
                    # Add other columns as metadata
                    for col in df.columns:
                        if col not in [prompt_col, category_col]:
                            value = row[col]
                            if pd.notna(value):
                                prompt["metadata"][col] = str(value)
                                
                    prompts.append(prompt)
                    
        return prompts
        
    async def store_prompt(self, prompt_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Store a prompt in the system
        
        Args:
            prompt_data: Prompt information
            
        Returns:
            Stored prompt with ID
        """
        try:
            # Generate unique ID
            prompt_id = self._generate_prompt_id(prompt_data["text"])
            
            # Create prompt document
            prompt_doc = {
                "id": prompt_id,
                "text": prompt_data["text"],
                "category": prompt_data.get("category", "general"),
                "bottler_id": self.bottler_id,
                "created_at": datetime.utcnow().isoformat(),
                "source": prompt_data.get("source", "manual"),
                "metadata": prompt_data.get("metadata", {}),
                "usage_count": 0,
                "last_used": None,
                "variations": [],
                "responses": []
            }
            
            # Classify the prompt
            prompt_doc["classification"] = self._classify_prompt(prompt_data["text"])
            
            # Store locally
            local_path = self.prompts_dir / f"{prompt_doc['category']}" / f"{prompt_id}.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(local_path, 'w', encoding='utf-8') as f:
                json.dump(prompt_doc, f, indent=2, ensure_ascii=False)
                
            # Store in Cosmos DB if available
            if self.mcp_bridge:
                await self.mcp_bridge.execute_tool(
                    server_name="cosmos",
                    tool_name="upsert_document",
                    arguments={
                        "container": "prompts",
                        "document": prompt_doc
                    }
                )
                
            logger.info(f"Stored prompt: {prompt_id}")
            return prompt_doc
            
        except Exception as e:
            logger.error(f"Error storing prompt: {str(e)}")
            raise
            
    def _generate_prompt_id(self, text: str) -> str:
        """Generate unique ID for a prompt"""
        # Create hash of the text
        hash_object = hashlib.md5(text.encode('utf-8'))
        hash_hex = hash_object.hexdigest()[:8]
        
        return f"prompt-{self.bottler_id}-{hash_hex}"
        
    def _classify_prompt(self, text: str) -> Dict[str, Any]:
        """Classify a prompt based on its content"""
        text_lower = text.lower()
        
        classification = {
            "query_type": "unknown",
            "domain": "general",
            "complexity": "simple",
            "entities": []
        }
        
        # Detect query type
        if any(word in text_lower for word in ["sales", "ventas", "revenue", "ingresos"]):
            classification["query_type"] = "sales"
            classification["domain"] = "financial"
        elif any(word in text_lower for word in ["cost", "costo", "expense", "gasto"]):
            classification["query_type"] = "costs"
            classification["domain"] = "financial"
        elif any(word in text_lower for word in ["margin", "margen", "profit", "ganancia"]):
            classification["query_type"] = "profitability"
            classification["domain"] = "financial"
        elif any(word in text_lower for word in ["inventory", "inventario", "stock"]):
            classification["query_type"] = "inventory"
            classification["domain"] = "operations"
            
        # Detect products
        products = ["coca-cola", "sprite", "fanta", "powerade", "del valle"]
        for product in products:
            if product in text_lower:
                classification["entities"].append({"type": "product", "value": product})
                
        # Detect time periods
        if any(word in text_lower for word in ["month", "mes", "monthly", "mensual"]):
            classification["entities"].append({"type": "period", "value": "monthly"})
        elif any(word in text_lower for word in ["year", "año", "annual", "anual"]):
            classification["entities"].append({"type": "period", "value": "yearly"})
            
        # Assess complexity
        word_count = len(text.split())
        if word_count > 20 or len(classification["entities"]) > 2:
            classification["complexity"] = "complex"
        elif word_count > 10:
            classification["complexity"] = "medium"
            
        return classification
        
    async def get_prompt_by_id(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a prompt by ID"""
        try:
            # Try local first
            for category_dir in (self.prompts_dir).iterdir():
                if category_dir.is_dir():
                    prompt_file = category_dir / f"{prompt_id}.json"
                    if prompt_file.exists():
                        with open(prompt_file, 'r', encoding='utf-8') as f:
                            return json.load(f)
                            
            # Try Cosmos DB
            if self.mcp_bridge:
                result = await self.mcp_bridge.execute_tool(
                    server_name="cosmos",
                    tool_name="query_documents",
                    arguments={
                        "container": "prompts",
                        "query": f"SELECT * FROM c WHERE c.id = '{prompt_id}'",
                        "max_items": 1
                    }
                )
                
                if result.get("success") and result.get("result"):
                    return result["result"][0]
                    
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving prompt: {str(e)}")
            return None
            
    async def search_prompts(self, query: str, category: str = None) -> List[Dict[str, Any]]:
        """Search for prompts"""
        try:
            # Build Cosmos query
            cosmos_query = f"SELECT * FROM c WHERE c.bottler_id = '{self.bottler_id}'"
            
            if category:
                cosmos_query += f" AND c.category = '{category}'"
                
            if query:
                cosmos_query += f" AND CONTAINS(LOWER(c.text), LOWER('{query}'))"
                
            cosmos_query += " ORDER BY c.usage_count DESC"
            
            if self.mcp_bridge:
                result = await self.mcp_bridge.execute_tool(
                    server_name="cosmos",
                    tool_name="query_documents",
                    arguments={
                        "container": "prompts",
                        "query": cosmos_query,
                        "max_items": 50
                    }
                )
                
                if result.get("success"):
                    return result.get("result", [])
                    
            # Fallback to local search
            prompts = []
            for category_dir in (self.prompts_dir).iterdir():
                if category_dir.is_dir() and (not category or category_dir.name == category):
                    for prompt_file in category_dir.glob("*.json"):
                        try:
                            with open(prompt_file, 'r', encoding='utf-8') as f:
                                prompt = json.load(f)
                                if not query or query.lower() in prompt["text"].lower():
                                    prompts.append(prompt)
                        except Exception:
                            continue
                                
            return sorted(prompts, key=lambda x: x.get("usage_count", 0), reverse=True)
            
        except Exception as e:
            logger.error(f"Error searching prompts: {str(e)}")
            return []
            
    async def update_prompt_usage(self, prompt_id: str, response: str = None):
        """Update prompt usage statistics"""
        try:
            prompt = await self.get_prompt_by_id(prompt_id)
            
            if prompt:
                prompt["usage_count"] = prompt.get("usage_count", 0) + 1
                prompt["last_used"] = datetime.utcnow().isoformat()
                
                if response:
                    prompt["responses"].append({
                        "response": response[:500],  # Store first 500 chars
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    # Keep only last 10 responses
                    prompt["responses"] = prompt["responses"][-10:]
                    
                # Update in Cosmos DB
                if self.mcp_bridge:
                    await self.mcp_bridge.execute_tool(
                        server_name="cosmos",
                        tool_name="upsert_document",
                        arguments={
                            "container": "prompts",
                            "document": prompt
                        }
                    )
                    
        except Exception as e:
            logger.error(f"Error updating prompt usage: {str(e)}")
