# Soft Bottler Manager - Complete Azure Infrastructure Deployment

# 🏭 Bottler Agent - Azure Deployment 

![Cross Tenant Architecture](deployment/img/Image%20cross.png)



This repository contains the Azure Resource Manager (ARM) templates for deploying the **COMPLETE** infrastructure required by the Soft Bottler Manager - a spoke component in the TCCC multi-tenant hub-spoke architecture.

## 📋 Overview

The Soft Bottler Manager is designed as a **SPOKE** in the hub-spoke architecture, meaning:
- ✅ **All communication flows through the TCCC Hub** (no direct spoke-to-spoke communication)
- ✅ **No APIM required** - the Hub's APIM handles all API management
- ✅ **Complete Azure infrastructure** for AI/ML capabilities, messaging, and data storage
- ✅ **Multi-tenant ready** - can be deployed in separate Azure subscriptions/tenants

## 🏗️ Infrastructure Components

### Core Services Included
- **🗄️ Storage Account** (Standard_GRS) - Primary storage with containers:
  - `bottler-cache` - Caching layer
  - `excel-processing` - Excel file processing
  - `mcp-data` - Model Context Protocol data
- **🔗 Cosmos DB** (Serverless) - NoSQL database with collections:
  - `financial_data` - Financial analysis data
  - `bottler_config` - Configuration storage
  - `prompts` - AI prompt management
  - `health_checks` - Health monitoring data
- **📨 Service Bus** (Standard/Premium) - Messaging infrastructure:
  - Queue: `bottler-processing`
  - Queue: `excel-processing`
  - Topic: `bottler-events`
- **⚡ Event Grid** - Event-driven architecture support
- **🤖 Azure ML Workspace** - AI Foundry capabilities for ML operations
- **📊 Application Insights & Log Analytics** - Full monitoring stack
- **🌐 Virtual Network** - Network isolation with subnets:
  - `function-subnet` (10.0.1.0/24) - For Function App
  - `ml-subnet` (10.0.2.0/24) - For ML Workspace
- **⚙️ Function App** (Python 3.11) - Serverless compute with all integrations

### Key Features
- **Hub Integration** - Pre-configured to communicate with TCCC Hub via APIM
- **MCP Support** - Model Context Protocol enabled for AI agent communication
- **Semantic Kernel** - Built-in support for SK plugins and memory
- **AutoGen** - Agent-to-agent orchestration capabilities
- **Security** - Managed identities, TLS 1.2 minimum, network isolation
- **Monitoring** - Full observability with Application Insights integration

## 🚀 Quick Deploy

Deploy this infrastructure to your Azure subscription with a single click:


[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FThe-Coca-Cola-Company%2Fsoft-bottler-infrastructure%2Fmain%2Fdeployment%2Fsoft-bottler-complete-infra.json)


## 📝 Pre-Deployment Requirements

### 1. Azure Subscription
- Active Azure subscription with appropriate permissions
- Resource Group created for the bottler deployment
- Sufficient quota for all services

### 2. TCCC Hub Information
You'll need the following from your TCCC Hub deployment:
- **APIM Gateway URL**: `https://tccc-soft-prod-apim-{uniqueid}.azure-api.net`
- **Subscription Key**: Your bottler's unique subscription key from the Hub's APIM

### 3. Azure AI Foundry Configuration
- **Endpoint**: Your Azure AI Foundry endpoint (e.g., `https://aif-aif-tccc-sbx-01.openai.azure.com/`)
- **Subscription Key**: Your Azure AI Foundry subscription key
- **Model Deployment**: Default is `tccc-model-router`
- **API Version**: Default is `2024-12-01-preview`

### 4. Bottler Identity
- **Bottler Code**: Unique identifier (e.g., `arca`, `femsa`, `andina`)
- **Location**: Azure region for deployment (e.g., `eastus`, `westus2`)

## 🔧 Deployment Instructions

### PHASE 1: Initial Infrastructure Deployment

```bash
# Step 1: Set only the values you know NOW
export RESOURCE_GROUP="rg-bottler-{your-code}-prod"
export LOCATION="eastus"  # or your preferred region
export BOTTLER_CODE="arca" # Your unique bottler code

# Step 2: Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Step 3: Deploy infrastructure with defaults
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file soft-bottler-complete-infra.json \
  --parameters bottlerCode=$BOTTLER_CODE
```

**What happens**: 
- ✅ All Azure resources are created
- ✅ Function App is deployed with placeholder values
- ⚠️ Function App won't work yet (needs real endpoints)

### PHASE 2: Post-Deployment Configuration

After getting the real values from TCCC Hub admin and AI Foundry:

```bash
# Update Function App settings with real values
FUNCTION_APP_NAME="soft-$BOTTLER_CODE-prod-func-xxxxx" # Get from deployment output

az functionapp config appsettings set \
  --name $FUNCTION_APP_NAME \
  --resource-group $RESOURCE_GROUP \
  --settings \
    "TCCC_HUB_URL=https://actual-hub-url.azure-api.net" \
    "TCCC_HUB_API_KEY=actual-subscription-key-from-hub" \
    "AZURE_AI_FOUNDRY_ENDPOINT=https://actual-ai-foundry.openai.azure.com/" \
    "AZURE_AI_FOUNDRY_KEY=actual-ai-foundry-key"
```

### Using PowerShell

**PHASE 1: Initial Deployment**
```powershell
# Set only what you know
$resourceGroup = "rg-bottler-{your-code}-prod"
$location = "eastus"
$bottlerCode = "arca"  # Your bottler code

# Create resource group
New-AzResourceGroup -Name $resourceGroup -Location $location

# Deploy infrastructure
New-AzResourceGroupDeployment `
  -ResourceGroupName $resourceGroup `
  -TemplateFile "soft-bottler-complete-infra.json" `
  -bottlerCode $bottlerCode
```

**PHASE 2: Update Configuration**
```powershell
$functionAppName = "soft-$bottlerCode-prod-func-xxxxx"  # From deployment output

# Update with real values when available
az functionapp config appsettings set `
  --name $functionAppName `
  --resource-group $resourceGroup `
  --settings `
    "TCCC_HUB_URL=https://actual-hub-url.azure-api.net" `
    "TCCC_HUB_API_KEY=actual-key" `
    "AZURE_AI_FOUNDRY_ENDPOINT=https://actual-endpoint.openai.azure.com/" `
    "AZURE_AI_FOUNDRY_KEY=actual-key"
```

## 🔐 Security Considerations

### Secrets Management
- **Never commit secrets** to source control
- Use **GitHub Secrets** or **Azure DevOps Secure Variables** for:
  - `TCCC_HUB_SUBSCRIPTION_KEY`
  - `AZURE_AI_FOUNDRY_KEY`
- All connection strings are automatically configured in Function App settings

### Network Security
- Function App deployed with VNet integration
- Storage accounts configured with service endpoints
- All services use TLS 1.2 minimum
- Public access can be disabled post-deployment

### Identity & Access
- Function App uses System Assigned Managed Identity
- Post-deployment, assign RBAC permissions:
  ```bash
  # Get the Function App's identity
  IDENTITY=$(az functionapp identity show \
    --name {functionAppName} \
    --resource-group $RESOURCE_GROUP \
    --query principalId -o tsv)
  
  # Assign necessary roles
  az role assignment create \
    --assignee $IDENTITY \
    --role "Storage Blob Data Contributor" \
    --scope /subscriptions/{subscription-id}/resourceGroups/$RESOURCE_GROUP
  ```

## 📊 Post-Deployment Configuration

### 1. Verify Deployment Outputs
```bash
# Get deployment outputs
az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name {deployment-name} \
  --query properties.outputs -o json
```

Key outputs include:
- `functionAppUrl` - Your bottler's endpoint
- `functionAppDefaultHostKey` - Authentication key
- `cosmosConnectionString` - Cosmos DB connection
- `serviceBusConnectionString` - Service Bus connection
- `storageAccountConnectionString` - Storage connection
- `eventGridTopicEndpoint` - Event Grid endpoint
- `mlWorkspaceId` - ML Workspace identifier

### 2. Register with TCCC Hub
```bash
# Get Function App details
FUNCTION_URL=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name {deployment-name} \
  --query properties.outputs.functionAppUrl.value -o tsv)

FUNCTION_KEY=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name {deployment-name} \
  --query properties.outputs.functionAppDefaultHostKey.value -o tsv)

# Register with Hub
curl -X POST "$TCCC_HUB_URL/api/agents/register" \
  -H "Ocp-Apim-Subscription-Key: $TCCC_HUB_SUBSCRIPTION_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "bottler_id": "'$BOTTLER_CODE'",
    "capabilities": ["financial_analysis", "excel_processing", "mcp_enabled"],
    "endpoint": "'$FUNCTION_URL'",
    "region": "'$LOCATION'",
    "api_key": "'$FUNCTION_KEY'"
  }'
```

### 3. Test Bottler Health
```bash
# Test health endpoint
curl -X GET "$FUNCTION_URL/api/health" \
  -H "x-functions-key: $FUNCTION_KEY"
```

Expected response:
```json
{
  "status": "healthy",
  "agent_type": "BOTTLER_FINANCIAL_EXPERT",
  "bottler_code": "your-bottler-code",
  "hub_connectivity": "connected",
  "services": {
    "cosmos_db": "connected",
    "service_bus": "connected",
    "storage": "connected",
    "event_grid": "connected",
    "ml_workspace": "connected"
  }
}
```

### 4. Configure MCP Servers
The infrastructure supports multiple MCP servers, all enabled by default:
- **Cosmos DB MCP**: For structured data operations
- **Blob Storage MCP**: For file operations
- **Service Bus MCP**: For messaging operations

## 🎯 Architecture Alignment

This infrastructure follows the TCCC Multi-Tenant Hub-Spoke pattern:

```
External Request → TCCC Hub (APIM) → Bottler Agent (Function App) → Response via Hub
                        ↓                      ↓
                   Rate Limiting         All Azure Services
                   Authentication        (Cosmos, Storage, etc.)
                   Hub-Spoke Check
```

### Key Architectural Rules
- ❌ **No direct spoke-to-spoke communication**
- ✅ **All requests flow through the Hub**
- ✅ **Hub-Spoke Enforcer validates all communications**
- ✅ **Each bottler can be in a separate tenant**
- ✅ **Complete service isolation per bottler**

## 📈 Monitoring & Diagnostics

### Application Insights Dashboard
```bash
# Open Application Insights
AI_NAME=$(az deployment group show \
  --resource-group $RESOURCE_GROUP \
  --name {deployment-name} \
  --query properties.outputs.appInsightsName.value -o tsv)

echo "https://portal.azure.com/#resource/subscriptions/{subscription-id}/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Insights/components/$AI_NAME/overview"
```

### Key Metrics to Monitor
- **Function Execution Count** - Track agent activity
- **Response Times** - Ensure <2s for most operations
- **Error Rate** - Should be <1%
- **Cosmos DB RU Consumption** - Monitor for scaling needs
- **Service Bus Message Count** - Track processing backlog

### Custom Queries
```kusto
// Hub communication performance
requests
| where name contains "hub" or customDimensions.target contains "TCCC"
| summarize avg(duration), percentiles(duration, 50, 95, 99) by bin(timestamp, 5m)
| render timechart

// Service health status
customEvents
| where name == "ServiceHealthCheck"
| summarize by tostring(customDimensions.service), tostring(customDimensions.status)
| render piechart
```

## 🔄 Updates & Maintenance

### Infrastructure Updates
```bash
# Update existing deployment (incremental)
az deployment group create \
  --resource-group $RESOURCE_GROUP \
  --template-file soft-bottler-complete-infra.json \
  --parameters @parameters.prod-complete-infra.json \
  --mode Incremental
```

### Scaling Considerations
- **Service Bus**: Upgrade to Premium for better performance
- **Cosmos DB**: Switch from Serverless to Provisioned for high volume
- **Function App**: Move to Premium plan for consistent performance
- **ML Workspace**: Scale compute clusters based on workload

## 🤝 Support & Contribution

### Common Issues
1. **Deployment Timeout**: Some resources (ML Workspace) take 10-15 minutes
2. **Service Bus Connection**: Ensure firewall rules allow Function App
3. **Cosmos DB Throttling**: Monitor RU consumption, scale if needed
4. **VNet Integration**: May require additional subnet configuration

### Getting Help
- Infrastructure issues: Create GitHub issue with `[INFRA]` tag
- Include deployment correlation ID and error messages
- Check Application Insights for detailed error traces

## 📚 Related Documentation

- [TCCC Hub Infrastructure](../../../tccc-soft-manager/deployment/README.md)
- [Multi-Tenant Architecture Guide](../../../../docs/architecture.md)
- [MCP Protocol Reference](../../../../docs/mcp-protocol.md)
- [Hub-Spoke Security Model](../../../../docs/security.md)

---

**Version**: 3.0.0  
**Last Updated**: 8-5-2025
**Template**: `soft-bottler-complete-infra.json`  

**Maintained By**: TCCC Engineering Team (cvanegas@coca-cola.com)

