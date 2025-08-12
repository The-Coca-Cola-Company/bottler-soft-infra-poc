"""
Bottler Integration Layer
========================

Integration components for the Bottler SPOKE agent.
Includes Semantic Kernel and AutoGen with direct database access.

Author: TCCC Emerging Technology
Version: 1.1.0
"""

from .semantic_kernel_integration import BottlerSemanticKernelIntegration as SemanticKernelIntegration
from .autogen_orchestrator import BottlerAutoGenOrchestrator as AutoGenOrchestrator

__all__ = [
    'SemanticKernelIntegration',
    'AutoGenOrchestrator'
]