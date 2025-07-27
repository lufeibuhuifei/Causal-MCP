"""
MCP-Knowledge Server

A Model Context Protocol server for biological knowledge retrieval and pathway analysis.
Provides access to biological databases and functional annotation services.
"""

__version__ = "0.1.0"

from .models import (
    PathwayAnalysisInput,
    PathwayAnalysisOutput,
    GeneAnnotationInput, 
    GeneAnnotationOutput,
    DrugTargetInput,
    DrugTargetOutput,
    PathwayTerm,
    ProteinInteraction,
    DiseaseAssociation,
    DrugCompound
)
from .main import mcp, analyze_pathways, annotate_gene, analyze_drug_targets, run
from .knowledge_apis import BiologicalKnowledgeAPI

__all__ = [
    "PathwayAnalysisInput",
    "PathwayAnalysisOutput", 
    "GeneAnnotationInput",
    "GeneAnnotationOutput",
    "DrugTargetInput",
    "DrugTargetOutput",
    "PathwayTerm",
    "ProteinInteraction", 
    "DiseaseAssociation",
    "DrugCompound",
    "mcp",
    "analyze_pathways",
    "annotate_gene", 
    "analyze_drug_targets",
    "run",
    "BiologicalKnowledgeAPI"
]
