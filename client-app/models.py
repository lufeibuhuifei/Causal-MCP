# client-app/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from enum import Enum

class AnalysisType(str, Enum):
    """Types of causal analysis supported."""
    FULL_CAUSAL_ANALYSIS = "full_causal_analysis"
    PATHWAY_ANALYSIS = "pathway_analysis"
    GENE_ANNOTATION = "gene_annotation"
    DRUG_TARGET_ANALYSIS = "drug_target_analysis"

class CausalAnalysisRequest(BaseModel):
    """
    Request for causal analysis using the Causal-MCP framework.
    """
    exposure_gene: str = Field(
        ...,
        description="Gene symbol for the exposure variable (e.g., 'PCSK9')",
        examples=["PCSK9", "LPA", "LDLR"]
    )
    outcome_trait: str = Field(
        ...,
        description="Outcome trait or disease (e.g., 'Coronary Artery Disease')",
        examples=["Coronary Artery Disease", "LDL Cholesterol", "Type 2 Diabetes"]
    )
    tissue_context: Optional[str] = Field(
        "Whole_Blood",
        description="Tissue context for eQTL analysis",
        examples=["Whole_Blood", "Liver", "Brain_Cortex"]
    )
    analysis_type: AnalysisType = Field(
        AnalysisType.FULL_CAUSAL_ANALYSIS,
        description="Type of analysis to perform"
    )
    include_pathway_analysis: bool = Field(
        True,
        description="Whether to include pathway enrichment analysis"
    )
    include_drug_analysis: bool = Field(
        True,
        description="Whether to include drug target analysis"
    )
    language: str = Field(
        "en",
        description="Language for interpretation and recommendations",
        examples=["en", "zh"]
    )
    show_thinking: bool = Field(
        False,
        description="Whether to show LLM thinking process in the output"
    )

class AnalysisStep(BaseModel):
    """
    Represents a single step in the analysis workflow.
    """
    step_name: str = Field(..., description="Name of the analysis step")
    status: str = Field(..., description="Status of the step (pending, running, completed, failed)")
    server_used: Optional[str] = Field(None, description="MCP server used for this step")
    input_data: Optional[Dict[str, Any]] = Field(None, description="Input data for this step")
    output_data: Optional[Dict[str, Any]] = Field(None, description="Output data from this step")
    error_message: Optional[str] = Field(None, description="Error message if step failed")
    execution_time: Optional[float] = Field(None, description="Execution time in seconds")

class CausalAnalysisResult(BaseModel):
    """
    Complete result from causal analysis.
    """
    request: CausalAnalysisRequest = Field(..., description="Original analysis request")
    analysis_steps: List[AnalysisStep] = Field(..., description="List of analysis steps performed")
    
    # Main results
    eqtl_instruments: Optional[List[Dict[str, Any]]] = Field(None, description="eQTL instruments found")
    harmonized_data: Optional[List[Dict[str, Any]]] = Field(None, description="Harmonized exposure-outcome data")
    mr_results: Optional[Dict[str, Any]] = Field(None, description="Mendelian Randomization results")
    
    # Additional analyses
    pathway_analysis: Optional[Dict[str, Any]] = Field(None, description="Pathway enrichment results")
    gene_annotation: Optional[Dict[str, Any]] = Field(None, description="Gene functional annotation")
    drug_analysis: Optional[Dict[str, Any]] = Field(None, description="Drug target analysis")
    
    # Summary and interpretation
    summary: Dict[str, Any] = Field(..., description="Summary of key findings")
    interpretation: str = Field(..., description="Detailed interpretation of results")
    recommendations: List[str] = Field(..., description="Recommendations for follow-up")
    
    # Metadata
    total_execution_time: float = Field(..., description="Total analysis time in seconds")
    success: bool = Field(..., description="Whether analysis completed successfully")
    warnings: List[str] = Field(default_factory=list, description="Any warnings generated")
    error_message: Optional[str] = Field(None, description="Error message if analysis failed")

class ServerStatus(BaseModel):
    """
    Status of an MCP server.
    """
    server_name: str = Field(..., description="Name of the MCP server")
    status: str = Field(..., description="Status (online, offline, error)")
    last_check: str = Field(..., description="Last status check timestamp")
    error_message: Optional[str] = Field(None, description="Error message if server is down")

class SystemStatus(BaseModel):
    """
    Overall system status.
    """
    servers: List[ServerStatus] = Field(..., description="Status of all MCP servers")
    system_health: str = Field(..., description="Overall system health (healthy, degraded, down)")
    last_updated: str = Field(..., description="Last status update timestamp")

# Simplified models for specific analysis types
class PathwayAnalysisRequest(BaseModel):
    """Request for pathway analysis only."""
    gene_list: List[str] = Field(..., description="List of gene symbols to analyze")
    organism: str = Field("hsapiens", description="Organism for analysis")

class GeneAnnotationRequest(BaseModel):
    """Request for gene annotation only."""
    gene_symbol: str = Field(..., description="Gene symbol to annotate")
    include_interactions: bool = Field(True, description="Include protein interactions")
    include_diseases: bool = Field(True, description="Include disease associations")

class DrugTargetRequest(BaseModel):
    """Request for drug target analysis only."""
    gene_symbol: str = Field(..., description="Gene symbol to analyze")
    include_clinical_trials: bool = Field(True, description="Include clinical trial data")
