# src/mcp_server_knowledge/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# --- 输入模型 ---
class PathwayAnnotationInput(BaseModel):
    """
    Input for pathway annotation (not enrichment analysis).
    """
    gene_symbol: str = Field(
        ...,
        description="Gene symbol to get pathway information for.",
        examples=["PCSK9"]
    )
    organism: Optional[str] = Field(
        "hsapiens",
        description="Organism for analysis (hsapiens, mmusculus, etc.)"
    )

class GeneAnnotationInput(BaseModel):
    """
    Input for gene functional annotation.
    """
    gene_symbol: str = Field(
        ...,
        description="Gene symbol to annotate.",
        examples=["PCSK9"]
    )
    include_interactions: bool = Field(
        True,
        description="Whether to include protein-protein interactions."
    )
    include_diseases: bool = Field(
        True,
        description="Whether to include disease associations."
    )

class DrugTargetInput(BaseModel):
    """
    Input for drug-target analysis.
    """
    gene_symbol: str = Field(
        ...,
        description="Gene symbol to search for drug targets.",
        examples=["PCSK9"]
    )
    include_clinical_trials: bool = Field(
        True,
        description="Whether to include clinical trial information."
    )

# --- 输出模型 ---
class PathwayInfo(BaseModel):
    """
    Represents pathway information for a gene (annotation, not enrichment analysis).
    """
    pathway_id: str = Field(..., description="Pathway identifier")
    pathway_name: str = Field(..., description="Pathway name")
    description: str = Field(..., description="Pathway description")
    gene_role: Optional[str] = Field(None, description="Role of the gene in this pathway")
    source_database: str = Field(..., description="Source database (KEGG, Reactome, etc.)")
    pathway_url: Optional[str] = Field(None, description="URL to pathway details")

class PathwayAnnotationOutput(BaseModel):
    """
    Output from pathway annotation (not enrichment analysis).
    """
    gene_symbol: str = Field(..., description="Gene symbol that was analyzed")
    summary: Dict[str, Any] = Field(
        ...,
        description="Summary information about the gene's pathway involvement"
    )
    pathways: List[PathwayInfo] = Field(
        ...,
        description="List of pathways the gene participates in"
    )
    interpretation: str = Field(
        ...,
        description="Natural language interpretation of the gene's pathway roles"
    )

class ProteinInteraction(BaseModel):
    """
    Represents a protein-protein interaction.
    """
    partner_gene: str = Field(..., description="Interacting gene symbol")
    partner_protein: str = Field(..., description="Interacting protein name")
    interaction_type: str = Field(..., description="Type of interaction")
    confidence_score: Optional[float] = Field(None, description="Confidence score")
    source_database: str = Field(..., description="Source database")

class DiseaseAssociation(BaseModel):
    """
    Represents a gene-disease association.
    """
    disease_name: str = Field(..., description="Disease name")
    disease_id: str = Field(..., description="Disease identifier (e.g., OMIM, DOID)")
    association_type: str = Field(..., description="Type of association")
    evidence_level: str = Field(..., description="Level of evidence")
    source_database: str = Field(..., description="Source database")

class GeneAnnotationOutput(BaseModel):
    """
    Output from gene functional annotation.
    """
    gene_info: Dict[str, Any] = Field(
        ...,
        description="Basic gene information"
    )
    functional_annotation: Dict[str, Any] = Field(
        ...,
        description="Functional annotations (GO terms, domains, etc.)"
    )
    protein_interactions: List[ProteinInteraction] = Field(
        default_factory=list,
        description="Protein-protein interactions"
    )
    disease_associations: List[DiseaseAssociation] = Field(
        default_factory=list,
        description="Disease associations"
    )
    interpretation: str = Field(
        ...,
        description="Functional interpretation of the gene"
    )

class DrugCompound(BaseModel):
    """
    Represents a drug compound.
    """
    drug_name: str = Field(..., description="Drug name")
    drug_id: str = Field(..., description="Drug identifier")
    mechanism_of_action: str = Field(..., description="Mechanism of action")
    development_stage: str = Field(..., description="Development stage")
    clinical_trials: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Clinical trial information"
    )
    source_database: str = Field(..., description="Source database")

class DrugTargetOutput(BaseModel):
    """
    Output from drug-target analysis.
    """
    target_info: Dict[str, Any] = Field(
        ...,
        description="Target gene/protein information"
    )
    targeting_drugs: List[DrugCompound] = Field(
        ...,
        description="Drugs targeting this gene/protein"
    )
    druggability_assessment: Dict[str, Any] = Field(
        ...,
        description="Assessment of druggability"
    )
    therapeutic_opportunities: str = Field(
        ...,
        description="Summary of therapeutic opportunities"
    )
