# src/mcp_server_mr/models.py
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import base64

# --- 输入模型 ---
class HarmonizedDataPoint(BaseModel):
    """
    Represents a single SNP with its effects on both exposure and outcome,
    ready for MR analysis.
    """
    SNP: str
    beta_exposure: float
    se_exposure: float
    pval_exposure: float
    beta_outcome: float
    se_outcome: float
    pval_outcome: float
    effect_allele: str
    other_allele: str
    harmonization_status: str
    outcome_study_id: Optional[str] = Field(default="unknown", description="GWAS study ID used for outcome data")

class MRToolInput(BaseModel):
    """
    Defines the parameters for the 'perform_mr_analysis' tool.
    """
    harmonized_data: List[HarmonizedDataPoint] = Field(
        ...,
        description="List of harmonized SNP data points from the GWAS server."
    )
    exposure_name: Optional[str] = Field(
        "Gene Expression",
        description="Name of the exposure variable for reporting."
    )
    outcome_name: Optional[str] = Field(
        "Disease Outcome", 
        description="Name of the outcome variable for reporting."
    )

# --- 输出模型 ---
class MRMethodResult(BaseModel):
    """
    Results from a single MR method.
    """
    method: str = Field(..., description="Name of the MR method")
    estimate: float = Field(..., description="Causal effect estimate")
    se: float = Field(..., description="Standard error of the estimate")
    ci_lower: float = Field(..., description="Lower bound of 95% confidence interval")
    ci_upper: float = Field(..., description="Upper bound of 95% confidence interval")
    p_value: float = Field(..., description="P-value for the causal effect")
    n_snps: int = Field(..., description="Number of SNPs used in the analysis")

class SensitivityAnalysis(BaseModel):
    """
    Results from sensitivity analyses.
    """
    heterogeneity_test: Dict[str, Any] = Field(
        default_factory=dict,
        description="Cochran's Q test for heterogeneity"
    )
    pleiotropy_test: Dict[str, Any] = Field(
        default_factory=dict,
        description="MR-Egger intercept test for pleiotropy"
    )
    leave_one_out: Dict[str, Any] = Field(
        default_factory=dict,
        description="Leave-one-out analysis results"
    )

class Visualization(BaseModel):
    """
    Visualization data for MR results.
    """
    scatter_plot: Optional[str] = Field(
        None,
        description="Base64-encoded scatter plot image"
    )
    forest_plot: Optional[str] = Field(
        None,
        description="Base64-encoded forest plot image"
    )
    funnel_plot: Optional[str] = Field(
        None,
        description="Base64-encoded funnel plot image"
    )

class MRToolOutput(BaseModel):
    """
    Complete output from MR analysis.
    """
    summary: Dict[str, str] = Field(
        ...,
        description="Summary of the analysis including exposure, outcome, and conclusion"
    )
    results: List[MRMethodResult] = Field(
        ...,
        description="Results from different MR methods"
    )
    sensitivity_analysis: SensitivityAnalysis = Field(
        ...,
        description="Results from sensitivity analyses"
    )
    visualizations: Visualization = Field(
        ...,
        description="Generated plots and visualizations"
    )
    interpretation: str = Field(
        ...,
        description="Interpretation of the results in plain language"
    )
