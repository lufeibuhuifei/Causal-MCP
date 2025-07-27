# src/mcp_server_eqtl/models.py
from pydantic import BaseModel, Field
from typing import List, Optional

# --- Tool Input Model (作为函数参数的类型提示) ---
class EQTLToolInput(BaseModel):
    """
    Defines the parameters for the 'find_eqtl_instruments' tool.
    This model is used for data validation and schema generation.
    """
    gene_symbol: str = Field(
        ...,
        description="The official HGNC symbol of the gene to query.",
        examples=["PCSK9", "LPA"]
    )
    tissue: Optional[str] = Field(
        "Whole_Blood",
        description="The tissue context for eQTLs. If not specified, defaults to a common tissue.",
        examples=["Liver", "Brain_Cortex"]
    )
    significance_threshold: float = Field(
        5e-8,
        description="The P-value threshold to select significant SNPs.",
        gt=0,
        le=1
    )

# --- Tool Output Model (作为函数返回值的类型提示) ---
class SNPInstrument(BaseModel):
    """
    Represents a single SNP that can be used as an instrumental variable.
    """
    snp_id: str = Field(..., description="The rsID of the SNP.", examples=["rs12345"])
    effect_allele: str = Field(..., description="The allele associated with a change in expression.", examples=["A"])
    other_allele: str = Field(..., description="The other allele.", examples=["G"])
    beta: float = Field(..., description="The effect size of the SNP on gene expression.")
    se: float = Field(..., description="The standard error of the effect size.")
    p_value: float = Field(..., description="The P-value of the association.")
    source_db: str = Field(..., description="The database source of this eQTL.", examples=["eQTLGen_2019"])

# FastMCP会自动将返回的List[SNPInstrument]包装在标准的成功响应中。
# 如果函数中引发异常，FastMCP会自动捕获并包装成标准的错误响应。
