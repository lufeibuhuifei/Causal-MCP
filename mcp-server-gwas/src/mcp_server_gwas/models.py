# src/mcp_server_gwas/models.py
from pydantic import BaseModel, Field
from typing import List

# --- 输入模型 ---
# 我们需要复用来自eQTL服务器的SNPInstrument模型
# 在实际项目中，这可以做成一个共享的库
class SNPInstrument(BaseModel):
    snp_id: str
    effect_allele: str
    other_allele: str
    beta: float
    se: float
    p_value: float
    source_db: str

class GWASToolInput(BaseModel):
    """
    Defines the parameters for the 'fetch_gwas_outcomes' tool.
    """
    # 接收来自eQTL服务器的完整数据
    exposure_instruments: List[SNPInstrument] = Field(
        ...,
        description="The list of SNP instruments from the eQTL server, used for harmonization."
    )
    # 使用ID而非名称，以保证精确性和可重复性
    outcome_id: str = Field(
        ...,
        description="The unique identifier for the GWAS study of the outcome (e.g., from IEU OpenGWAS).",
        examples=["ieu-a-7"]  # ieu-a-7 is Coronary Artery Disease
    )

# --- 输出模型 ---
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
    
    # 增加一个字段来表示和谐化操作
    harmonization_status: str = Field(..., examples=["Harmonized: alleles matched.", "Harmonized: alleles flipped."])

    # 添加研究ID信息
    outcome_study_id: str = Field(default="unknown", description="GWAS study ID used for outcome data")

class GWASToolOutput(BaseModel):
    """
    Output from the GWAS tool containing harmonized data ready for MR analysis.
    """
    harmonized_data: List[HarmonizedDataPoint] = Field(
        ...,
        description="List of harmonized SNP data points ready for MR analysis."
    )
    summary: str = Field(
        ...,
        description="Summary of the harmonization process.",
        examples=["Successfully harmonized 40 out of 42 SNPs."]
    )
    excluded_snps: List[str] = Field(
        default_factory=list,
        description="List of SNP IDs that were excluded during harmonization."
    )
