"""
MCP-GWAS Server

A Model Context Protocol server for fetching GWAS outcome data and performing
data harmonization for Mendelian Randomization analysis.
"""

__version__ = "0.1.0"

from .models import GWASToolInput, GWASToolOutput, HarmonizedDataPoint, SNPInstrument
from .main import mcp, fetch_gwas_outcomes, run
from .harmonize import harmonize_datasets, harmonize_snp_data

__all__ = [
    "GWASToolInput", 
    "GWASToolOutput", 
    "HarmonizedDataPoint", 
    "SNPInstrument",
    "mcp", 
    "fetch_gwas_outcomes", 
    "run",
    "harmonize_datasets",
    "harmonize_snp_data"
]
