"""
MCP-eQTL Server

A Model Context Protocol server for querying eQTL data to find instrumental variables
for Mendelian Randomization analysis.
"""

__version__ = "0.1.0"

from .models import EQTLToolInput, SNPInstrument
from .main import mcp, find_eqtl_instruments, run

__all__ = ["EQTLToolInput", "SNPInstrument", "mcp", "find_eqtl_instruments", "run"]
