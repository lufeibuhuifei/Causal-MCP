"""
MCP-MR Server

A Model Context Protocol server for performing Mendelian Randomization analysis
for causal inference in biological systems.
"""

__version__ = "0.1.0"

from .models import (
    MRToolInput, 
    MRToolOutput, 
    MRMethodResult, 
    SensitivityAnalysis, 
    Visualization,
    HarmonizedDataPoint
)
from .main import mcp, perform_mr_analysis, run
from .mr_analysis import MRAnalyzer
from .visualization import MRVisualizer

__all__ = [
    "MRToolInput",
    "MRToolOutput", 
    "MRMethodResult",
    "SensitivityAnalysis",
    "Visualization",
    "HarmonizedDataPoint",
    "mcp",
    "perform_mr_analysis", 
    "run",
    "MRAnalyzer",
    "MRVisualizer"
]
