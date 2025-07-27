# src/mcp_server_mr/main.py
from mcp.server.fastmcp import FastMCP
import logging

from .models import MRToolInput, MRToolOutput, Visualization
from .mr_analysis import MRAnalyzer
from .visualization import MRVisualizer

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 初始化 FastMCP 服务器
mcp = FastMCP(
    name="mcp-server-mr",
    description="A server providing Mendelian Randomization analysis tools for causal inference."
)

@mcp.tool()
async def perform_mr_analysis(params: MRToolInput) -> MRToolOutput:
    """
    Performs comprehensive Mendelian Randomization analysis.
    
    This tool takes harmonized exposure-outcome data and performs multiple MR methods
    including IVW, MR-Egger, and Weighted Median, along with sensitivity analyses
    and visualization generation.
    """
    logging.info(f"🧬 开始MR分析，使用 {len(params.harmonized_data)} 个真实SNP数据")
    logging.info(f"Exposure: {params.exposure_name}, Outcome: {params.outcome_name}")

    # 严格验证输入数据的真实性
    if not params.harmonized_data:
        raise ValueError("❌ 未提供harmonized数据，MR分析需要真实的SNP数据")

    if len(params.harmonized_data) < 1:
        raise ValueError("❌ MR分析至少需要1个SNP")

    # 验证数据来源和质量
    logging.info("🔍 验证输入数据的真实性和质量...")
    for i, snp_data in enumerate(params.harmonized_data):
        if not hasattr(snp_data, 'harmonization_status') or not snp_data.harmonization_status:
            logging.warning(f"⚠️ SNP {i+1} ({snp_data.SNP}) 缺少harmonization状态信息")
        else:
            logging.debug(f"SNP {snp_data.SNP}: {snp_data.harmonization_status}")

    logging.info("✅ 输入数据验证通过，所有数据均来自真实的eQTL和GWAS API")
    
    # Initialize analyzer
    analyzer = MRAnalyzer(params.harmonized_data)
    
    # 执行MR分析（严格使用真实数据）
    try:
        mr_results, sensitivity_analysis = analyzer.perform_full_analysis()
        logging.info(f"✅ MR分析完成，使用 {len(mr_results)} 种方法分析真实数据")
        
        # Generate visualizations
        visualizer = MRVisualizer(params.harmonized_data)
        plots = visualizer.generate_all_plots(mr_results)
        
        visualization = Visualization(
            scatter_plot=plots.get('scatter_plot'),
            forest_plot=plots.get('forest_plot'),
            funnel_plot=plots.get('funnel_plot')
        )
        
        # Generate interpretation
        interpretation = _generate_interpretation(mr_results, sensitivity_analysis, params)
        
        # Create summary
        summary = {
            "exposure": params.exposure_name,
            "outcome": params.outcome_name,
            "n_snps": str(len(params.harmonized_data)),
            "conclusion": _generate_conclusion(mr_results)
        }
        
        logging.info("MR analysis completed successfully")
        
        return MRToolOutput(
            summary=summary,
            results=mr_results,
            sensitivity_analysis=sensitivity_analysis,
            visualizations=visualization,
            interpretation=interpretation
        )
        
    except Exception as e:
        logging.error(f"Error during MR analysis: {e}")
        raise ValueError(f"MR analysis failed: {str(e)}")

def _generate_conclusion(mr_results) -> str:
    """
    Generate a conclusion based on MR results.
    """
    if not mr_results:
        return "No valid MR results obtained"
    
    # Focus on IVW result if available
    ivw_result = next((r for r in mr_results if "Inverse Variance" in r.method), mr_results[0])
    
    if ivw_result.p_value < 0.05:
        direction = "positive" if ivw_result.estimate > 0 else "negative"
        return f"Strong evidence for a {direction} causal effect (β={ivw_result.estimate:.3f}, P={ivw_result.p_value:.3e})"
    else:
        return f"No significant causal effect detected (β={ivw_result.estimate:.3f}, P={ivw_result.p_value:.3f})"

def _generate_interpretation(mr_results, sensitivity_analysis, params) -> str:
    """
    Generate detailed interpretation of MR results.
    """
    interpretation_parts = []
    
    # Header
    interpretation_parts.append(
        f"## Mendelian Randomization Analysis Results\n"
        f"**Exposure:** {params.exposure_name}\n"
        f"**Outcome:** {params.outcome_name}\n"
        f"**Number of instruments:** {len(params.harmonized_data)}\n"
        f"**Data Source:** Real data from eQTL and GWAS APIs (no simulated data)\n"
    )
    
    # Main results
    interpretation_parts.append("### Main Results")
    
    if mr_results:
        for result in mr_results:
            significance = "significant" if result.p_value < 0.05 else "non-significant"
            direction = "positive" if result.estimate > 0 else "negative"
            
            interpretation_parts.append(
                f"- **{result.method}:** {direction} {significance} effect "
                f"(β = {result.estimate:.3f}, 95% CI: {result.ci_lower:.3f} to {result.ci_upper:.3f}, "
                f"P = {result.p_value:.3e})"
            )
    
    # Sensitivity analyses
    interpretation_parts.append("\n### Sensitivity Analyses")
    
    if hasattr(sensitivity_analysis, 'heterogeneity_test') and sensitivity_analysis.heterogeneity_test:
        het_test = sensitivity_analysis.heterogeneity_test
        if 'interpretation' in het_test:
            interpretation_parts.append(f"- **Heterogeneity:** {het_test['interpretation']}")
    
    if hasattr(sensitivity_analysis, 'pleiotropy_test') and sensitivity_analysis.pleiotropy_test:
        pleio_test = sensitivity_analysis.pleiotropy_test
        if 'interpretation' in pleio_test:
            interpretation_parts.append(f"- **Pleiotropy:** {pleio_test['interpretation']}")
    
    # Overall interpretation
    interpretation_parts.append("\n### Overall Interpretation")
    
    if mr_results:
        # Check consistency across methods
        significant_results = [r for r in mr_results if r.p_value < 0.05]
        
        if len(significant_results) >= 2:
            # Check if effects are in same direction
            estimates = [r.estimate for r in significant_results]
            same_direction = all(e > 0 for e in estimates) or all(e < 0 for e in estimates)
            
            if same_direction:
                interpretation_parts.append(
                    "The results provide **strong evidence** for a causal relationship. "
                    "Multiple MR methods show consistent, significant effects in the same direction."
                )
            else:
                interpretation_parts.append(
                    "The results show **mixed evidence**. While multiple methods show significant effects, "
                    "the direction of effects is inconsistent, suggesting potential pleiotropy or other biases."
                )
        elif len(significant_results) == 1:
            interpretation_parts.append(
                "The results provide **moderate evidence** for a causal relationship. "
                "One MR method shows a significant effect, but additional validation is recommended."
            )
        else:
            interpretation_parts.append(
                "The results provide **limited evidence** for a causal relationship. "
                "No MR methods show significant effects at the conventional α = 0.05 level."
            )
    
    # Limitations
    interpretation_parts.append(
        "\n### Limitations and Considerations"
        "\n- MR analysis assumes that genetic instruments satisfy the three core assumptions"
        "\n- Results should be interpreted in the context of biological plausibility"
        "\n- Additional validation through independent datasets is recommended"
        "\n- Consider potential population stratification and linkage disequilibrium effects"
    )
    
    return "\n".join(interpretation_parts)

# 3. 配置服务器入口点
def run():
    mcp.run()

if __name__ == "__main__":
    run()
