# client-app/mcp_client.py
"""
MCP Client Manager for coordinating with multiple MCP servers.
"""

import asyncio
import logging
import time
import os
from typing import Dict, List, Any, Optional
from datetime import datetime
import subprocess
import json

from models import ServerStatus, SystemStatus, AnalysisStep

logger = logging.getLogger(__name__)

class MCPClientManager:
    """
    Manages connections to multiple MCP servers and coordinates analysis workflows.
    """
    
    def __init__(self):
        """Initialize the MCP client manager."""
        # 获取项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        self.servers = {
            "eqtl": {
                "name": "mcp-server-eqtl",
                "command": ["python", "-m", "mcp_server_eqtl.main"],
                "cwd": os.path.join(project_root, "mcp-server-eqtl", "src"),
                "status": "offline",
                "process": None,
                "last_check": None
            },
            "gwas": {
                "name": "mcp-server-gwas",
                "command": ["python", "-m", "mcp_server_gwas.main"],
                "cwd": os.path.join(project_root, "mcp-server-gwas", "src"),
                "status": "offline",
                "process": None,
                "last_check": None
            },
            "mr": {
                "name": "mcp-server-mr",
                "command": ["python", "-m", "mcp_server_mr.main"],
                "cwd": os.path.join(project_root, "mcp-server-mr", "src"),
                "status": "offline",
                "process": None,
                "last_check": None
            },
            "knowledge": {
                "name": "mcp-server-knowledge",
                "command": ["python", "-m", "mcp_server_knowledge.main"],
                "cwd": os.path.join(project_root, "mcp-server-knowledge", "src"),
                "status": "offline",
                "process": None,
                "last_check": None
            }
        }
        
        # 系统配置MCP服务器
        self.simulation_mode = False
        
    async def start_servers(self) -> bool:
        """
        Start all MCP servers.
        
        Returns:
            bool: True if all servers started successfully
        """
        logger.info("Starting MCP servers...")
        
        if self.simulation_mode:
            # Simulate server startup
            for server_id, server_info in self.servers.items():
                server_info["status"] = "online"
                server_info["last_check"] = datetime.now().isoformat()
                logger.info(f"Simulated startup of {server_info['name']}")
            return True
        
        # Real server startup would go here
        success_count = 0
        for server_id, server_info in self.servers.items():
            try:
                # Start server process with correct working directory
                process = subprocess.Popen(
                    server_info["command"],
                    cwd=server_info.get("cwd"),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                server_info["process"] = process
                server_info["status"] = "online"
                server_info["last_check"] = datetime.now().isoformat()
                success_count += 1
                logger.info(f"Started {server_info['name']} in {server_info.get('cwd', 'current directory')}")

            except Exception as e:
                logger.error(f"Failed to start {server_info['name']}: {e}")
                server_info["status"] = "error"
                server_info["error_message"] = str(e)
        
        return success_count == len(self.servers)
    
    async def stop_servers(self):
        """Stop all MCP servers."""
        logger.info("Stopping MCP servers...")
        
        for server_id, server_info in self.servers.items():
            if server_info.get("process"):
                try:
                    server_info["process"].terminate()
                    server_info["process"].wait(timeout=5)
                    logger.info(f"Stopped {server_info['name']}")
                except Exception as e:
                    logger.error(f"Error stopping {server_info['name']}: {e}")
                    try:
                        server_info["process"].kill()
                    except:
                        pass
                finally:
                    server_info["status"] = "offline"
                    server_info["process"] = None
    
    async def check_server_status(self) -> SystemStatus:
        """
        Check the status of all MCP servers.
        
        Returns:
            SystemStatus: Current system status
        """
        server_statuses = []
        healthy_count = 0
        
        for server_id, server_info in self.servers.items():
            status = ServerStatus(
                server_name=server_info["name"],
                status=server_info["status"],
                last_check=server_info.get("last_check", "Never"),
                error_message=server_info.get("error_message")
            )
            server_statuses.append(status)
            
            if server_info["status"] == "online":
                healthy_count += 1
        
        # Determine overall system health
        if healthy_count == len(self.servers):
            system_health = "healthy"
        elif healthy_count > 0:
            system_health = "degraded"
        else:
            system_health = "down"
        
        return SystemStatus(
            servers=server_statuses,
            system_health=system_health,
            last_updated=datetime.now().isoformat()
        )
    
    async def call_eqtl_server(self, gene_symbol: str, tissue: str = "Whole_Blood") -> Dict[str, Any]:
        """
        Call the eQTL server to find instrumental variables.
        
        Args:
            gene_symbol: Gene symbol to query
            tissue: Tissue context
            
        Returns:
            Dict containing eQTL results
        """
        logger.info(f"Calling eQTL server for gene: {gene_symbol}")
        
        # 直接调用eQTL服务获取真实数据
        try:
            # 导入eQTL服务
            import sys
            import os
            eqtl_path = os.path.join(os.path.dirname(__file__), '..', 'mcp-server-eqtl', 'src')
            if eqtl_path not in sys.path:
                sys.path.insert(0, eqtl_path)
            from mcp_server_eqtl.main import RealEQTLDataClient

            # 创建eQTL客户端
            eqtl_client = RealEQTLDataClient()

            # 获取真实eQTL工具变量
            instruments = await eqtl_client.get_eqtl_instruments(gene_symbol, tissue, 5e-8)

            if instruments:
                # 转换为字典格式
                instruments_dict = []
                for instrument in instruments:
                    instruments_dict.append({
                        "snp_id": instrument.snp_id,
                        "effect_allele": instrument.effect_allele,
                        "other_allele": instrument.other_allele,
                        "beta": instrument.beta,
                        "se": instrument.se,
                        "p_value": instrument.p_value,
                        "source_db": instrument.source_db
                    })

                return {
                    "instruments": instruments_dict,
                    "summary": f"Found {len(instruments_dict)} real instruments for {gene_symbol} in {tissue}",
                    "gene_symbol": gene_symbol,
                    "tissue": tissue,
                    "data_source": "Real_GTEx_Data"
                }
            else:
                logger.warning(f"No real eQTL data found for {gene_symbol}")
                return {
                    "instruments": [],
                    "summary": f"No real eQTL data found for {gene_symbol} in {tissue}",
                    "gene_symbol": gene_symbol,
                    "tissue": tissue,
                    "data_source": "No_Real_Data_Available"
                }

        except Exception as e:
            logger.error(f"eQTL service call failed: {e}")
            return {
                "instruments": [],
                "summary": f"eQTL service failed: {str(e)}",
                "gene_symbol": gene_symbol,
                "tissue": tissue,
                "data_source": "Service_Error"
            }
    
    async def call_gwas_server(self, instruments: List[Dict], outcome_id: str) -> Dict[str, Any]:
        """
        Call the GWAS server for outcome data and harmonization.
        
        Args:
            instruments: List of instrumental variables from eQTL server
            outcome_id: GWAS outcome identifier
            
        Returns:
            Dict containing harmonized data
        """
        logger.info(f"Calling GWAS server for outcome: {outcome_id}")
        
        # 直接调用GWAS服务获取真实数据
        try:
            # 导入GWAS服务
            import sys
            import os
            gwas_path = os.path.join(os.path.dirname(__file__), '..', 'mcp-server-gwas', 'src')
            if gwas_path not in sys.path:
                sys.path.insert(0, gwas_path)
            from mcp_server_gwas.main import RealGWASDataClient
            from mcp_server_gwas.models import SNPInstrument

            # 转换instruments为SNPInstrument对象
            snp_instruments = []
            for instrument in instruments:
                snp_inst = SNPInstrument(
                    snp_id=instrument.get("snp_id", ""),
                    effect_allele=instrument.get("effect_allele", "A"),
                    other_allele=instrument.get("other_allele", "G"),
                    beta=instrument.get("beta", 0.0),
                    se=instrument.get("se", 0.1),
                    p_value=instrument.get("p_value", 1.0),
                    source_db=instrument.get("source_db", "Unknown")
                )
                snp_instruments.append(snp_inst)

            # 创建GWAS客户端
            gwas_client = RealGWASDataClient()

            # 获取SNP列表
            snp_list = [inst.snp_id for inst in snp_instruments]

            # 获取GWAS关联数据
            associations = await gwas_client.get_associations(outcome_id, snp_list)

            if associations:
                # 构建协调数据
                harmonized_data = []
                excluded_snps = []

                for instrument in snp_instruments:
                    snp_id = instrument.snp_id
                    if snp_id in associations:
                        outcome_data = associations[snp_id]

                        # 创建扁平化的协调数据点，包含MR分析所需的所有字段
                        harmonized_point = {
                            "snp_id": snp_id,
                            # 暴露数据（来自eQTL）
                            "beta_exposure": instrument.beta,
                            "se_exposure": instrument.se,
                            "p_value_exposure": instrument.p_value,
                            # 结局数据（来自GWAS）
                            "beta_outcome": outcome_data.get("beta"),
                            "se_outcome": outcome_data.get("se"),
                            "p_value_outcome": outcome_data.get("pval"),
                            # 等位基因信息
                            "effect_allele": instrument.effect_allele,
                            "other_allele": instrument.other_allele,
                            # 协调状态
                            "harmonization_status": "harmonized",
                            # 为了向后兼容，保留原始嵌套结构
                            "exposure_data": {
                                "beta": instrument.beta,
                                "se": instrument.se,
                                "p_value": instrument.p_value
                            },
                            "outcome_data": {
                                "beta": outcome_data.get("beta"),
                                "se": outcome_data.get("se"),
                                "pval": outcome_data.get("pval"),
                                "source": outcome_data.get("source", "Unknown"),
                                "study_id": outcome_id  # 添加研究ID信息
                            },
                            # 质量控制所需的字段（使用结局数据）
                            "beta": outcome_data.get("beta"),
                            "se": outcome_data.get("se"),
                            "p_value": outcome_data.get("pval")
                        }
                        harmonized_data.append(harmonized_point)
                    else:
                        excluded_snps.append(snp_id)

                return {
                    "harmonized_data": harmonized_data,
                    "summary": f"Successfully harmonized {len(harmonized_data)} out of {len(instruments)} SNPs using real GWAS data.",
                    "excluded_snps": excluded_snps,
                    "outcome_id": outcome_id,
                    "data_source": "Real_GWAS_Data"
                }
            else:
                logger.warning(f"No GWAS data found for outcome {outcome_id}")
                return {
                    "harmonized_data": [],
                    "summary": f"No real GWAS data available for outcome {outcome_id}",
                    "excluded_snps": [inst.snp_id for inst in snp_instruments],
                    "outcome_id": outcome_id,
                    "data_source": "No_Real_Data_Available"
                }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"GWAS service call failed: {error_msg}")

            # 分析错误类型并提供具体信息
            if "invalid syntax" in error_msg:
                detailed_error = "GWAS服务代码语法错误，这是技术问题，不是数据问题"
                error_type = "Code_Syntax_Error"
            elif "timeout" in error_msg.lower():
                detailed_error = "OpenGWAS API连接超时，这是网络技术问题"
                error_type = "Network_Timeout"
            elif "connection" in error_msg.lower():
                detailed_error = "无法连接到OpenGWAS API，这是网络技术问题"
                error_type = "Network_Connection_Error"
            elif "authentication" in error_msg.lower() or "401" in error_msg:
                detailed_error = "OpenGWAS API认证失败，JWT令牌可能无效或过期"
                error_type = "Authentication_Error"
            else:
                detailed_error = f"GWAS服务技术错误: {error_msg}"
                error_type = "Service_Error"

            return {
                "harmonized_data": [],
                "summary": detailed_error,
                "excluded_snps": [inst.get("snp_id", "") for inst in instruments],
                "outcome_id": outcome_id,
                "data_source": "Service_Error",
                "error_type": error_type,
                "technical_details": error_msg
            }
    
    async def call_mr_server(self, harmonized_data: List[Dict], exposure_name: str, outcome_name: str, language: str = "zh") -> Dict[str, Any]:
        """
        Call the MR server for causal analysis.
        
        Args:
            harmonized_data: Harmonized exposure-outcome data
            exposure_name: Name of exposure variable
            outcome_name: Name of outcome variable
            
        Returns:
            Dict containing MR analysis results
        """
        logger.info(f"Calling MR server for causal analysis: {exposure_name} -> {outcome_name}")
        
        # 直接调用MR服务进行真实计算
        try:
            # 导入MR服务
            import sys
            import os
            mr_path = os.path.join(os.path.dirname(__file__), '..', 'mcp-server-mr', 'src')
            if mr_path not in sys.path:
                sys.path.insert(0, mr_path)
            from mcp_server_mr.mr_analysis import MRAnalyzer
            from mcp_server_mr.models import HarmonizedDataPoint

            # 转换数据格式
            harmonized_points = []
            for data_point in harmonized_data:
                if 'exposure_data' in data_point and 'outcome_data' in data_point:
                    exp_data = data_point['exposure_data']
                    out_data = data_point['outcome_data']

                    point = HarmonizedDataPoint(
                        SNP=data_point.get('snp_id', 'unknown'),
                        beta_exposure=exp_data.get('beta', 0.0),
                        se_exposure=exp_data.get('se', 0.1),
                        pval_exposure=exp_data.get('p_value', 1.0),
                        beta_outcome=out_data.get('beta', 0.0),
                        se_outcome=out_data.get('se', 0.1),
                        pval_outcome=out_data.get('pval', 1.0),
                        effect_allele=data_point.get('effect_allele', 'A'),
                        other_allele=data_point.get('other_allele', 'G'),
                        harmonization_status=data_point.get('harmonization_status', 'harmonized'),
                        outcome_study_id=out_data.get('study_id', 'unknown')  # 添加研究ID
                    )
                    harmonized_points.append(point)

            if not harmonized_points:
                logger.warning("No valid harmonized data points for MR analysis")
                return {"error": "No valid data for MR analysis"}

            # 执行MR分析
            analyzer = MRAnalyzer(harmonized_points)
            mr_results, sensitivity_analysis = analyzer.perform_full_analysis()

            # 生成可视化图表
            from mcp_server_mr.visualization import MRVisualizer
            visualizer = MRVisualizer(harmonized_points)
            plots = visualizer.generate_all_plots(mr_results)

            logger.info(f"Generated {len(plots)} visualization plots")

            # 生成结论和详细解释
            conclusion = self._generate_mr_conclusion(mr_results, language)
            interpretation = self._generate_mr_interpretation(mr_results, sensitivity_analysis, exposure_name, outcome_name, len(harmonized_points), language)

            # 转换结果格式
            results_dict = {
                "summary": {
                    "exposure": exposure_name,
                    "outcome": outcome_name,
                    "n_snps": len(harmonized_points),
                    "data_source": "Real_MR_Calculation",
                    "conclusion": conclusion
                },
                "results": [
                    {
                        "method": result.method,
                        "estimate": result.estimate,
                        "se": result.se,
                        "ci_lower": result.ci_lower,
                        "ci_upper": result.ci_upper,
                        "p_value": result.p_value,
                        "n_snps": result.n_snps
                    } for result in mr_results
                ],
                "interpretation": interpretation,  # 添加详细解释
                "visualization": {
                    "scatter_plot": plots.get('scatter_plot'),
                    "forest_plot": plots.get('forest_plot'),
                    "funnel_plot": plots.get('funnel_plot')
                }
            }

            if sensitivity_analysis:
                results_dict["sensitivity_analysis"] = {
                    "heterogeneity_test": sensitivity_analysis.heterogeneity_test,
                    "pleiotropy_test": sensitivity_analysis.pleiotropy_test,
                    "leave_one_out": sensitivity_analysis.leave_one_out
                }

            logger.info(f"MR analysis completed with {len(mr_results)} methods")
            return results_dict

        except Exception as e:
            logger.error(f"MR analysis failed: {e}")
            return {"error": f"MR analysis failed: {str(e)}"}

    def _generate_mr_conclusion(self, mr_results, language: str = "zh") -> str:
        """
        Generate a conclusion based on MR results.
        """
        if not mr_results:
            return "无有效的MR分析结果" if language == "zh" else "No valid MR results obtained"

        # Focus on IVW result if available
        ivw_result = None
        for result in mr_results:
            if "Inverse Variance" in result.method:
                ivw_result = result
                break

        if not ivw_result:
            ivw_result = mr_results[0]  # Use first result as fallback

        if ivw_result.p_value < 0.05:
            if language == "zh":
                direction = "正向" if ivw_result.estimate > 0 else "负向"
                return f"强有力的证据支持{direction}因果效应 (β={ivw_result.estimate:.3f}, P={ivw_result.p_value:.3e})"
            else:
                direction = "positive" if ivw_result.estimate > 0 else "negative"
                return f"Strong evidence for a {direction} causal effect (β={ivw_result.estimate:.3f}, P={ivw_result.p_value:.3e})"
        else:
            if language == "zh":
                return f"未检测到显著的因果效应 (β={ivw_result.estimate:.3f}, P={ivw_result.p_value:.3f})"
            else:
                return f"No significant causal effect detected (β={ivw_result.estimate:.3f}, P={ivw_result.p_value:.3f})"

    def _generate_mr_interpretation(self, mr_results, sensitivity_analysis, exposure_name: str, outcome_name: str, n_snps: int, language: str = "zh") -> str:
        """
        Generate detailed interpretation of MR results using real data.
        """
        interpretation_parts = []

        # Header
        if language == "zh":
            interpretation_parts.append(
                f"## 孟德尔随机化分析结果\n"
                f"**暴露变量:** {exposure_name}\n"
                f"**结局变量:** {outcome_name}\n"
                f"**工具变量数量:** {n_snps}\n"
                f"**数据来源:** 真实eQTL和GWAS API数据\n"
            )
            # Main results
            interpretation_parts.append("### 主要结果")
        else:
            interpretation_parts.append(
                f"## Mendelian Randomization Analysis Results\n"
                f"**Exposure:** {exposure_name}\n"
                f"**Outcome:** {outcome_name}\n"
                f"**Number of instruments:** {n_snps}\n"
                f"**Data Source:** Real eQTL and GWAS API data\n"
            )
            # Main results
            interpretation_parts.append("### Main Results")

        if mr_results:
            for result in mr_results:
                if language == "zh":
                    significance = "显著" if result.p_value < 0.05 else "非显著"
                    direction = "正向" if result.estimate > 0 else "负向"
                    interpretation_parts.append(
                        f"- **{result.method}:** {direction}{significance}效应 "
                        f"(β = {result.estimate:.3f}, 95% CI: {result.ci_lower:.3f} 至 {result.ci_upper:.3f}, "
                        f"P = {result.p_value:.3e})"
                    )
                else:
                    significance = "significant" if result.p_value < 0.05 else "non-significant"
                    direction = "positive" if result.estimate > 0 else "negative"
                    interpretation_parts.append(
                        f"- **{result.method}:** {direction} {significance} effect "
                        f"(β = {result.estimate:.3f}, 95% CI: {result.ci_lower:.3f} to {result.ci_upper:.3f}, "
                        f"P = {result.p_value:.3e})"
                    )

        # Sensitivity analyses
        if language == "zh":
            interpretation_parts.append("\n### 敏感性分析")
        else:
            interpretation_parts.append("\n### Sensitivity Analysis")

        if sensitivity_analysis:
            if hasattr(sensitivity_analysis, 'heterogeneity_test') and sensitivity_analysis.heterogeneity_test:
                het_test = sensitivity_analysis.heterogeneity_test
                if isinstance(het_test, dict) and 'p_value' in het_test:
                    if language == "zh":
                        het_status = "无显著异质性" if het_test['p_value'] > 0.05 else "存在异质性"
                        interpretation_parts.append(f"- **异质性检验:** {het_status} (P = {het_test['p_value']:.3f})")
                    else:
                        het_status = "no significant heterogeneity" if het_test['p_value'] > 0.05 else "heterogeneity detected"
                        interpretation_parts.append(f"- **Heterogeneity test:** {het_status} (P = {het_test['p_value']:.3f})")

            if hasattr(sensitivity_analysis, 'pleiotropy_test') and sensitivity_analysis.pleiotropy_test:
                pleio_test = sensitivity_analysis.pleiotropy_test
                if isinstance(pleio_test, dict) and 'p_value' in pleio_test:
                    if language == "zh":
                        pleio_status = "无显著多效性" if pleio_test['p_value'] > 0.05 else "存在多效性"
                        interpretation_parts.append(f"- **多效性检验:** {pleio_status} (P = {pleio_test['p_value']:.3f})")
                    else:
                        pleio_status = "no significant pleiotropy" if pleio_test['p_value'] > 0.05 else "pleiotropy detected"
                        interpretation_parts.append(f"- **Pleiotropy test:** {pleio_status} (P = {pleio_test['p_value']:.3f})")

        # Overall interpretation
        if language == "zh":
            interpretation_parts.append("\n### 总体解释")
        else:
            interpretation_parts.append("\n### Overall Interpretation")

        if mr_results:
            # Check consistency across methods
            significant_results = [r for r in mr_results if r.p_value < 0.05]

            if len(significant_results) >= 2:
                # Check if effects are in same direction
                estimates = [r.estimate for r in significant_results]
                same_direction = all(e > 0 for e in estimates) or all(e < 0 for e in estimates)

                if same_direction:
                    if language == "zh":
                        interpretation_parts.append(
                            "结果提供了**强有力的证据**支持因果关系。"
                            "多种MR方法显示一致的、显著的同向效应。"
                        )
                    else:
                        interpretation_parts.append(
                            "Results provide **strong evidence** for a causal relationship. "
                            "Multiple MR methods show consistent, significant effects in the same direction."
                        )
                else:
                    if language == "zh":
                        interpretation_parts.append(
                            "结果显示**混合证据**。虽然多种方法显示显著效应，"
                            "但效应方向不一致，提示可能存在多效性或其他偏倚。"
                        )
                    else:
                        interpretation_parts.append(
                            "Results show **mixed evidence**. While multiple methods show significant effects, "
                            "the directions are inconsistent, suggesting possible pleiotropy or other biases."
                        )
            elif len(significant_results) == 1:
                if language == "zh":
                    interpretation_parts.append(
                        "结果提供了**中等程度的证据**支持因果关系。"
                        "一种MR方法显示显著效应，但建议进行额外验证。"
                    )
                else:
                    interpretation_parts.append(
                        "Results provide **moderate evidence** for a causal relationship. "
                        "One MR method shows significant effect, but additional validation is recommended."
                    )
            else:
                if language == "zh":
                    interpretation_parts.append(
                        "结果提供了**有限的证据**支持因果关系。"
                        "在常规α = 0.05水平下，没有MR方法显示显著效应。"
                    )
                else:
                    interpretation_parts.append(
                        "Results provide **limited evidence** for a causal relationship. "
                        "No MR methods show significant effects at the conventional α = 0.05 level."
                    )

        # Limitations
        if language == "zh":
            interpretation_parts.append(
                "\n### 局限性和注意事项"
                "\n- MR分析假设遗传工具变量满足三个核心假设"
                "\n- 结果应在生物学合理性的背景下解释"
                "\n- 建议通过独立数据集进行额外验证"
                "\n- 考虑潜在的人群分层和连锁不平衡效应"
            )
        else:
            interpretation_parts.append(
                "\n### Limitations and Considerations"
                "\n- MR analysis assumes genetic instruments satisfy three core assumptions"
                "\n- Results should be interpreted in the context of biological plausibility"
                "\n- Additional validation through independent datasets is recommended"
                "\n- Consider potential population stratification and linkage disequilibrium effects"
            )

        return "\n".join(interpretation_parts)

    async def call_knowledge_server(self, gene_symbol: str, analysis_type: str = "gene_annotation") -> Dict[str, Any]:
        """
        Call the knowledge server for biological insights - 只使用真实API数据

        Args:
            gene_symbol: Gene symbol to analyze
            analysis_type: Type of analysis (gene_annotation, pathway_analysis, drug_targets)

        Returns:
            Dict containing knowledge analysis results
        """
        logger.info(f"Calling knowledge server for {analysis_type} on gene: {gene_symbol}")

        try:
            # 使用集成的真实知识服务
            from real_knowledge_service import RealKnowledgeService

            knowledge_service = RealKnowledgeService()

            if analysis_type == "gene_annotation":
                result = await knowledge_service.get_gene_annotation(gene_symbol)
                logger.info(f"✅ 基因注释获取完成，数据来源: {result.get('data_source', 'Unknown')}")
                return result
            else:
                logger.warning(f"Analysis type {analysis_type} not yet supported")
                return {
                    "error": f"Analysis type {analysis_type} not supported",
                    "data_source": "Service_Error",
                    "message": f"分析类型 {analysis_type} 暂不支持"
                }

        except Exception as e:
            logger.error(f"❌ Knowledge server call failed: {e}")
            return {
                "error": f"Knowledge server call failed: {str(e)}",
                "data_source": "Service_Error",
                "message": "知识服务器调用失败"
            }

    async def call_knowledge_server_drug(self, gene_symbol: str) -> Dict[str, Any]:
        """
        调用知识服务器进行药物分析 - 只使用真实API数据

        Args:
            gene_symbol: 基因符号

        Returns:
            包含药物分析结果的字典
        """
        logger.info(f"Calling knowledge server for drug analysis on gene: {gene_symbol}")

        try:
            # 使用集成的真实知识服务
            from real_knowledge_service import RealKnowledgeService

            knowledge_service = RealKnowledgeService()
            result = await knowledge_service.get_drug_targets(gene_symbol)

            logger.info(f"✅ 药物分析完成，数据来源: {result.get('data_source', 'Unknown')}")
            return result

        except Exception as e:
            logger.error(f"❌ Drug analysis failed: {e}")
            return {
                "error": f"Drug analysis failed: {str(e)}",
                "data_source": "Service_Error",
                "message": "药物分析服务调用失败"
            }
