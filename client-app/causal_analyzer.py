# client-app/causal_analyzer.py
"""
Main causal analysis coordinator using the Causal-MCP framework.
"""

import asyncio
import logging
import time
import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime

# 确保当前目录在路径中，优先于其他路径
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from models import (
    CausalAnalysisRequest, CausalAnalysisResult, AnalysisStep, AnalysisType,
    PathwayAnalysisRequest, GeneAnnotationRequest, DrugTargetRequest
)
from mcp_client import MCPClientManager
from llm_service import LLMService
from data_quality_control import DataQualityController, DataQualityLevel
from disease_mapper import DiseaseMapper
try:
    from i18n import get_text
except ImportError:
    # 如果在其他目录运行，尝试相对导入
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from i18n import get_text

logger = logging.getLogger(__name__)

class CausalAnalyzer:
    """
    Main class for coordinating causal analysis using multiple MCP servers.
    """
    
    def __init__(self):
        """Initialize the causal analyzer."""
        self.mcp_client = MCPClientManager()
        self.llm_service = LLMService()
        self.quality_controller = DataQualityController()
        # 初始化智能疾病映射系统
        self.disease_mapper = DiseaseMapper()
    
    async def initialize(self) -> bool:
        """
        Initialize the analyzer by starting MCP servers and LLM service.

        Returns:
            bool: True if initialization successful
        """
        logger.info("Initializing Causal-MCP framework...")

        # 初始化MCP服务器
        mcp_success = await self.mcp_client.start_servers()

        # 初始化LLM服务
        llm_success = await self.llm_service.initialize()

        if mcp_success:
            logger.info("✅ MCP servers initialized successfully")
        else:
            logger.error("❌ Failed to initialize MCP servers")

        if llm_success:
            logger.info("✅ LLM service initialized successfully")
        else:
            logger.warning("⚠️ LLM service initialization failed, using fallback methods")

        # 只要MCP服务器成功就认为初始化成功，LLM是增强功能
        success = mcp_success

        if success:
            logger.info("🎉 Causal-MCP framework initialized successfully")
        else:
            logger.error("❌ Failed to initialize Causal-MCP framework")

        return success
    
    async def shutdown(self):
        """Shutdown the analyzer and stop MCP servers."""
        logger.info("Shutting down Causal-MCP framework...")
        await self.mcp_client.stop_servers()

    async def reinitialize_llm(self) -> bool:
        """
        重新初始化LLM服务（在配置更改后调用）

        Returns:
            bool: 是否初始化成功
        """
        logger.info("重新初始化LLM服务...")
        return await self.llm_service.initialize()

    def update_llm_config(self, new_config: Dict[str, Any]) -> tuple[bool, str]:
        """
        更新LLM配置

        Args:
            new_config: 新的LLM配置

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        return self.llm_service.update_config(new_config)

    def get_llm_service(self) -> 'LLMService':
        """获取LLM服务实例"""
        return self.llm_service

    async def perform_full_causal_analysis(self, request: CausalAnalysisRequest) -> CausalAnalysisResult:
        """
        Perform complete causal analysis workflow.
        
        Args:
            request: Analysis request parameters
            
        Returns:
            CausalAnalysisResult: Complete analysis results
        """
        logger.info(f"Starting full causal analysis: {request.exposure_gene} -> {request.outcome_trait}")
        
        start_time = time.time()
        warnings = []

        # Initialize result structure
        result = CausalAnalysisResult(
            request=request,
            analysis_steps=[],
            summary={},
            interpretation="",
            recommendations=[],
            total_execution_time=0.0,
            success=False,
            warnings=warnings
        )
        
        try:
            # Step 1: Find eQTL instruments
            step1 = AnalysisStep(
                step_name="Find eQTL Instruments",
                status="running",
                server_used="mcp-server-eqtl"
            )
            result.analysis_steps.append(step1)
            
            step1_start = time.time()
            eqtl_result = await self.mcp_client.call_eqtl_server(
                request.exposure_gene, 
                request.tissue_context
            )
            step1.execution_time = time.time() - step1_start
            step1.status = "completed"
            step1.output_data = eqtl_result
            result.eqtl_instruments = eqtl_result.get("instruments", [])
            
            if not result.eqtl_instruments:
                raise ValueError(f"No eQTL instruments found for {request.exposure_gene}")

            # Data Quality Control for eQTL data
            logger.info(f"Validating {len(result.eqtl_instruments)} eQTL instruments")
            eqtl_data_for_validation = {
                'instruments': result.eqtl_instruments,
                'data_source': 'Real_GTEx_Data'
            }
            eqtl_quality = self.quality_controller.validate_eqtl_data(eqtl_data_for_validation)

            # 检查质量是否可接受
            if not self.quality_controller.is_quality_acceptable(eqtl_quality):
                raise ValueError(f"eQTL data quality insufficient: {eqtl_quality.overall_score:.3f}")

            quality_level = self.quality_controller.get_quality_level(eqtl_quality)
            if quality_level == DataQualityLevel.POOR:
                warnings.append(f"Low eQTL data quality (score: {eqtl_quality.overall_score:.3f})")

            logger.info(f"eQTL quality: {quality_level.value} ({len(result.eqtl_instruments)} instruments retained)")
            
            # Step 2: Get GWAS outcomes and harmonize data
            step2 = AnalysisStep(
                step_name="Fetch GWAS Outcomes & Harmonize",
                status="running",
                server_used="mcp-server-gwas"
            )
            result.analysis_steps.append(step2)
            
            # 使用智能疾病映射系统
            outcome_id = self.disease_mapper.get_study_id_for_disease(request.outcome_trait)
            if not outcome_id:
                # 验证输入并获取推荐
                validation_result = self.disease_mapper.validate_input(request.outcome_trait)
                error_msg = f"无法找到疾病 '{request.outcome_trait}' 对应的GWAS研究"
                if validation_result["recommendations"]:
                    error_msg += f"\n推荐的疾病名称: {', '.join(validation_result['recommendations'][:5])}"

                step2.status = "failed"
                step2.error_message = error_msg
                result.analysis_steps.append(step2)
                result.status = "failed"
                result.error_message = error_msg
                return result
            
            step2_start = time.time()
            gwas_result = await self.mcp_client.call_gwas_server(
                result.eqtl_instruments,
                outcome_id
            )
            step2.execution_time = time.time() - step2_start
            step2.status = "completed"
            step2.output_data = gwas_result
            result.harmonized_data = gwas_result.get("harmonized_data", [])
            
            if not result.harmonized_data:
                # 检查GWAS结果中的错误信息
                error_type = gwas_result.get("error_type", "Unknown")
                technical_details = gwas_result.get("technical_details", "")
                summary = gwas_result.get("summary", "No harmonized data available for MR analysis")

                # 构建详细的错误信息
                if error_type == "Code_Syntax_Error":
                    error_message = f"技术错误: {summary}。请联系技术支持修复代码问题。"
                elif error_type == "Network_Timeout":
                    error_message = f"网络超时: {summary}。请检查网络连接或稍后重试。"
                elif error_type == "Network_Connection_Error":
                    error_message = f"网络连接错误: {summary}。请检查网络设置。"
                elif error_type == "Authentication_Error":
                    error_message = f"认证错误: {summary}。请检查API令牌配置。"
                else:
                    error_message = f"数据获取失败: {summary}"

                # 添加技术细节（如果有）
                if technical_details:
                    error_message += f"\n技术细节: {technical_details}"

                raise ValueError(error_message)

            # Data Quality Control for harmonized data
            logger.info(f"Validating {len(result.harmonized_data)} harmonized data points")
            validated_harmonized, harmonized_quality = self.quality_controller.validate_harmonized_data(result.harmonized_data)
            result.harmonized_data = validated_harmonized

            # 获取质量等级
            quality_level = self.quality_controller.get_quality_level(harmonized_quality)

            if quality_level == DataQualityLevel.POOR:
                raise ValueError(f"Harmonized data quality insufficient: {harmonized_quality.overall_score:.3f}")
            elif quality_level == DataQualityLevel.FAIR:
                warnings.append(f"Fair harmonized data quality (score: {harmonized_quality.overall_score:.3f})")

            logger.info(f"Harmonized data quality: {quality_level.value} ({len(validated_harmonized)} SNPs retained)")
            
            # Step 3: Perform Mendelian Randomization analysis
            step3 = AnalysisStep(
                step_name="Mendelian Randomization Analysis",
                status="running",
                server_used="mcp-server-mr"
            )
            result.analysis_steps.append(step3)
            
            step3_start = time.time()
            mr_result = await self.mcp_client.call_mr_server(
                result.harmonized_data,
                f"{request.exposure_gene} Expression",
                request.outcome_trait,
                request.language if hasattr(request, 'language') else "zh"
            )
            step3.execution_time = time.time() - step3_start
            step3.status = "completed"
            step3.output_data = mr_result
            result.mr_results = mr_result
            
            logger.info("MR analysis completed")
            
            # Step 4: Gene annotation (if requested)
            if request.include_pathway_analysis:
                step4 = AnalysisStep(
                    step_name="Gene Functional Annotation",
                    status="running",
                    server_used="mcp-server-knowledge"
                )
                result.analysis_steps.append(step4)
                
                step4_start = time.time()
                annotation_result = await self.mcp_client.call_knowledge_server(
                    request.exposure_gene,
                    "gene_annotation"
                )
                step4.execution_time = time.time() - step4_start
                step4.status = "completed"
                step4.output_data = annotation_result
                result.gene_annotation = annotation_result
                
                logger.info("Gene annotation completed")
            
            # Step 5: Drug target analysis (if requested)
            if request.include_drug_analysis:
                step5 = AnalysisStep(
                    step_name="Drug Target Analysis",
                    status="running",
                    server_used="mcp-server-knowledge"
                )
                result.analysis_steps.append(step5)
                
                step5_start = time.time()
                drug_result = await self.mcp_client.call_knowledge_server_drug(
                    request.exposure_gene
                )
                step5.execution_time = time.time() - step5_start
                step5.status = "completed"
                step5.output_data = drug_result
                result.drug_analysis = drug_result
                
                logger.info("Drug target analysis completed")
            
            # Generate summary and interpretation
            result.summary = self._generate_summary(result)
            result.interpretation = await self._generate_interpretation(
                result,
                request.language if hasattr(request, 'language') else "zh",
                request.show_thinking if hasattr(request, 'show_thinking') else False
            )
            result.recommendations = await self._generate_recommendations(
                result,
                request.language if hasattr(request, 'language') else "zh",
                request.show_thinking if hasattr(request, 'show_thinking') else False
            )
            
            result.success = True
            logger.info("Full causal analysis completed successfully")
            
        except Exception as e:
            logger.error(f"Error in causal analysis: {e}")
            
            # Mark current step as failed
            if result.analysis_steps and result.analysis_steps[-1].status == "running":
                result.analysis_steps[-1].status = "failed"
                result.analysis_steps[-1].error_message = str(e)
            
            result.interpretation = f"Analysis failed: {str(e)}"
            result.recommendations = ["Review input parameters and try again"]
            result.success = False
        
        finally:
            result.total_execution_time = time.time() - start_time
        
        return result
    
    def _generate_summary(self, result: CausalAnalysisResult) -> Dict[str, Any]:
        """Generate analysis summary."""
        summary = {
            "exposure_gene": result.request.exposure_gene,
            "outcome_trait": result.request.outcome_trait,
            "tissue_context": result.request.tissue_context,
            "n_instruments": len(result.eqtl_instruments) if result.eqtl_instruments else 0,
            "n_harmonized_snps": len(result.harmonized_data) if result.harmonized_data else 0,
            "analysis_completed": result.success
        }
        
        if result.mr_results:
            mr_summary = result.mr_results.get("summary", {})
            summary.update({
                "causal_conclusion": mr_summary.get("conclusion", "No conclusion available"),
                "primary_method": "Inverse Variance Weighted"
            })
        
        return summary
    
    async def _generate_interpretation(self, result: CausalAnalysisResult, language: str = "zh", show_thinking: bool = False) -> str:
        """Generate detailed interpretation using LLM."""

        # 预先生成基因和药物的LLM解释
        gene_llm_interp = ""
        drug_llm_interp = ""

        if self.llm_service.is_available:
            # 尝试生成基因功能的LLM解释
            if result.gene_annotation:
                gene_info = result.gene_annotation.get("gene_info", {})
                if gene_info:
                    try:
                        gene_llm_interp = await self.llm_service.generate_gene_interpretation(
                            gene_symbol=result.request.exposure_gene,
                            gene_info=gene_info,
                            language=language,
                            show_thinking=show_thinking
                        )
                        if gene_llm_interp:
                            logger.info("✅ Generated gene function interpretation using LLM")
                    except Exception as e:
                        logger.warning(f"LLM基因解释生成失败: {e}")

            # 尝试生成药物治疗的LLM解释
            if result.drug_analysis:
                drug_targets = result.drug_analysis.get("drug_targets", [])
                if drug_targets:
                    try:
                        drug_llm_interp = await self.llm_service.generate_drug_interpretation(
                            gene_symbol=result.request.exposure_gene,
                            drug_targets=drug_targets,
                            language=language,
                            show_thinking=show_thinking
                        )
                        if drug_llm_interp:
                            logger.info("✅ Generated drug treatment interpretation using LLM")
                    except Exception as e:
                        logger.warning(f"LLM药物解释生成失败: {e}")

        # 尝试使用LLM生成主要的MR分析解释
        if self.llm_service.is_available and result.mr_results:
            try:
                llm_interpretation = await self.llm_service.generate_analysis_interpretation(
                    exposure_gene=result.request.exposure_gene,
                    outcome_trait=result.request.outcome_trait,
                    mr_results=result.mr_results,
                    language=language,
                    show_thinking=show_thinking
                )

                if llm_interpretation:
                    logger.info("✅ Generated intelligent analysis interpretation using LLM")
                    # 如果有LLM生成的基因和药物解释，添加到主解释中
                    if gene_llm_interp:
                        llm_interpretation += f"\n\n## {get_text('gene_function_context', language)}\n{gene_llm_interp}"
                    if drug_llm_interp:
                        llm_interpretation += f"\n\n## {get_text('therapeutic_implications', language)}\n{drug_llm_interp}"
                    return llm_interpretation

            except Exception as e:
                logger.warning(f"LLM解释生成失败，使用备用方法: {e}")

        # 备用方案：基于规则的解释，传递LLM生成的内容
        return self._generate_fallback_interpretation(result, language, gene_llm_interp, drug_llm_interp)

    def _generate_fallback_interpretation(self, result: CausalAnalysisResult, language: str = "en", gene_llm_interp: str = "", drug_llm_interp: str = "") -> str:
        """Generate fallback interpretation using rule-based approach."""
        interpretation_parts = []

        # 使用国际化的标题和文本
        interpretation_parts.append(
            f"# {get_text('causal_analysis_report', language).format(result.request.exposure_gene, result.request.outcome_trait)}\n"
        )

        interpretation_parts.append(
            f"## {get_text('analysis_overview', language)}\n"
            f"{get_text('analysis_overview_text', language).format(result.request.exposure_gene, result.request.outcome_trait)}\n"
        )

        if result.mr_results:
            mr_interpretation = result.mr_results.get("interpretation", "")
            interpretation_parts.append(f"## {get_text('causal_analysis_results', language)}\n{mr_interpretation}\n")

        if result.gene_annotation:
            # 优先使用LLM生成的解释
            gene_interp = gene_llm_interp

            # 如果没有LLM解释，尝试从数据结构中获取
            if not gene_interp:
                gene_interp = result.gene_annotation.get("interpretation", "")

            # 如果仍然没有，使用基于规则的方法生成
            if not gene_interp:
                gene_info = result.gene_annotation.get("gene_info", {})
                if gene_info:
                    gene_interp = self._generate_gene_summary(gene_info, result.request.exposure_gene, language)

            if gene_interp and gene_interp.strip():  # 只有当有实际内容时才显示
                interpretation_parts.append(f"## {get_text('gene_function_context', language)}\n{gene_interp}\n")

        if result.drug_analysis:
            # 优先使用LLM生成的解释
            drug_summary = drug_llm_interp

            # 如果没有LLM解释，尝试从数据结构中获取
            if not drug_summary:
                drug_summary = result.drug_analysis.get("therapeutic_opportunities", "")

            # 如果仍然没有，使用基于规则的方法生成
            if not drug_summary:
                drug_targets = result.drug_analysis.get("drug_targets", [])
                if drug_targets:
                    drug_summary = self._generate_drug_summary(drug_targets, result.request.exposure_gene, language)

            if drug_summary and drug_summary.strip():  # 只有当有实际内容时才显示
                interpretation_parts.append(f"## {get_text('therapeutic_implications', language)}\n{drug_summary}\n")

        return "\n".join(interpretation_parts)

    def _generate_gene_summary(self, gene_info: Dict[str, Any], gene_symbol: str, language: str = "zh") -> str:
        """从基因信息生成简单摘要"""
        try:
            summary_parts = []

            if language == "zh":
                summary_parts.append(f"**基因 {gene_symbol} 的功能信息：**")

                # 蛋白质名称
                if gene_info.get("protein_name"):
                    summary_parts.append(f"- 蛋白质名称：{gene_info['protein_name']}")

                # 功能描述
                if gene_info.get("function"):
                    function_text = gene_info["function"]
                    if len(function_text) > 200:
                        function_text = function_text[:200] + "..."
                    summary_parts.append(f"- 分子功能：{function_text}")

                # 亚细胞定位
                if gene_info.get("subcellular_location"):
                    summary_parts.append(f"- 亚细胞定位：{gene_info['subcellular_location']}")

                # 基因描述
                if gene_info.get("description"):
                    desc_text = gene_info["description"]
                    if len(desc_text) > 150:
                        desc_text = desc_text[:150] + "..."
                    summary_parts.append(f"- 基因描述：{desc_text}")

            else:  # English
                summary_parts.append(f"**Functional information for gene {gene_symbol}:**")

                if gene_info.get("protein_name"):
                    summary_parts.append(f"- Protein name: {gene_info['protein_name']}")

                if gene_info.get("function"):
                    function_text = gene_info["function"]
                    if len(function_text) > 200:
                        function_text = function_text[:200] + "..."
                    summary_parts.append(f"- Molecular function: {function_text}")

                if gene_info.get("subcellular_location"):
                    summary_parts.append(f"- Subcellular location: {gene_info['subcellular_location']}")

                if gene_info.get("description"):
                    desc_text = gene_info["description"]
                    if len(desc_text) > 150:
                        desc_text = desc_text[:150] + "..."
                    summary_parts.append(f"- Gene description: {desc_text}")

            return "\n".join(summary_parts) if summary_parts else ""

        except Exception as e:
            logger.error(f"基因摘要生成失败: {e}")
            return ""

    def _generate_drug_summary(self, drug_targets: List[Dict[str, Any]], gene_symbol: str, language: str = "zh") -> str:
        """从药物靶点信息生成简单摘要"""
        try:
            if not drug_targets:
                if language == "zh":
                    return f"目前没有已知药物直接靶向 {gene_symbol} 基因，这可能代表潜在的药物开发机会。"
                else:
                    return f"No known drugs currently target {gene_symbol}. This may represent a potential drug development opportunity."

            summary_parts = []

            if language == "zh":
                summary_parts.append(f"**基因 {gene_symbol} 的治疗意义：**")
                summary_parts.append(f"- 发现 {len(drug_targets)} 个相关药物靶点")

                # 显示前几个药物
                for i, drug in enumerate(drug_targets[:3]):
                    # 尝试多个可能的字段名获取药物名称
                    drug_name = (drug.get("compound_name") or
                               drug.get("drug_name") or
                               drug.get("name") or
                               drug.get("compound_id") or
                               f"药物 {i+1}")
                    summary_parts.append(f"- {drug_name}")

                if len(drug_targets) > 3:
                    summary_parts.append(f"- 以及其他 {len(drug_targets) - 3} 个药物...")

            else:  # English
                summary_parts.append(f"**Therapeutic implications for gene {gene_symbol}:**")
                summary_parts.append(f"- Found {len(drug_targets)} related drug targets")

                for i, drug in enumerate(drug_targets[:3]):
                    # 尝试多个可能的字段名获取药物名称
                    drug_name = (drug.get("compound_name") or
                               drug.get("drug_name") or
                               drug.get("name") or
                               drug.get("compound_id") or
                               f"Drug {i+1}")
                    summary_parts.append(f"- {drug_name}")

                if len(drug_targets) > 3:
                    summary_parts.append(f"- And {len(drug_targets) - 3} other drugs...")

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"药物摘要生成失败: {e}")
            return ""

    async def _generate_recommendations(self, result: CausalAnalysisResult, language: str = "zh", show_thinking: bool = False) -> List[str]:
        """Generate recommendations for follow-up using LLM."""

        # 尝试使用LLM生成智能建议
        if self.llm_service.is_available and result.success:
            try:
                analysis_summary = {
                    "summary": result.summary,
                    "mr_results": result.mr_results,
                    "gene_annotation": result.gene_annotation,
                    "drug_analysis": result.drug_analysis
                }

                llm_recommendations = await self.llm_service.generate_recommendations(
                    analysis_result=analysis_summary,
                    language=language,
                    show_thinking=show_thinking
                )

                if llm_recommendations:
                    logger.info("✅ 使用LLM生成了智能建议")
                    return llm_recommendations

            except Exception as e:
                logger.warning(f"LLM建议生成失败，使用备用方法: {e}")

        # 备用方案：基于规则的建议
        return self._generate_fallback_recommendations(result, language)

    def _generate_fallback_recommendations(self, result: CausalAnalysisResult, language: str = "zh") -> List[str]:
        """Generate fallback recommendations using rule-based approach."""
        recommendations = []

        if result.success and result.mr_results:
            # Check if causal effect was found
            ivw_result = None
            for method_result in result.mr_results.get("results", []):
                if "Inverse Variance" in method_result.get("method", ""):
                    ivw_result = method_result
                    break

            if ivw_result and ivw_result.get("p_value", 1) < 0.05:
                recommendations.extend([
                    get_text("strong_causal_evidence", language),
                    get_text("investigate_mechanisms", language),
                    get_text("clinical_implications", language)
                ])

                if result.drug_analysis:
                    drugs = result.drug_analysis.get("targeting_drugs", [])
                    if drugs:
                        recommendations.append(get_text("existing_drugs_target", language))
                    else:
                        recommendations.append(get_text("no_drugs_found", language))
            else:
                recommendations.extend([
                    get_text("no_significant_effect", language),
                    get_text("alternative_measures", language),
                    get_text("investigate_confounding", language)
                ])
        else:
            recommendations.extend([
                get_text("analysis_incomplete", language),
                get_text("ensure_servers_running", language),
                get_text("check_data_availability", language)
            ])

        return recommendations
    
    async def get_system_status(self):
        """Get current system status including LLM service."""
        mcp_status = await self.mcp_client.check_server_status()
        llm_status = self.llm_service.get_status()

        # 添加LLM服务状态到服务器列表
        if hasattr(mcp_status, 'servers'):
            llm_server_status = type('ServerStatus', (), {
                'server_name': 'LLM Service',
                'status': 'online' if llm_status['available'] else 'offline',
                'model': llm_status.get('model', 'N/A')
            })()
            mcp_status.servers.append(llm_server_status)

        return mcp_status
