# client-app/llm_workflow_coordinator.py
"""
LLM驱动的工作流协调器
负责协调整个因果推断分析流程，由LLM主导决策
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from models import CausalAnalysisRequest, CausalAnalysisResult, AnalysisType
from natural_language_parser import NaturalLanguageParser, ParsedParameters

logger = logging.getLogger(__name__)

class WorkflowStage(Enum):
    """工作流阶段"""
    INPUT_PARSING = "input_parsing"
    PARAMETER_VALIDATION = "parameter_validation"
    EQTL_ANALYSIS = "eqtl_analysis"
    GWAS_ANALYSIS = "gwas_analysis"
    MR_ANALYSIS = "mr_analysis"
    KNOWLEDGE_INTEGRATION = "knowledge_integration"
    RESULT_INTERPRETATION = "result_interpretation"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class WorkflowState:
    """工作流状态"""
    stage: WorkflowStage
    progress: float
    message: str
    data: Dict[str, Any]
    errors: List[str]

class LLMWorkflowCoordinator:
    """LLM驱动的工作流协调器"""
    
    def __init__(self, causal_analyzer, llm_service, input_validator):
        """
        初始化协调器
        
        Args:
            causal_analyzer: 因果分析器实例
            llm_service: LLM服务实例
            input_validator: 输入验证器实例
        """
        self.causal_analyzer = causal_analyzer
        self.llm_service = llm_service
        self.input_validator = input_validator
        self.parser = NaturalLanguageParser(llm_service, input_validator)
        
        # 工作流状态
        self.current_state = WorkflowState(
            stage=WorkflowStage.INPUT_PARSING,
            progress=0.0,
            message="准备开始分析",
            data={},
            errors=[]
        )
        
        # 回调函数（用于UI更新）
        self.progress_callback = None
    
    def set_progress_callback(self, callback):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    async def _update_progress(self, stage: WorkflowStage, progress: float, message: str, data: Dict = None):
        """更新工作流进度"""
        self.current_state.stage = stage
        self.current_state.progress = progress
        self.current_state.message = message
        if data:
            self.current_state.data.update(data)
        
        logger.info(f"工作流进度: {stage.value} - {progress:.1%} - {message}")
        
        if self.progress_callback:
            await self.progress_callback(self.current_state)
    
    async def execute_analysis(self, user_input: str, language: str = "zh") -> CausalAnalysisResult:
        """
        执行完整的因果推断分析流程
        
        Args:
            user_input: 用户的自然语言输入
            language: 语言代码
            
        Returns:
            CausalAnalysisResult: 分析结果
        """
        try:
            # 阶段1: 输入解析
            await self._update_progress(
                WorkflowStage.INPUT_PARSING, 
                0.1, 
                "正在解析用户输入..." if language == "zh" else "Parsing user input..."
            )
            
            parsed_params = await self.parser.parse_input(user_input, language)
            
            # 检查是否需要澄清
            if parsed_params.missing_params:
                clarification = self.parser.generate_clarification_prompt(parsed_params, language)
                await self._update_progress(
                    WorkflowStage.PARAMETER_VALIDATION,
                    0.15,
                    f"需要补充信息: {clarification}" if language == "zh" else f"Need clarification: {clarification}",
                    {"parsed_params": parsed_params, "clarification": clarification}
                )
                # 返回部分结果，等待用户补充
                return self._create_partial_result(parsed_params, clarification, language)
            
            # 阶段2: 参数验证
            await self._update_progress(
                WorkflowStage.PARAMETER_VALIDATION,
                0.2,
                "验证分析参数..." if language == "zh" else "Validating parameters..."
            )
            
            # 验证参数
            validation_result = await self._validate_parameters(parsed_params, language)
            if not validation_result["valid"]:
                error_msg = validation_result["message"]
                self.current_state.errors.append(error_msg)
                await self._update_progress(
                    WorkflowStage.ERROR,
                    0.2,
                    error_msg
                )
                return self._create_error_result(error_msg, language)
            
            # 创建分析请求
            request = CausalAnalysisRequest(
                exposure_gene=parsed_params.gene,
                outcome_trait=parsed_params.disease,
                tissue_context=parsed_params.tissue or "Whole_Blood",
                analysis_type=AnalysisType.FULL_CAUSAL_ANALYSIS,
                include_pathway_analysis=True,
                include_drug_analysis=True,
                language=language,
                show_thinking=False
            )
            
            # 阶段3-7: LLM驱动的分析执行
            result = await self._execute_llm_driven_analysis(request, language)
            
            # 阶段8: 完成
            await self._update_progress(
                WorkflowStage.COMPLETED,
                1.0,
                "分析完成！" if language == "zh" else "Analysis completed!",
                {"result": result}
            )
            
            return result
            
        except Exception as e:
            error_msg = f"分析过程中发生错误: {str(e)}" if language == "zh" else f"Error during analysis: {str(e)}"
            logger.error(error_msg, exc_info=True)
            self.current_state.errors.append(error_msg)
            await self._update_progress(WorkflowStage.ERROR, 0.0, error_msg)
            return self._create_error_result(error_msg, language)
    
    async def _validate_parameters(self, parsed_params: ParsedParameters, language: str) -> Dict[str, Any]:
        """验证解析的参数"""
        errors = []
        
        # 验证基因
        if parsed_params.gene:
            gene_valid, gene_error = self.input_validator.validate_gene(parsed_params.gene, language)
            if not gene_valid:
                errors.append(f"基因验证失败: {gene_error}" if language == "zh" else f"Gene validation failed: {gene_error}")
        
        # 验证疾病
        if parsed_params.disease:
            trait_valid, trait_message, trait_info = self.input_validator.validate_gwas_trait(parsed_params.disease, language)
            if not trait_valid:
                errors.append(f"疾病验证失败: {trait_message}" if language == "zh" else f"Disease validation failed: {trait_message}")
        
        # 验证组织
        if parsed_params.tissue:
            tissue_valid, tissue_error = self.input_validator.validate_tissue(parsed_params.tissue, language)
            if not tissue_valid:
                errors.append(f"组织验证失败: {tissue_error}" if language == "zh" else f"Tissue validation failed: {tissue_error}")
        
        if errors:
            return {
                "valid": False,
                "message": "; ".join(errors),
                "errors": errors
            }
        
        return {"valid": True, "message": "参数验证通过" if language == "zh" else "Parameters validated"}
    
    async def _execute_llm_driven_analysis(self, request: CausalAnalysisRequest, language: str) -> CausalAnalysisResult:
        """执行LLM驱动的因果分析流程"""

        # 初始化结果结构
        from models import CausalAnalysisResult, AnalysisStep
        import time

        start_time = time.time()
        result = CausalAnalysisResult(
            request=request,
            analysis_steps=[],
            summary={},
            interpretation="",
            recommendations=[],
            total_execution_time=0.0,
            success=False,
            warnings=[]
        )

        # 记录分析步骤
        from models import AnalysisStep

        try:
            # 阶段3: LLM驱动的eQTL分析
            await self._update_progress(
                WorkflowStage.EQTL_ANALYSIS,
                0.3,
                f"🤖 LLM正在分析{request.exposure_gene}基因的eQTL策略..." if language == "zh" else f"🤖 LLM analyzing eQTL strategy for {request.exposure_gene}..."
            )

            eqtl_step_start = time.time()
            eqtl_result = await self._llm_guided_eqtl_analysis(request, language)
            result.eqtl_instruments = eqtl_result.get("instruments", [])

            # 记录eQTL步骤
            result.analysis_steps.append(AnalysisStep(
                step_name="eQTL Analysis",
                status="completed",
                server_used="eQTL Server",
                execution_time=time.time() - eqtl_step_start,
                error_message=None
            ))

            # LLM评估eQTL质量并决定下一步
            eqtl_assessment = await self._llm_assess_eqtl_quality(eqtl_result, language)

            if not eqtl_assessment["proceed"]:
                # LLM建议停止分析
                result.interpretation = eqtl_assessment["reason"]
                result.recommendations = eqtl_assessment["recommendations"]
                return result

            # 阶段4: LLM驱动的GWAS分析
            await self._update_progress(
                WorkflowStage.GWAS_ANALYSIS,
                0.5,
                f"🤖 LLM正在优化{request.outcome_trait}的GWAS查询..." if language == "zh" else f"🤖 LLM optimizing GWAS query for {request.outcome_trait}..."
            )

            gwas_step_start = time.time()
            gwas_result = await self._llm_guided_gwas_analysis(request, eqtl_result, language)
            result.harmonized_data = gwas_result.get("harmonized_data", [])

            # 记录GWAS步骤
            result.analysis_steps.append(AnalysisStep(
                step_name="GWAS Analysis",
                status="completed",
                server_used="GWAS Server",
                execution_time=time.time() - gwas_step_start,
                error_message=None
            ))

            # LLM评估数据质量
            data_assessment = await self._llm_assess_data_quality(eqtl_result, gwas_result, language)

            # 阶段5: LLM驱动的MR分析
            await self._update_progress(
                WorkflowStage.MR_ANALYSIS,
                0.7,
                f"🤖 LLM正在选择最佳MR方法..." if language == "zh" else f"🤖 LLM selecting optimal MR methods..."
            )

            mr_step_start = time.time()
            mr_result = await self._llm_guided_mr_analysis(request, result.harmonized_data, data_assessment, language)
            result.mr_results = mr_result

            # 记录MR步骤
            result.analysis_steps.append(AnalysisStep(
                step_name="Mendelian Randomization",
                status="completed",
                server_used="MR Server",
                execution_time=time.time() - mr_step_start,
                error_message=None
            ))

            # 阶段6: LLM驱动的知识整合
            await self._update_progress(
                WorkflowStage.KNOWLEDGE_INTEGRATION,
                0.85,
                f"🤖 LLM正在整合生物学知识..." if language == "zh" else f"🤖 LLM integrating biological knowledge..."
            )

            knowledge_step_start = time.time()
            knowledge_result = await self._llm_guided_knowledge_integration(request, mr_result, language)
            result.gene_annotation = knowledge_result.get("gene_annotation")
            result.drug_analysis = knowledge_result.get("drug_analysis")

            # 记录知识整合步骤
            result.analysis_steps.append(AnalysisStep(
                step_name="Knowledge Integration",
                status="completed",
                server_used="Knowledge Server",
                execution_time=time.time() - knowledge_step_start,
                error_message=None
            ))

            # 阶段7: LLM驱动的结果解释
            await self._update_progress(
                WorkflowStage.RESULT_INTERPRETATION,
                0.95,
                f"🤖 LLM正在生成智能解释..." if language == "zh" else f"🤖 LLM generating intelligent interpretation..."
            )

            interpretation_result = await self._llm_generate_comprehensive_interpretation(result, language)
            result.interpretation = interpretation_result["interpretation"]
            result.recommendations = interpretation_result["recommendations"]

            # 生成完整的summary信息
            result.summary = self._generate_summary(result)

            # 保留MR结论
            if result.mr_results and isinstance(result.mr_results, dict):
                mr_conclusion = result.mr_results.get('summary', {}).get('conclusion', '')
                if mr_conclusion:
                    result.summary["causal_conclusion"] = mr_conclusion

            # 添加LLM生成的其他摘要信息
            llm_summary = interpretation_result.get("summary", {})
            for key, value in llm_summary.items():
                if key not in result.summary:  # 不覆盖已有的关键信息
                    result.summary[key] = value

            result.success = True

        except Exception as e:
            logger.error(f"LLM工作流执行失败: {e}")
            result.interpretation = f"分析失败: {str(e)}" if language == "zh" else f"Analysis failed: {str(e)}"
            result.success = False

        finally:
            result.total_execution_time = time.time() - start_time

        return result

    def _generate_summary(self, result: 'CausalAnalysisResult') -> Dict[str, Any]:
        """生成分析摘要"""
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

    async def _generate_fallback_interpretation(self, result: 'CausalAnalysisResult', language: str) -> str:
        """生成备用详细解释（当LLM不可用时）"""
        # 使用传统模式的解释生成逻辑
        return await self.causal_analyzer._generate_interpretation(result, language, show_thinking=False)

    async def _generate_fallback_recommendations(self, result: 'CausalAnalysisResult', language: str) -> List[str]:
        """生成备用推荐建议（当LLM不可用时）"""
        # 使用传统模式的建议生成逻辑
        return await self.causal_analyzer._generate_recommendations(result, language, show_thinking=False)
    
    def _create_partial_result(self, parsed_params: ParsedParameters, clarification: str, language: str) -> CausalAnalysisResult:
        """创建部分结果（需要用户澄清）"""
        from models import AnalysisStep
        
        # 创建一个虚拟的请求对象
        request = CausalAnalysisRequest(
            exposure_gene=parsed_params.gene or "UNKNOWN",
            outcome_trait=parsed_params.disease or "UNKNOWN",
            tissue_context=parsed_params.tissue or "Whole_Blood",
            language=language
        )
        
        step = AnalysisStep(
            step_name="Parameter Clarification",
            status="pending",
            input_data={"parsed_params": parsed_params.__dict__},
            output_data={"clarification": clarification}
        )
        
        return CausalAnalysisResult(
            request=request,
            analysis_steps=[step],
            summary={"status": "需要澄清参数" if language == "zh" else "Parameter clarification needed"},
            interpretation=clarification,
            recommendations=[],
            total_execution_time=0.0,
            success=False,
            warnings=[clarification]
        )
    
    def _create_error_result(self, error_message: str, language: str) -> CausalAnalysisResult:
        """创建错误结果"""
        from models import AnalysisStep
        
        request = CausalAnalysisRequest(
            exposure_gene="ERROR",
            outcome_trait="ERROR",
            tissue_context="Whole_Blood",
            language=language
        )
        
        step = AnalysisStep(
            step_name="Error",
            status="failed",
            error_message=error_message
        )
        
        return CausalAnalysisResult(
            request=request,
            analysis_steps=[step],
            summary={"status": "分析失败" if language == "zh" else "Analysis failed"},
            interpretation=error_message,
            recommendations=[],
            total_execution_time=0.0,
            success=False,
            error_message=error_message
        )

    async def _llm_guided_eqtl_analysis(self, request: CausalAnalysisRequest, language: str) -> Dict[str, Any]:
        """LLM指导的eQTL分析"""

        # LLM决策：选择最佳组织和分析策略
        strategy_prompt = self._create_eqtl_strategy_prompt(request, language)

        if self.llm_service.is_available:
            try:
                strategy_response = await self.llm_service._generate_text(strategy_prompt, max_length=512)
                strategy = self._parse_llm_strategy(strategy_response)
                logger.info(f"🤖 LLM eQTL策略: {strategy}")
            except Exception as e:
                logger.warning(f"LLM策略生成失败，使用默认策略: {e}")
                strategy = {"tissue": request.tissue_context, "approach": "standard"}
        else:
            strategy = {"tissue": request.tissue_context, "approach": "standard"}

        # 执行eQTL分析
        eqtl_result = await self.causal_analyzer.mcp_client.call_eqtl_server(
            request.exposure_gene,
            strategy.get("tissue", request.tissue_context)
        )

        # 添加LLM策略信息
        eqtl_result["llm_strategy"] = strategy
        return eqtl_result

    async def _llm_assess_eqtl_quality(self, eqtl_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """LLM评估eQTL数据质量"""

        instruments = eqtl_result.get("instruments", [])

        if not self.llm_service.is_available:
            # 简单规则评估
            return {
                "proceed": len(instruments) > 0,
                "reason": f"找到 {len(instruments)} 个工具变量" if language == "zh" else f"Found {len(instruments)} instruments",
                "recommendations": []
            }

        # LLM评估
        assessment_prompt = self._create_eqtl_assessment_prompt(eqtl_result, language)

        try:
            assessment_response = await self.llm_service._generate_text(assessment_prompt, max_length=512)
            assessment = self._parse_llm_assessment(assessment_response)
            return assessment
        except Exception as e:
            logger.warning(f"LLM评估失败，使用默认评估: {e}")
            return {
                "proceed": len(instruments) > 0,
                "reason": f"找到 {len(instruments)} 个工具变量" if language == "zh" else f"Found {len(instruments)} instruments",
                "recommendations": []
            }

    async def _llm_guided_gwas_analysis(self, request: CausalAnalysisRequest, eqtl_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """LLM指导的GWAS分析"""

        # LLM优化疾病查询
        if self.llm_service.is_available:
            try:
                query_prompt = self._create_gwas_query_prompt(request, eqtl_result, language)
                query_response = await self.llm_service._generate_text(query_prompt, max_length=256)
                optimized_query = self._parse_llm_query(query_response, request.outcome_trait)
            except Exception as e:
                logger.warning(f"LLM查询优化失败: {e}")
                optimized_query = request.outcome_trait
        else:
            optimized_query = request.outcome_trait

        # 使用疾病映射器获取研究ID
        outcome_id = self.causal_analyzer.disease_mapper.get_study_id_for_disease(optimized_query)
        if not outcome_id:
            outcome_id = self.causal_analyzer.disease_mapper.get_study_id_for_disease(request.outcome_trait)

        if not outcome_id:
            raise ValueError(f"无法找到疾病 '{request.outcome_trait}' 对应的GWAS研究")

        # 执行GWAS分析
        gwas_result = await self.causal_analyzer.mcp_client.call_gwas_server(
            eqtl_result.get("instruments", []),
            outcome_id
        )

        gwas_result["llm_optimized_query"] = optimized_query
        return gwas_result

    async def _llm_assess_data_quality(self, eqtl_result: Dict[str, Any], gwas_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """LLM评估数据质量并推荐分析策略"""

        if not self.llm_service.is_available:
            # 简单规则评估
            harmonized_count = len(gwas_result.get("harmonized_data", []))
            return {
                "quality_score": min(harmonized_count / 10, 1.0),
                "recommended_methods": ["IVW", "Weighted Median"],
                "confidence_level": "medium"
            }

        # LLM评估数据质量
        quality_prompt = self._create_data_quality_prompt(eqtl_result, gwas_result, language)

        try:
            quality_response = await self.llm_service._generate_text(quality_prompt, max_length=512)
            quality_assessment = self._parse_llm_quality_assessment(quality_response)
            return quality_assessment
        except Exception as e:
            logger.warning(f"LLM数据质量评估失败: {e}")
            harmonized_count = len(gwas_result.get("harmonized_data", []))
            return {
                "quality_score": min(harmonized_count / 10, 1.0),
                "recommended_methods": ["IVW", "Weighted Median"],
                "confidence_level": "medium"
            }

    async def _llm_guided_mr_analysis(self, request: CausalAnalysisRequest, harmonized_data: List[Dict], data_assessment: Dict[str, Any], language: str) -> Dict[str, Any]:
        """LLM指导的MR分析"""

        # LLM选择最佳MR方法
        if self.llm_service.is_available:
            try:
                method_prompt = self._create_mr_method_prompt(harmonized_data, data_assessment, language)
                method_response = await self.llm_service._generate_text(method_prompt, max_length=256)
                selected_methods = self._parse_llm_methods(method_response)
                logger.info(f"🤖 LLM推荐的MR方法: {selected_methods}")
            except Exception as e:
                logger.warning(f"LLM方法选择失败: {e}")
                selected_methods = data_assessment.get("recommended_methods", ["IVW", "Weighted Median"])
        else:
            selected_methods = data_assessment.get("recommended_methods", ["IVW", "Weighted Median"])

        # 执行MR分析
        mr_result = await self.causal_analyzer.mcp_client.call_mr_server(
            harmonized_data,
            f"{request.exposure_gene} Expression",
            request.outcome_trait,
            language
        )

        mr_result["llm_selected_methods"] = selected_methods
        mr_result["data_quality_assessment"] = data_assessment
        return mr_result

    def _create_data_quality_prompt(self, eqtl_result: Dict[str, Any], gwas_result: Dict[str, Any], language: str) -> str:
        """创建数据质量评估的LLM提示"""

        eqtl_count = len(eqtl_result.get("instruments", []))
        harmonized_count = len(gwas_result.get("harmonized_data", []))

        if language == "zh":
            prompt = f"""
作为生物统计学专家，请评估以下孟德尔随机化分析的数据质量：

eQTL数据：
- 工具变量数量：{eqtl_count}
- 数据来源：{eqtl_result.get('data_source', 'Unknown')}

GWAS数据：
- 协调后SNP数量：{harmonized_count}
- 数据来源：{gwas_result.get('data_source', 'Unknown')}

请评估：
1. 数据质量等级（高/中/低）
2. 推荐的MR方法（从IVW、MR-Egger、Weighted Median中选择）
3. 置信度水平（高/中/低）
4. 潜在的分析限制

请以JSON格式回复：
{{
    "quality_level": "高/中/低",
    "recommended_methods": ["方法1", "方法2"],
    "confidence_level": "高/中/低",
    "limitations": "分析限制说明"
}}
"""
        else:
            prompt = f"""
As a biostatistics expert, please assess the data quality for the following Mendelian Randomization analysis:

eQTL Data:
- Number of instruments: {eqtl_count}
- Data source: {eqtl_result.get('data_source', 'Unknown')}

GWAS Data:
- Number of harmonized SNPs: {harmonized_count}
- Data source: {gwas_result.get('data_source', 'Unknown')}

Please assess:
1. Data quality level (High/Medium/Low)
2. Recommended MR methods (choose from IVW, MR-Egger, Weighted Median)
3. Confidence level (High/Medium/Low)
4. Potential analysis limitations

Please respond in JSON format:
{{
    "quality_level": "High/Medium/Low",
    "recommended_methods": ["method1", "method2"],
    "confidence_level": "High/Medium/Low",
    "limitations": "Analysis limitations description"
}}
"""

        return prompt

    async def _llm_guided_knowledge_integration(self, request: CausalAnalysisRequest, mr_result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """LLM指导的知识整合"""

        knowledge_result = {}

        # 基因注释
        if request.include_pathway_analysis:
            gene_annotation = await self.causal_analyzer.mcp_client.call_knowledge_server(
                request.exposure_gene,
                "gene_annotation"
            )
            knowledge_result["gene_annotation"] = gene_annotation

        # 药物分析
        if request.include_drug_analysis:
            drug_analysis = await self.causal_analyzer.mcp_client.call_knowledge_server_drug(
                request.exposure_gene
            )
            knowledge_result["drug_analysis"] = drug_analysis

        # LLM整合知识
        if self.llm_service.is_available:
            try:
                integration_prompt = self._create_knowledge_integration_prompt(request, mr_result, knowledge_result, language)
                integration_response = await self.llm_service._generate_text(integration_prompt, max_length=1024)
                integrated_knowledge = self._parse_llm_integration(integration_response)
                knowledge_result["llm_integration"] = integrated_knowledge
            except Exception as e:
                logger.warning(f"LLM知识整合失败: {e}")

        return knowledge_result

    async def _llm_generate_comprehensive_interpretation(self, result: 'CausalAnalysisResult', language: str) -> Dict[str, Any]:
        """LLM生成综合解释"""

        if not self.llm_service.is_available:
            # 使用传统模式的详细解释生成逻辑
            detailed_interpretation = await self._generate_fallback_interpretation(result, language)
            fallback_recommendations = await self._generate_fallback_recommendations(result, language)

            mr_conclusion = "无MR结果"
            if result.mr_results and isinstance(result.mr_results, dict):
                mr_conclusion = result.mr_results.get('summary', {}).get('conclusion', '无结论')

            return {
                "interpretation": detailed_interpretation,
                "recommendations": fallback_recommendations,
                "summary": {
                    "status": "completed",
                    "causal_conclusion": mr_conclusion
                }
            }

        # LLM生成综合解释
        try:
            interpretation_prompt = self._create_comprehensive_interpretation_prompt(result, language)
            interpretation_response = await self.llm_service._generate_text(interpretation_prompt, max_length=2048)

            # 检查LLM是否返回了有效响应
            if interpretation_response and interpretation_response.strip():
                interpretation_result = self._parse_llm_interpretation(interpretation_response)
                # 验证解析结果是否包含必要的字段
                if interpretation_result.get("interpretation") and interpretation_result.get("recommendations"):
                    return interpretation_result
                else:
                    logger.warning("LLM解释解析结果不完整，使用备用方法")
            else:
                logger.warning("LLM返回空响应，使用备用方法")

        except Exception as e:
            logger.warning(f"LLM解释生成失败: {e}，使用备用方法")

        # LLM失败时，使用备用的详细解释生成
        logger.info("使用传统模式的详细解释生成逻辑")
        detailed_interpretation = await self._generate_fallback_interpretation(result, language)
        fallback_recommendations = await self._generate_fallback_recommendations(result, language)

        mr_conclusion = "无MR结果"
        if result.mr_results and isinstance(result.mr_results, dict):
            mr_conclusion = result.mr_results.get('summary', {}).get('conclusion', '无结论')

        return {
            "interpretation": detailed_interpretation,
            "recommendations": fallback_recommendations,
            "summary": {
                "status": "completed",
                "causal_conclusion": mr_conclusion
            }
        }

    # ========== LLM提示词创建方法 ==========

    def _create_eqtl_strategy_prompt(self, request: CausalAnalysisRequest, language: str) -> str:
        """创建eQTL策略提示词"""
        if language == "zh":
            return f"""
你是一个专业的基因表达数量性状位点(eQTL)分析专家。请为以下研究设计最佳的eQTL分析策略：

研究目标：分析{request.exposure_gene}基因对{request.outcome_trait}的因果效应
当前组织：{request.tissue_context}

请考虑以下因素：
1. 该基因在不同组织中的表达模式
2. 疾病的生物学相关性
3. eQTL效应量的组织特异性

请以JSON格式返回策略：
{{
    "tissue": "推荐的最佳组织",
    "approach": "分析方法(standard/multi-tissue/tissue-specific)",
    "reasoning": "选择理由"
}}
"""
        else:
            return f"""
You are a professional eQTL analysis expert. Please design the optimal eQTL analysis strategy for:

Research goal: Analyze causal effect of {request.exposure_gene} gene on {request.outcome_trait}
Current tissue: {request.tissue_context}

Consider:
1. Gene expression patterns across tissues
2. Disease biological relevance
3. Tissue-specific eQTL effect sizes

Return strategy in JSON format:
{{
    "tissue": "recommended optimal tissue",
    "approach": "analysis method (standard/multi-tissue/tissue-specific)",
    "reasoning": "selection rationale"
}}
"""

    def _create_eqtl_assessment_prompt(self, eqtl_result: Dict[str, Any], language: str) -> str:
        """创建eQTL质量评估提示词"""
        instruments = eqtl_result.get("instruments", [])
        instrument_count = len(instruments)

        if language == "zh":
            return f"""
作为eQTL数据质量专家，请评估以下eQTL分析结果：

工具变量数量：{instrument_count}个
数据来源：GTEx数据库

评估标准：
1. 工具变量数量是否充足（通常需要≥3个）
2. 效应量是否合理
3. 是否适合进行孟德尔随机化分析

请以JSON格式返回评估：
{{
    "proceed": true/false,
    "reason": "评估理由",
    "recommendations": ["建议列表"]
}}
"""
        else:
            return f"""
As an eQTL data quality expert, please assess the following eQTL analysis results:

Number of instruments: {instrument_count}
Data source: GTEx database

Assessment criteria:
1. Sufficient number of instruments (usually ≥3)
2. Reasonable effect sizes
3. Suitability for Mendelian randomization

Return assessment in JSON format:
{{
    "proceed": true/false,
    "reason": "assessment rationale",
    "recommendations": ["recommendation list"]
}}
"""

    def _create_gwas_query_prompt(self, request: CausalAnalysisRequest, eqtl_result: Dict[str, Any], language: str) -> str:
        """创建GWAS查询优化提示词"""
        if language == "zh":
            return f"""
作为GWAS数据专家，请优化以下疾病的查询策略：

目标疾病：{request.outcome_trait}
基因：{request.exposure_gene}
eQTL工具变量数量：{len(eqtl_result.get("instruments", []))}

请考虑：
1. 疾病的标准术语和同义词
2. 相关的表型和亚型
3. 数据库中可能的命名方式

请返回优化后的查询词（单个最佳术语）：
"""
        else:
            return f"""
As a GWAS data expert, please optimize the query strategy for:

Target disease: {request.outcome_trait}
Gene: {request.exposure_gene}
Number of eQTL instruments: {len(eqtl_result.get("instruments", []))}

Consider:
1. Standard terminology and synonyms
2. Related phenotypes and subtypes
3. Possible naming conventions in databases

Return optimized query term (single best term):
"""

    # ========== LLM响应解析方法 ==========

    def _parse_llm_strategy(self, response: str) -> Dict[str, Any]:
        """解析LLM策略响应"""
        import json
        import re

        try:
            # 尝试提取JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                strategy = json.loads(json_match.group())
                return strategy
        except:
            pass

        # 备用解析
        return {
            "tissue": "Whole_Blood",
            "approach": "standard",
            "reasoning": "Default strategy due to parsing failure"
        }

    def _parse_llm_assessment(self, response: str) -> Dict[str, Any]:
        """解析LLM评估响应"""
        import json
        import re

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                return assessment
        except:
            pass

        # 备用解析
        proceed = "proceed" in response.lower() or "继续" in response
        return {
            "proceed": proceed,
            "reason": "Parsed from text response",
            "recommendations": []
        }

    def _parse_llm_query(self, response: str, original_query: str) -> str:
        """解析LLM查询优化响应"""
        # 简单提取：取第一行非空文本
        lines = [line.strip() for line in response.split('\n') if line.strip()]
        if lines:
            optimized = lines[0]
            # 移除可能的引号
            optimized = optimized.strip('"\'')
            return optimized if optimized else original_query
        return original_query

    def _parse_llm_quality_assessment(self, response: str) -> Dict[str, Any]:
        """解析LLM数据质量评估响应"""
        import json
        import re

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                assessment = json.loads(json_match.group())
                return assessment
        except:
            pass

        # 备用解析
        return {
            "quality_score": 0.7,
            "recommended_methods": ["IVW", "Weighted Median"],
            "confidence_level": "medium"
        }

    def _create_mr_method_prompt(self, harmonized_data: List[Dict], data_assessment: Dict[str, Any], language: str) -> str:
        """创建MR方法选择的LLM提示"""

        snp_count = len(harmonized_data)
        quality_level = data_assessment.get("quality_level", "medium")

        if language == "zh":
            prompt = f"""
作为孟德尔随机化分析专家，请为以下数据选择最适合的MR方法：

数据概况：
- SNP数量：{snp_count}
- 数据质量：{quality_level}
- 当前评估：{data_assessment.get('limitations', '无特殊限制')}

可选方法：
1. IVW (Inverse Variance Weighted) - 标准方法，假设无多效性
2. MR-Egger - 检测和校正多效性偏倚
3. Weighted Median - 对异常值稳健
4. Weighted Mode - 对多效性稳健
5. Simple Mode - 简单模式估计

请根据数据特点选择2-3个最适合的方法，用逗号分隔：
"""
        else:
            prompt = f"""
As a Mendelian Randomization analysis expert, please select the most appropriate MR methods for the following data:

Data Overview:
- Number of SNPs: {snp_count}
- Data quality: {quality_level}
- Current assessment: {data_assessment.get('limitations', 'No specific limitations')}

Available methods:
1. IVW (Inverse Variance Weighted) - Standard method, assumes no pleiotropy
2. MR-Egger - Detects and corrects for pleiotropic bias
3. Weighted Median - Robust to outliers
4. Weighted Mode - Robust to pleiotropy
5. Simple Mode - Simple mode estimation

Please select 2-3 most suitable methods based on data characteristics, separated by commas:
"""

        return prompt

    def _create_eqtl_strategy_prompt(self, request: CausalAnalysisRequest, language: str) -> str:
        """创建eQTL策略选择的LLM提示"""

        if language == "zh":
            prompt = f"""
作为基因表达分析专家，请为基因 {request.exposure_gene} 制定最佳的eQTL分析策略：

分析目标：
- 基因：{request.exposure_gene}
- 目标疾病：{request.outcome_trait}
- 指定组织：{request.tissue_context}

请考虑：
1. 该基因在不同组织中的表达模式
2. 与目标疾病最相关的组织类型
3. 数据可用性和质量

请以JSON格式回复：
{{
    "tissue": "推荐的组织类型",
    "approach": "分析方法",
    "rationale": "选择理由"
}}
"""
        else:
            prompt = f"""
As a gene expression analysis expert, please develop the optimal eQTL analysis strategy for gene {request.exposure_gene}:

Analysis Target:
- Gene: {request.exposure_gene}
- Target disease: {request.outcome_trait}
- Specified tissue: {request.tissue_context}

Please consider:
1. Expression patterns of this gene in different tissues
2. Most relevant tissue types for the target disease
3. Data availability and quality

Please respond in JSON format:
{{
    "tissue": "recommended tissue type",
    "approach": "analysis method",
    "rationale": "selection rationale"
}}
"""

        return prompt

    def _parse_llm_strategy(self, response: str) -> Dict[str, Any]:
        """解析LLM策略响应"""
        import json
        import re

        try:
            # 尝试解析JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                strategy_data = json.loads(json_match.group())
                return strategy_data
        except:
            pass

        # 备用解析
        return {
            "tissue": "Whole_Blood",
            "approach": "standard",
            "rationale": "Default strategy"
        }

    def _create_eqtl_assessment_prompt(self, eqtl_result: Dict[str, Any], language: str) -> str:
        """创建eQTL质量评估的LLM提示"""

        instruments_count = len(eqtl_result.get("instruments", []))

        if language == "zh":
            prompt = f"""
作为eQTL分析专家，请评估以下eQTL分析结果的质量：

结果概况：
- 找到的工具变量数量：{instruments_count}
- 数据来源：{eqtl_result.get('data_source', 'Unknown')}

请评估：
1. 是否应该继续进行MR分析
2. 数据质量评级
3. 潜在问题和建议

请以JSON格式回复：
{{
    "proceed": true/false,
    "quality": "高/中/低",
    "reason": "评估理由",
    "recommendations": ["建议1", "建议2"]
}}
"""
        else:
            prompt = f"""
As an eQTL analysis expert, please assess the quality of the following eQTL analysis results:

Results Overview:
- Number of instruments found: {instruments_count}
- Data source: {eqtl_result.get('data_source', 'Unknown')}

Please assess:
1. Whether to proceed with MR analysis
2. Data quality rating
3. Potential issues and recommendations

Please respond in JSON format:
{{
    "proceed": true/false,
    "quality": "High/Medium/Low",
    "reason": "assessment rationale",
    "recommendations": ["recommendation1", "recommendation2"]
}}
"""

        return prompt

    def _parse_llm_assessment(self, response: str) -> Dict[str, Any]:
        """解析LLM评估响应"""
        import json
        import re

        try:
            # 尝试解析JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                assessment_data = json.loads(json_match.group())
                return assessment_data
        except:
            pass

        # 备用解析
        return {
            "proceed": True,
            "quality": "medium",
            "reason": "Default assessment",
            "recommendations": []
        }

    def _create_gwas_query_prompt(self, request: CausalAnalysisRequest, eqtl_result: Dict[str, Any], language: str) -> str:
        """创建GWAS查询优化的LLM提示"""

        if language == "zh":
            prompt = f"""
作为GWAS数据库专家，请将以下疾病名称标准化为GWAS数据库中常用的术语：

原始疾病名称：{request.outcome_trait}

重要规则：
1. 只返回标准的疾病名称，不要添加基因信息或其他修饰词
2. 使用GWAS数据库中常见的疾病术语
3. 保持简洁，避免复合查询
4. 常见标准化示例：
   - "冠心病" → "coronary heart disease"
   - "心脏病" → "coronary heart disease"
   - "糖尿病" → "type 2 diabetes"
   - "阿尔茨海默病" → "alzheimer disease"

请只返回标准化的疾病名称（不要JSON格式）：
"""
        else:
            prompt = f"""
As a GWAS database expert, please standardize the following disease name to common terminology used in GWAS databases:

Original disease name: {request.outcome_trait}

Important rules:
1. Return only the standard disease name, do not add gene information or other modifiers
2. Use common disease terminology found in GWAS databases
3. Keep it simple, avoid compound queries
4. Common standardization examples:
   - "heart disease" → "coronary heart disease"
   - "CHD" → "coronary heart disease"
   - "T2D" → "type 2 diabetes"
   - "alzheimer" → "alzheimer disease"

Please return only the standardized disease name (no JSON format):
"""

        return prompt

    def _parse_llm_query(self, response: str, original_query: str) -> str:
        """解析LLM查询优化响应"""

        # 新的简化解析：直接提取疾病名称
        optimized = response.strip()

        # 移除可能的引号和多余字符
        optimized = optimized.strip('"\'`')

        # 移除可能的前缀文本
        if ':' in optimized:
            optimized = optimized.split(':')[-1].strip()

        # 验证优化结果的合理性
        if self._is_valid_disease_name(optimized):
            return optimized

        # 如果优化结果不合理，返回原始查询
        return original_query

    def _is_valid_disease_name(self, disease_name: str) -> bool:
        """验证疾病名称是否合理"""
        if not disease_name or len(disease_name) < 3:
            return False

        # 检查是否包含不应该出现的内容
        invalid_patterns = [
            'AND', '&', '+', 'gene', 'expression', 'PCSK9', 'APOE',  # 基因相关
            'analysis', 'study', 'research',  # 研究相关
            'with', 'including', 'associated',  # 复合查询
        ]

        disease_lower = disease_name.lower()
        for pattern in invalid_patterns:
            if pattern.lower() in disease_lower:
                return False

        return True

    def _create_knowledge_integration_prompt(self, request: CausalAnalysisRequest, mr_result: Dict[str, Any], knowledge_result: Dict[str, Any], language: str) -> str:
        """创建知识整合的LLM提示"""

        if language == "zh":
            prompt = f"""
作为生物医学知识专家，请整合以下分析结果：

基因：{request.exposure_gene}
疾病：{request.outcome_trait}
MR分析结果：{mr_result.get('summary', {}).get('conclusion', '无结论')}

可用知识：
- 基因注释：{knowledge_result.get('gene_annotation', {}).get('gene_info', {}).get('description', '无信息')}
- 药物信息：{len(knowledge_result.get('drug_analysis', {}).get('targeting_drugs', []))} 个相关药物

请提供：
1. 综合生物学解释
2. 临床意义评估
3. 研究局限性
4. 后续研究建议

请以JSON格式回复：
{{
    "biological_explanation": "生物学机制解释",
    "clinical_significance": "临床意义",
    "limitations": "研究局限性",
    "future_directions": "后续研究方向"
}}
"""
        else:
            prompt = f"""
As a biomedical knowledge expert, please integrate the following analysis results:

Gene: {request.exposure_gene}
Disease: {request.outcome_trait}
MR analysis results: {mr_result.get('summary', {}).get('conclusion', 'No conclusion')}

Available knowledge:
- Gene annotation: {knowledge_result.get('gene_annotation', {}).get('gene_info', {}).get('description', 'No information')}
- Drug information: {len(knowledge_result.get('drug_analysis', {}).get('targeting_drugs', []))} related drugs

Please provide:
1. Comprehensive biological explanation
2. Clinical significance assessment
3. Study limitations
4. Future research suggestions

Please respond in JSON format:
{{
    "biological_explanation": "biological mechanism explanation",
    "clinical_significance": "clinical significance",
    "limitations": "study limitations",
    "future_directions": "future research directions"
}}
"""

        return prompt

    def _parse_llm_integration(self, response: str) -> Dict[str, Any]:
        """解析LLM知识整合响应"""
        import json
        import re

        try:
            # 尝试解析JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                integration_data = json.loads(json_match.group())
                return integration_data
        except:
            pass

        # 备用解析
        return {
            "biological_explanation": "Knowledge integration completed",
            "clinical_significance": "Moderate",
            "limitations": "Standard MR limitations apply",
            "future_directions": "Further validation needed"
        }

    def _create_comprehensive_interpretation_prompt(self, result, language: str) -> str:
        """创建综合解释的LLM提示"""

        # 提取MR结论和详细结果
        mr_conclusion = "无MR结果"
        mr_details = ""
        if result.mr_results and isinstance(result.mr_results, dict):
            mr_conclusion = result.mr_results.get('summary', {}).get('conclusion', '无结论')

            # 提取MR方法结果
            mr_results_list = result.mr_results.get('results', [])
            if mr_results_list:
                mr_details = "\nMR方法结果：\n"
                for method_result in mr_results_list[:3]:  # 显示前3个方法
                    method_name = method_result.get('method', 'Unknown')
                    estimate = method_result.get('estimate', 0)
                    p_value = method_result.get('p_value', 1)
                    mr_details += f"- {method_name}: β={estimate:.4f}, P={p_value:.3e}\n"

        # 提取基因注释信息
        gene_info = ""
        if result.gene_annotation:
            gene_data = result.gene_annotation.get('gene_info', {})
            if gene_data:
                protein_name = gene_data.get('protein_name', '')
                function = gene_data.get('function', '')
                if protein_name:
                    gene_info += f"\n蛋白质：{protein_name}"
                if function:
                    gene_info += f"\n功能：{function[:200]}..."

        # 提取药物信息
        drug_info = ""
        if result.drug_analysis:
            drug_targets = result.drug_analysis.get('targeting_drugs', [])
            if drug_targets:
                drug_info = f"\n相关药物：发现{len(drug_targets)}个相关药物靶点"

        if language == "zh":
            prompt = f"""
作为专业的遗传流行病学和因果推断专家，请为以下孟德尔随机化分析提供全面深入的解释：

## 分析概览
- 暴露基因：{result.request.exposure_gene}
- 结局疾病：{result.request.outcome_trait}
- 组织背景：{result.request.tissue_context}
- eQTL工具变量：{result.summary.get('n_instruments', 0)}个
- 协调SNP：{result.summary.get('n_harmonized_snps', 0)}个

## MR分析结果
{mr_conclusion}{mr_details}

## 生物学背景{gene_info}{drug_info}

请提供详细的专业解释，包括：
1. **主要发现总结** - 清晰概述因果关系证据
2. **生物学机制解释** - 基于基因功能和疾病病理的机制分析
3. **统计学证据评估** - MR方法结果的可靠性分析
4. **临床转化意义** - 对疾病预防和治疗的启示
5. **研究局限性** - 当前分析的限制和注意事项
6. **后续研究建议** - 具体的验证和扩展研究方向

请以JSON格式回复：
{{
    "interpretation": "详细的专业解释（包含以上6个方面，使用markdown格式）",
    "recommendations": ["具体建议1", "具体建议2", "具体建议3", "具体建议4", "具体建议5"],
    "summary": {{
        "causal_conclusion": "因果结论",
        "confidence_level": "高/中/低",
        "clinical_relevance": "临床相关性评估"
    }}
}}
"""
        else:
            prompt = f"""
As a professional genetic epidemiologist and causal inference expert, please provide a comprehensive and detailed interpretation for the following Mendelian Randomization analysis:

## Analysis Overview
- Exposure Gene: {result.request.exposure_gene}
- Outcome Disease: {result.request.outcome_trait}
- Tissue Context: {result.request.tissue_context}
- eQTL Instruments: {result.summary.get('n_instruments', 0)}
- Harmonized SNPs: {result.summary.get('n_harmonized_snps', 0)}

## MR Analysis Results
{mr_conclusion}{mr_details}

## Biological Context{gene_info}{drug_info}

Please provide a detailed professional interpretation covering:
1. **Main Findings Summary** - Clear overview of causal evidence
2. **Biological Mechanism Explanation** - Mechanism analysis based on gene function and disease pathology
3. **Statistical Evidence Assessment** - Reliability analysis of MR method results
4. **Clinical Translation Significance** - Implications for disease prevention and treatment
5. **Study Limitations** - Current analysis limitations and considerations
6. **Future Research Directions** - Specific validation and extension research directions

Please respond in JSON format:
{{
    "interpretation": "detailed professional interpretation (covering all 6 aspects above, using markdown format)",
    "recommendations": ["specific recommendation1", "specific recommendation2", "specific recommendation3", "specific recommendation4", "specific recommendation5"],
    "summary": {{
        "causal_conclusion": "causal conclusion",
        "confidence_level": "high/medium/low",
        "clinical_relevance": "clinical relevance assessment"
    }}
}}
"""

        return prompt

    def _parse_llm_interpretation(self, response: str) -> Dict[str, Any]:
        """解析LLM综合解释响应"""
        import json
        import re

        try:
            # 尝试解析JSON
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                interpretation_data = json.loads(json_match.group())
                # 验证必要字段是否存在
                if interpretation_data.get("interpretation") and interpretation_data.get("recommendations"):
                    return interpretation_data
        except Exception as e:
            logger.warning(f"JSON解析失败: {e}")

        # 如果JSON解析失败，尝试从原始响应中提取内容
        logger.info("尝试从原始LLM响应中提取内容")

        # 提取解释内容（查找常见的标题模式）
        interpretation_content = response
        if "interpretation" in response.lower() or "解释" in response:
            # 尝试提取解释部分
            lines = response.split('\n')
            interpretation_lines = []
            in_interpretation = False

            for line in lines:
                if any(keyword in line.lower() for keyword in ["interpretation", "解释", "分析结果", "主要发现"]):
                    in_interpretation = True
                    continue
                elif any(keyword in line.lower() for keyword in ["recommendation", "建议", "suggestions"]):
                    break
                elif in_interpretation and line.strip():
                    interpretation_lines.append(line.strip())

            if interpretation_lines:
                interpretation_content = '\n'.join(interpretation_lines)

        # 备用解析 - 使用原始响应内容
        return {
            "interpretation": interpretation_content if interpretation_content.strip() else "LLM生成了响应，但格式解析失败。请查看详细的MR分析结果。",
            "recommendations": [
                "查看MR分析结果中的因果证据",
                "考虑生物学背景和机制",
                "通过额外研究验证发现",
                "评估研究的局限性",
                "考虑临床转化的可能性"
            ],
            "summary": {
                "causal_conclusion": "分析已完成",
                "confidence_level": "中等",
                "clinical_relevance": "中等"
            }
        }

    def _parse_llm_methods(self, response: str) -> List[str]:
        """解析LLM方法选择响应"""
        methods = []

        # 查找常见MR方法
        common_methods = ["IVW", "Weighted Median", "MR-Egger", "Weighted Mode", "Simple Mode"]

        for method in common_methods:
            if method.lower() in response.lower():
                methods.append(method)

        # 如果没有找到，使用默认方法
        if not methods:
            methods = ["IVW", "Weighted Median"]

        return methods

    def _parse_llm_integration(self, response: str) -> Dict[str, Any]:
        """解析LLM知识整合响应"""
        return {
            "integrated_summary": response[:500],  # 取前500字符
            "key_insights": [response[:200]],  # 简化处理
            "biological_context": response[200:400] if len(response) > 200 else ""
        }

    def _parse_llm_interpretation(self, response: str) -> Dict[str, Any]:
        """解析LLM综合解释响应"""
        import json
        import re

        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                interpretation = json.loads(json_match.group())
                return interpretation
        except:
            pass

        # 备用解析：将响应分段
        lines = [line.strip() for line in response.split('\n') if line.strip()]

        interpretation = lines[0] if lines else "分析完成"
        recommendations = lines[1:3] if len(lines) > 1 else ["查看详细结果"]

        return {
            "interpretation": interpretation,
            "recommendations": recommendations,
            "summary": {"status": "completed", "llm_generated": True}
        }
    
    async def continue_analysis_with_clarification(self, clarified_input: str, language: str = "zh") -> CausalAnalysisResult:
        """
        基于用户澄清继续分析
        
        Args:
            clarified_input: 用户澄清后的输入
            language: 语言代码
            
        Returns:
            CausalAnalysisResult: 分析结果
        """
        # 重新解析澄清后的输入
        return await self.execute_analysis(clarified_input, language)
