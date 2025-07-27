#!/usr/bin/env python3
"""
数据质量控制模块
用于验证和评估MCP服务返回的数据质量
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class DataQualityLevel(Enum):
    """数据质量等级"""
    EXCELLENT = "excellent"  # 优秀 (>= 0.9)
    GOOD = "good"           # 良好 (>= 0.7)
    FAIR = "fair"           # 一般 (>= 0.5)
    POOR = "poor"           # 较差 (< 0.5)

@dataclass
class QualityMetrics:
    """数据质量指标"""
    completeness: float  # 数据完整性 (0-1)
    consistency: float   # 数据一致性 (0-1)
    accuracy: float      # 数据准确性 (0-1)
    timeliness: float    # 数据时效性 (0-1)
    overall_score: float # 总体质量评分 (0-1)
    issues: List[str]    # 发现的问题列表

class DataQualityController:
    """数据质量控制器"""
    
    def __init__(self):
        self.min_quality_threshold = 0.7
        self.logger = logging.getLogger(__name__)
    
    def validate_eqtl_data(self, eqtl_data: Dict[str, Any]) -> QualityMetrics:
        """验证eQTL数据质量"""
        issues = []
        
        # 检查数据完整性
        completeness = self._check_eqtl_completeness(eqtl_data, issues)
        
        # 检查数据一致性
        consistency = self._check_eqtl_consistency(eqtl_data, issues)
        
        # 检查数据准确性
        accuracy = self._check_eqtl_accuracy(eqtl_data, issues)
        
        # 时效性（对于eQTL数据，主要检查数据来源）
        timeliness = self._check_data_source_timeliness(eqtl_data, issues)
        
        # 计算总体评分
        overall_score = (completeness + consistency + accuracy + timeliness) / 4
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            overall_score=overall_score,
            issues=issues
        )
    
    def validate_gwas_data(self, gwas_data: Dict[str, Any]) -> QualityMetrics:
        """验证GWAS数据质量"""
        issues = []
        
        # 检查数据完整性
        completeness = self._check_gwas_completeness(gwas_data, issues)
        
        # 检查数据一致性
        consistency = self._check_gwas_consistency(gwas_data, issues)
        
        # 检查数据准确性
        accuracy = self._check_gwas_accuracy(gwas_data, issues)
        
        # 时效性
        timeliness = self._check_data_source_timeliness(gwas_data, issues)
        
        # 计算总体评分
        overall_score = (completeness + consistency + accuracy + timeliness) / 4
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            overall_score=overall_score,
            issues=issues
        )
    
    def validate_mr_results(self, mr_data: Dict[str, Any]) -> QualityMetrics:
        """验证MR分析结果质量"""
        issues = []
        
        # 检查结果完整性
        completeness = self._check_mr_completeness(mr_data, issues)
        
        # 检查结果一致性
        consistency = self._check_mr_consistency(mr_data, issues)
        
        # 检查统计有效性
        accuracy = self._check_mr_statistical_validity(mr_data, issues)
        
        # 时效性（计算是否基于最新数据）
        timeliness = 1.0  # MR计算是实时的
        
        # 计算总体评分
        overall_score = (completeness + consistency + accuracy + timeliness) / 4
        
        return QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            overall_score=overall_score,
            issues=issues
        )
    
    def _check_eqtl_completeness(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查eQTL数据完整性"""
        if not data:
            issues.append("eQTL数据为空")
            return 0.0
        
        instruments = data.get('instruments', [])
        if not instruments:
            issues.append("无eQTL工具变量")
            return 0.0
        
        # 检查必需字段
        required_fields = ['snp_id', 'beta', 'se', 'p_value']
        complete_instruments = 0
        
        for instrument in instruments:
            if all(field in instrument and instrument[field] is not None 
                   for field in required_fields):
                complete_instruments += 1
        
        completeness = complete_instruments / len(instruments)
        
        if completeness < 1.0:
            missing_count = len(instruments) - complete_instruments
            issues.append(f"{missing_count}个工具变量缺少必需字段")
        
        return completeness
    
    def _check_eqtl_consistency(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查eQTL数据一致性"""
        instruments = data.get('instruments', [])
        if not instruments:
            return 0.0
        
        # 检查数据类型一致性
        consistency_score = 1.0
        
        for instrument in instruments:
            # 检查beta值合理性
            beta = instrument.get('beta')
            if beta is not None and abs(beta) > 5:
                issues.append(f"SNP {instrument.get('snp_id')} beta值异常: {beta}")
                consistency_score -= 0.1
            
            # 检查p值合理性
            p_value = instrument.get('p_value')
            if p_value is not None and (p_value < 0 or p_value > 1):
                issues.append(f"SNP {instrument.get('snp_id')} p值异常: {p_value}")
                consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _check_eqtl_accuracy(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查eQTL数据准确性"""
        data_source = data.get('data_source', '')
        
        # 基于数据来源评估准确性
        if 'Real_GTEx_Data' in data_source:
            return 1.0
        elif 'GTEx' in data_source and 'real' in data_source.lower():
            return 0.9
        elif 'mock' in data_source.lower() or 'simulate' in data_source.lower():
            issues.append("检测到测试数据")
            return 0.0
        else:
            issues.append(f"未知数据来源: {data_source}")
            return 0.5
    
    def _check_gwas_completeness(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查GWAS数据完整性"""
        harmonized_data = data.get('harmonized_data', [])
        if not harmonized_data:
            issues.append("无GWAS协调数据")
            return 0.0
        
        # 检查必需字段
        complete_snps = 0
        for snp_data in harmonized_data:
            outcome_data = snp_data.get('outcome_data', {})
            if all(field in outcome_data for field in ['beta', 'se', 'pval']):
                complete_snps += 1
        
        completeness = complete_snps / len(harmonized_data)
        
        if completeness < 1.0:
            missing_count = len(harmonized_data) - complete_snps
            issues.append(f"{missing_count}个SNP缺少完整GWAS数据")
        
        return completeness
    
    def _check_gwas_consistency(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查GWAS数据一致性"""
        harmonized_data = data.get('harmonized_data', [])
        if not harmonized_data:
            return 0.0
        
        consistency_score = 1.0
        
        for snp_data in harmonized_data:
            outcome_data = snp_data.get('outcome_data', {})
            
            # 检查beta值合理性
            beta = outcome_data.get('beta')
            if beta is not None and abs(beta) > 10:
                issues.append(f"GWAS beta值异常: {beta}")
                consistency_score -= 0.1
            
            # 检查p值合理性
            pval = outcome_data.get('pval')
            if pval is not None and (pval < 0 or pval > 1):
                issues.append(f"GWAS p值异常: {pval}")
                consistency_score -= 0.1
        
        return max(0.0, consistency_score)
    
    def _check_gwas_accuracy(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查GWAS数据准确性"""
        data_source = data.get('data_source', '')
        
        # 基于数据来源评估准确性
        if 'OpenGWAS_API_Real' in data_source:
            return 1.0
        elif 'EBI_GWAS_Catalog' in data_source:
            return 0.9
        elif 'mock' in data_source.lower() or 'simulate' in data_source.lower():
            issues.append("检测到测试GWAS数据")
            return 0.0
        else:
            issues.append(f"未知GWAS数据来源: {data_source}")
            return 0.5
    
    def _check_mr_completeness(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查MR结果完整性"""
        results = data.get('results', [])
        if not results:
            issues.append("无MR分析结果")
            return 0.0
        
        # 检查必需字段
        complete_results = 0
        for result in results:
            if all(field in result for field in ['method', 'estimate', 'se', 'p_value']):
                complete_results += 1
        
        completeness = complete_results / len(results)
        
        if completeness < 1.0:
            missing_count = len(results) - complete_results
            issues.append(f"{missing_count}个MR结果缺少必需字段")
        
        return completeness
    
    def _check_mr_consistency(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查MR结果一致性"""
        results = data.get('results', [])
        if not results:
            return 0.0
        
        # 检查不同方法结果的一致性
        estimates = [r.get('estimate') for r in results if r.get('estimate') is not None]
        
        if len(estimates) < 2:
            return 1.0
        
        # 计算估计值的变异系数
        mean_estimate = sum(estimates) / len(estimates)
        if mean_estimate == 0:
            return 1.0
        
        variance = sum((e - mean_estimate) ** 2 for e in estimates) / len(estimates)
        cv = (variance ** 0.5) / abs(mean_estimate)
        
        # CV < 0.5 认为一致性较好
        consistency = max(0.0, 1.0 - cv)
        
        if consistency < 0.7:
            issues.append(f"MR方法间结果差异较大 (CV: {cv:.2f})")
        
        return consistency
    
    def _check_mr_statistical_validity(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查MR统计有效性"""
        results = data.get('results', [])
        if not results:
            return 0.0
        
        validity_score = 1.0
        
        for result in results:
            # 检查置信区间合理性
            ci_lower = result.get('ci_lower')
            ci_upper = result.get('ci_upper')
            
            if ci_lower is not None and ci_upper is not None:
                if ci_lower > ci_upper:
                    issues.append(f"置信区间异常: [{ci_lower}, {ci_upper}]")
                    validity_score -= 0.2
        
        return max(0.0, validity_score)
    
    def _check_data_source_timeliness(self, data: Dict[str, Any], issues: List[str]) -> float:
        """检查数据来源时效性"""
        data_source = data.get('data_source', '')
        
        # 真实数据源认为是最新的
        if any(keyword in data_source for keyword in ['Real', 'API', 'GTEx', 'OpenGWAS']):
            return 1.0
        elif 'mock' in data_source.lower() or 'simulate' in data_source.lower():
            issues.append("使用了测试数据")
            return 0.0
        else:
            return 0.8
    
    def is_quality_acceptable(self, metrics: QualityMetrics) -> bool:
        """判断数据质量是否可接受"""
        return metrics.overall_score >= self.min_quality_threshold
    
    def get_quality_level(self, metrics: QualityMetrics) -> DataQualityLevel:
        """获取数据质量等级"""
        if metrics.overall_score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif metrics.overall_score >= 0.7:
            return DataQualityLevel.GOOD
        elif metrics.overall_score >= 0.5:
            return DataQualityLevel.FAIR
        else:
            return DataQualityLevel.POOR

    def get_quality_summary(self, metrics: QualityMetrics) -> str:
        """获取质量评估摘要"""
        level = self.get_quality_level(metrics)
        level_names = {
            DataQualityLevel.EXCELLENT: "优秀",
            DataQualityLevel.GOOD: "良好",
            DataQualityLevel.FAIR: "一般",
            DataQualityLevel.POOR: "较差"
        }

        return f"数据质量: {level_names[level]} (评分: {metrics.overall_score:.3f})"

    def get_quality_level(self, metrics: QualityMetrics) -> DataQualityLevel:
        """根据评分获取质量等级"""
        score = metrics.overall_score

        if score >= 0.9:
            return DataQualityLevel.EXCELLENT
        elif score >= 0.7:
            return DataQualityLevel.GOOD
        elif score >= 0.5:
            return DataQualityLevel.FAIR
        else:
            return DataQualityLevel.POOR

    def validate_harmonized_data(self, harmonized_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], QualityMetrics]:
        """验证协调后的数据质量"""
        issues = []

        if not harmonized_data:
            return [], QualityMetrics(
                completeness=0.0,
                consistency=0.0,
                accuracy=0.0,
                timeliness=1.0,
                overall_score=0.0,
                issues=["No harmonized data available"]
            )

        # 检查数据完整性
        completeness = self._check_harmonized_completeness(harmonized_data, issues)

        # 检查数据一致性
        consistency = self._check_harmonized_consistency(harmonized_data, issues)

        # 检查数据准确性
        accuracy = self._check_harmonized_accuracy(harmonized_data, issues)

        # 时效性（协调数据通常是实时的）
        timeliness = 1.0

        # 计算总体评分
        overall_score = (completeness + consistency + accuracy + timeliness) / 4

        metrics = QualityMetrics(
            completeness=completeness,
            consistency=consistency,
            accuracy=accuracy,
            timeliness=timeliness,
            overall_score=overall_score,
            issues=issues
        )

        # 过滤掉质量过低的数据点
        validated_data = self._filter_low_quality_harmonized_data(harmonized_data)

        return validated_data, metrics

    def _check_harmonized_completeness(self, data: List[Dict[str, Any]], issues: List[str]) -> float:
        """检查协调数据的完整性"""
        if not data:
            return 0.0

        required_fields = ['snp_id', 'beta', 'se', 'p_value', 'effect_allele', 'other_allele']
        total_fields = len(required_fields) * len(data)
        missing_fields = 0

        for item in data:
            for field in required_fields:
                if field not in item or item[field] is None:
                    missing_fields += 1

        completeness = 1.0 - (missing_fields / total_fields)

        if missing_fields > 0:
            issues.append(f"协调数据中有 {missing_fields} 个缺失字段")

        return completeness

    def _check_harmonized_consistency(self, data: List[Dict[str, Any]], issues: List[str]) -> float:
        """检查协调数据的一致性"""
        if len(data) < 2:
            return 1.0

        consistency_score = 1.0

        # 检查效应方向的一致性
        positive_effects = sum(1 for item in data if item.get('beta', 0) > 0)
        negative_effects = len(data) - positive_effects

        if positive_effects > 0 and negative_effects > 0:
            # 有正负效应混合，检查是否合理
            ratio = min(positive_effects, negative_effects) / len(data)
            if ratio > 0.4:  # 如果正负效应比例都很高，可能有问题
                consistency_score *= 0.8
                issues.append("协调数据中正负效应混合，需要注意")

        return consistency_score

    def _check_harmonized_accuracy(self, data: List[Dict[str, Any]], issues: List[str]) -> float:
        """检查协调数据的准确性"""
        accuracy_score = 1.0

        for item in data:
            # 检查p值范围
            p_value = item.get('p_value')
            if p_value is not None:
                if p_value < 0 or p_value > 1:
                    accuracy_score *= 0.5
                    issues.append(f"SNP {item.get('snp_id', 'unknown')} 的p值超出有效范围")

            # 检查标准误
            se = item.get('se')
            if se is not None and se <= 0:
                accuracy_score *= 0.8
                issues.append(f"SNP {item.get('snp_id', 'unknown')} 的标准误无效")

        return accuracy_score

    def _filter_low_quality_harmonized_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """过滤低质量的协调数据"""
        filtered_data = []

        for item in data:
            # 基本质量检查
            if (item.get('p_value') is not None and
                0 <= item.get('p_value') <= 1 and
                item.get('se') is not None and
                item.get('se') > 0 and
                item.get('beta') is not None):
                filtered_data.append(item)

        return filtered_data
