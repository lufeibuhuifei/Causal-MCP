"""
输入验证器模块

提供快速的输入参数验证功能，使用预加载的数据集进行O(1)时间复杂度的验证。
"""

import pickle
import logging
from pathlib import Path
from typing import Set, Optional, Dict, List, Tuple
import difflib

try:
    from .i18n import get_text
    from .disease_mapper import DiseaseMapper
except ImportError:
    from i18n import get_text
    from disease_mapper import DiseaseMapper

logger = logging.getLogger(__name__)

class InputValidator:
    """输入验证器类"""
    
    def __init__(self):
        """初始化验证器"""
        self.validation_data = None
        self.valid_genes: Set[str] = set()
        self.valid_gwas_trait_names: Set[str] = set()
        self.valid_gwas_study_ids: Set[str] = set()
        self.valid_tissues: Set[str] = set()

        # 模糊匹配缓存
        self._fuzzy_cache: Dict[str, List[str]] = {}

        # 初始化智能疾病映射器
        self.disease_mapper = DiseaseMapper()

        self.load_validation_data()
    
    def load_validation_data(self) -> bool:
        """加载验证数据"""
        try:
            # 查找验证数据文件
            pickle_path = Path("validation_data.pkl")

            if not pickle_path.exists():
                logger.warning("验证数据文件不存在")
                self._load_fallback_data()
                return False

            # 加载pickle数据
            with open(pickle_path, "rb") as f:
                self.validation_data = pickle.load(f)

            # 提取各类数据到集合中（快速查找）
            self.valid_genes = self.validation_data.get("genes", set())
            # GWAS数据合并了性状名称和研究ID
            gwas_all = self.validation_data.get("gwas_traits", set())
            self.valid_gwas_trait_names = gwas_all
            self.valid_gwas_study_ids = gwas_all
            self.valid_tissues = self.validation_data.get("tissues", set())
            
            logger.info(f"✅ 验证数据加载成功:")
            logger.info(f"   基因: {len(self.valid_genes)}")
            logger.info(f"   GWAS性状: {len(self.valid_gwas_trait_names)}")
            logger.info(f"   GWAS ID: {len(self.valid_gwas_study_ids)}")
            logger.info(f"   组织: {len(self.valid_tissues)}")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载验证数据失败: {e}")
            self._load_fallback_data()
            return False
    
    def _load_fallback_data(self):
        """加载备用验证数据"""
        logger.info("使用备用验证数据...")
        
        self.valid_genes = {
            "PCSK9", "LPA", "APOE", "IL6R", "LDLR", "BRCA1", "BRCA2", "TP53",
            "EGFR", "KRAS", "PIK3CA", "AKT1", "MTOR", "VEGFA", "HIF1A",
            "TNF", "IL1B", "IL6", "IFNG", "TGFB1", "PDGFA", "FGF2", "IGF1"
        }
        
        self.valid_gwas_trait_names = {
            "Coronary Artery Disease", "Type 2 Diabetes", "LDL Cholesterol",
            "Body Mass Index", "Alzheimer's Disease", "Rheumatoid Arthritis",
            "Breast Cancer", "Prostate Cancer", "Hypertension", "Stroke"
        }
        
        self.valid_gwas_study_ids = {
            "ieu-a-7", "ieu-a-26", "ieu-a-89", "ieu-a-300", "ieu-a-297",
            "ieu-a-832", "ieu-a-1126", "ieu-a-957", "ukb-b-19953", "ukb-b-12493"
        }
        
        self.valid_tissues = {
            "Whole_Blood", "Liver", "Brain_Cortex", "Heart_Left_Ventricle",
            "Lung", "Muscle_Skeletal", "Adipose_Subcutaneous", "Skin_Sun_Exposed",
            "Thyroid", "Pancreas"
        }
    
    def validate_gene(self, gene_symbol: str, language: str = "zh") -> Tuple[bool, Optional[str]]:
        """
        验证基因符号

        Args:
            gene_symbol: 基因符号
            language: 语言代码

        Returns:
            (是否有效, 错误信息或None)
        """
        if not gene_symbol:
            return False, get_text("gene_symbol_empty", language)

        gene_clean = gene_symbol.upper().strip()

        if gene_clean in self.valid_genes:
            return True, None
        else:
            # 提供相似建议
            suggestions = self.get_gene_suggestions(gene_clean)
            if suggestions:
                return False, get_text("gene_not_found_with_suggestions", language).format(gene_symbol, ', '.join(suggestions[:3]))
            else:
                return False, get_text("gene_not_found", language).format(gene_symbol)
    
    def validate_gwas_trait(self, trait_input: str, language: str = "zh") -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        验证GWAS性状或研究ID，使用智能疾病映射系统
        增强版：提供友好提示但不阻止用户操作

        Args:
            trait_input: 性状名称或研究ID
            language: 语言代码

        Returns:
            (是否有效, 消息内容, 额外信息字典)
        """
        if not trait_input:
            return False, get_text("outcome_trait_empty", language), None

        # 使用原始输入进行验证，不进行自动清理
        trait_input_clean = trait_input.strip()

        # 使用智能疾病映射系统进行验证
        validation_result = self.disease_mapper.validate_input(trait_input_clean)

        if validation_result["is_valid"]:
            # Exact match found
            message = f"✅ Match found: {validation_result['matched_result']}"

            return True, message, {
                "match_type": "exact",
                "original_input": trait_input,
                "matched_result": validation_result['matched_result']
            }
        else:
            # No exact match, but provide friendly suggestions without blocking operation
            suggestions = validation_result.get("recommendations", [])

            if suggestions:
                # 有建议，友好提示
                message = f"💡 No exact match found for '{trait_input_clean}', but you can continue analysis\n"
                message += f"Similar options: {', '.join(suggestions[:3])}"
                if len(suggestions) > 3:
                    message += f" and {len(suggestions) - 3} more options"
                message += f"\n💭 Tip: System will still search for related studies in GWAS database"
            else:
                # 无建议，鼓励性提示
                message = f"🔍 '{trait_input_clean}' not found in disease database, but you can still continue analysis\n"
                message += f"💡 Tip: You can use GWAS study ID (e.g., ieu-b-110) or try other disease names"

            return True, message, {  # 注意：这里返回True，不阻止用户操作
                "match_type": "no_match_but_allowed",
                "original_input": trait_input,
                "suggestions": suggestions,
                "warning": True
            }


    
    def validate_tissue(self, tissue_name: str, language: str = "zh") -> Tuple[bool, Optional[str]]:
        """
        验证组织名称

        Args:
            tissue_name: 组织名称
            language: 语言代码

        Returns:
            (是否有效, 错误信息或None)
        """
        if not tissue_name:
            return False, get_text("tissue_context_empty", language)

        tissue_clean = tissue_name.strip()

        if tissue_clean in self.valid_tissues:
            return True, None
        else:
            # 提供相似建议
            suggestions = self.get_tissue_suggestions(tissue_clean)
            if suggestions:
                return False, get_text("tissue_not_found_with_suggestions", language).format(tissue_name, ', '.join(suggestions[:3]))
            else:
                return False, get_text("tissue_not_found", language).format(tissue_name)
    
    def get_gene_suggestions(self, partial_gene: str, limit: int = 5) -> List[str]:
        """获取基因符号建议"""
        return self._get_fuzzy_matches(partial_gene, self.valid_genes, limit)
    
    def get_gwas_suggestions(self, partial_trait: str, limit: int = 5) -> List[str]:
        """获取GWAS性状建议"""
        # 合并性状名称和研究ID
        all_gwas = self.valid_gwas_trait_names | self.valid_gwas_study_ids
        return self._get_fuzzy_matches(partial_trait, all_gwas, limit)
    
    def get_tissue_suggestions(self, partial_tissue: str, limit: int = 5) -> List[str]:
        """获取组织名称建议"""
        return self._get_fuzzy_matches(partial_tissue, self.valid_tissues, limit)
    
    def _get_fuzzy_matches(self, query: str, candidates: Set[str], limit: int) -> List[str]:
        """获取模糊匹配结果"""
        if not query:
            return []
        
        # 检查缓存
        cache_key = f"{query}_{len(candidates)}_{limit}"
        if cache_key in self._fuzzy_cache:
            return self._fuzzy_cache[cache_key]
        
        query_lower = query.lower()
        matches = []
        
        # 1. Exact match (case insensitive)
        for candidate in candidates:
            if candidate.lower() == query_lower:
                matches.append(candidate)
        
        # 2. 前缀匹配
        if len(matches) < limit:
            for candidate in candidates:
                if candidate.lower().startswith(query_lower) and candidate not in matches:
                    matches.append(candidate)
                    if len(matches) >= limit:
                        break
        
        # 3. 包含匹配
        if len(matches) < limit:
            for candidate in candidates:
                if query_lower in candidate.lower() and candidate not in matches:
                    matches.append(candidate)
                    if len(matches) >= limit:
                        break
        
        # 4. 相似度匹配
        if len(matches) < limit:
            remaining_candidates = [c for c in candidates if c not in matches]
            similar = difflib.get_close_matches(
                query, remaining_candidates, 
                n=limit-len(matches), cutoff=0.6
            )
            matches.extend(similar)
        
        # 缓存结果
        self._fuzzy_cache[cache_key] = matches[:limit]
        return matches[:limit]
    
    def get_validation_stats(self) -> Dict[str, int]:
        """获取验证数据统计信息"""
        return {
            "genes": len(self.valid_genes),
            "gwas_trait_names": len(self.valid_gwas_trait_names),
            "gwas_study_ids": len(self.valid_gwas_study_ids),
            "tissues": len(self.valid_tissues)
        }

# 全局验证器实例
_validator_instance: Optional[InputValidator] = None

def get_validator() -> InputValidator:
    """获取全局验证器实例"""
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = InputValidator()
    return _validator_instance
