# client-app/natural_language_parser.py
"""
自然语言输入解析器
使用LLM解析用户的自然语言输入，提取基因、疾病、组织等参数
"""

import logging
import json
import re
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ParsedParameters:
    """解析后的参数"""
    gene: Optional[str] = None
    disease: Optional[str] = None
    tissue: Optional[str] = None
    confidence: float = 0.0
    raw_input: str = ""
    missing_params: List[str] = None
    suggestions: List[str] = None

class NaturalLanguageParser:
    """自然语言输入解析器"""
    
    def __init__(self, llm_service, input_validator):
        """
        初始化解析器

        Args:
            llm_service: LLM服务实例
            input_validator: 输入验证器实例
        """
        self.llm_service = llm_service
        self.input_validator = input_validator
        
        # 预定义的模式匹配规则（备用方案）
        self.gene_patterns = [
            r'([A-Z][A-Z0-9]{2,10})\s*基因',
            r'([A-Z][A-Z0-9]{2,10})\s*gene',
            r'基因\s*([A-Z][A-Z0-9]{2,10})',
            r'gene\s*([A-Z][A-Z0-9]{2,10})',
            r'\b([A-Z]{3,10})\b',  # 匹配3-10个大写字母的基因符号
            r'([A-Z][A-Z0-9]{2,10})'  # 最后匹配基因符号模式
        ]
        
        self.tissue_patterns = [
            r'(whole blood|blood)',
            r'(liver)',
            r'(brain|cortex)',
            r'(heart)',
            r'(muscle)',
            r'(lung)',
            r'(kidney)'
        ]
        
        # 组织映射 - 基于GTEx v8的49个组织
        self.tissue_mapping = {
            # 中文映射
            '全血': 'Whole_Blood',
            '肝脏': 'Liver',
            '大脑': 'Brain_Cortex',
            '脑': 'Brain_Cortex',
            '大脑皮层': 'Brain_Cortex',
            '小脑': 'Brain_Cerebellum',
            '海马': 'Brain_Hippocampus',
            '额叶皮层': 'Brain_Frontal_Cortex_BA9',
            '心脏': 'Heart_Left_Ventricle',
            '左心室': 'Heart_Left_Ventricle',
            '心房': 'Heart_Atrial_Appendage',
            '肌肉': 'Muscle_Skeletal',
            '骨骼肌': 'Muscle_Skeletal',
            '肺': 'Lung',
            '肾': 'Kidney_Cortex',
            '肾皮质': 'Kidney_Cortex',
            '脂肪': 'Adipose_Subcutaneous',
            '皮下脂肪': 'Adipose_Subcutaneous',
            '内脏脂肪': 'Adipose_Visceral_Omentum',
            '皮肤': 'Skin_Sun_Exposed_Lower_leg',
            '甲状腺': 'Thyroid',
            '胰腺': 'Pancreas',
            '脾脏': 'Spleen',
            '胃': 'Stomach',
            '结肠': 'Colon_Transverse',
            '食管': 'Esophagus_Mucosa',
            '小肠': 'Small_Intestine_Terminal_Ileum',
            '肾上腺': 'Adrenal_Gland',
            '神经': 'Nerve_Tibial',
            '垂体': 'Pituitary',
            '卵巢': 'Ovary',
            '睾丸': 'Testis',
            '前列腺': 'Prostate',
            '子宫': 'Uterus',
            '阴道': 'Vagina',
            '乳腺': 'Breast_Mammary_Tissue',
            '唾液腺': 'Minor_Salivary_Gland',

            # 英文映射
            'whole blood': 'Whole_Blood',
            'blood': 'Whole_Blood',
            'liver': 'Liver',
            'brain': 'Brain_Cortex',
            'cortex': 'Brain_Cortex',
            'brain cortex': 'Brain_Cortex',
            'cerebellum': 'Brain_Cerebellum',
            'hippocampus': 'Brain_Hippocampus',
            'frontal cortex': 'Brain_Frontal_Cortex_BA9',
            'heart': 'Heart_Left_Ventricle',
            'left ventricle': 'Heart_Left_Ventricle',
            'atrial': 'Heart_Atrial_Appendage',
            'muscle': 'Muscle_Skeletal',
            'skeletal muscle': 'Muscle_Skeletal',
            'lung': 'Lung',
            'kidney': 'Kidney_Cortex',
            'kidney cortex': 'Kidney_Cortex',
            'adipose': 'Adipose_Subcutaneous',
            'fat': 'Adipose_Subcutaneous',
            'subcutaneous': 'Adipose_Subcutaneous',
            'visceral': 'Adipose_Visceral_Omentum',
            'skin': 'Skin_Sun_Exposed_Lower_leg',
            'thyroid': 'Thyroid',
            'pancreas': 'Pancreas',
            'spleen': 'Spleen',
            'stomach': 'Stomach',
            'colon': 'Colon_Transverse',
            'esophagus': 'Esophagus_Mucosa',
            'small intestine': 'Small_Intestine_Terminal_Ileum',
            'adrenal': 'Adrenal_Gland',
            'nerve': 'Nerve_Tibial',
            'pituitary': 'Pituitary',
            'ovary': 'Ovary',
            'testis': 'Testis',
            'prostate': 'Prostate',
            'uterus': 'Uterus',
            'vagina': 'Vagina',
            'breast': 'Breast_Mammary_Tissue',
            'mammary': 'Breast_Mammary_Tissue',
            'salivary': 'Minor_Salivary_Gland'
        }
    
    async def parse_input(self, user_input: str, language: str = "zh") -> ParsedParameters:
        """
        解析用户的自然语言输入
        
        Args:
            user_input: 用户输入的自然语言
            language: 语言代码
            
        Returns:
            ParsedParameters: 解析结果
        """
        logger.info(f"开始解析用户输入: {user_input}")
        
        # 首先尝试使用LLM解析
        if self.llm_service.is_available:
            try:
                llm_result = await self._parse_with_llm(user_input, language)
                if llm_result and llm_result.confidence > 0.7:
                    logger.info("✅ LLM解析成功")
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM解析失败，使用规则解析: {e}")
        
        # 备用方案：使用规则解析
        rule_result = self._parse_with_rules(user_input, language)
        logger.info("使用规则解析完成")
        return rule_result
    
    async def _parse_with_llm(self, user_input: str, language: str) -> Optional[ParsedParameters]:
        """使用LLM解析用户输入"""
        
        if language == "zh":
            prompt = f"""
你是一个专业的生物医学信息提取专家。请从用户的自然语言输入中提取基因-疾病因果推断分析所需的参数。

用户输入: "{user_input}"

请提取以下信息：
1. 基因名称（基因符号，如PCSK9、APOE、LDLR、LPA、BRCA1、TP53等）
2. 疾病名称（如冠心病、糖尿病、阿尔茨海默病、高血压、癌症等）
3. 组织类型（如全血、肝脏、大脑、心脏、肌肉等）

基因识别规则：
- 基因符号通常是2-15个字符的大写字母和数字组合
- 常见格式：PCSK9, APOE, LDLR, LPA, BRCA1, BRCA2, TP53, EGFR, KRAS等
- 可能包含数字：IL6R, HLA-A, CYP2D6等
- 宽松识别：即使不在已知列表中，符合格式的也接受

疾病识别规则：
- 包括常见疾病：心血管疾病、糖尿病、神经系统疾病、癌症等
- 包括症状和综合征：高血压、肥胖、抑郁症等
- 宽松识别：接受合理的疾病描述，即使不是标准术语

组织类型识别和默认规则：
- 如果明确提到组织：大脑/脑→Brain_Cortex，肝脏→Liver，心脏→Heart_Left_Ventricle，肺→Lung等
- 如果没有明确指定组织，根据疾病类型推荐默认组织：
  * 心血管疾病（冠心病、高血压等）→ Whole_Blood（全血）
  * 神经疾病（阿尔茨海默病、帕金森病等）→ Brain_Cortex（大脑皮层）
  * 代谢疾病（糖尿病、肥胖等）→ Whole_Blood（全血）
  * 癌症类疾病 → 根据癌症类型：乳腺癌→Breast_Mammary_Tissue，肺癌→Lung，肝癌→Liver
  * 其他疾病 → Whole_Blood（全血，最常用的组织）

请以JSON格式返回结果：
{{
    "gene": "基因符号或null",
    "disease": "疾病名称或null",
    "tissue": "推荐的组织类型（明确指定或智能推荐）",
    "confidence": 0.0-1.0的置信度,
    "missing_params": ["缺失的参数列表"],
    "reasoning": "提取理由和组织选择依据"
}}

重要提示：
- 优先宽松识别，不要因为不确定而拒绝合理的候选
- 必须为tissue字段提供值，不能为null（除非完全无法确定）
- 如果用户明确指定组织，使用用户指定的；否则根据疾病类型智能推荐
- 置信度反映提取的准确性
- 如果某些参数缺失，在missing_params中列出
"""
        else:
            prompt = f"""
You are a professional biomedical information extraction expert. Please extract parameters needed for gene-disease causal inference analysis from the user's natural language input.

User input: "{user_input}"

Please extract the following information:
1. Gene name (gene symbol, e.g., PCSK9, APOE, LDLR, LPA, BRCA1, TP53, etc.)
2. Disease name (e.g., coronary heart disease, diabetes, Alzheimer's disease, hypertension, cancer, etc.)
3. Tissue type (e.g., whole blood, liver, brain, heart, muscle, etc.)

Gene identification rules:
- Gene symbols are usually 2-15 character combinations of uppercase letters and numbers
- Common formats: PCSK9, APOE, LDLR, LPA, BRCA1, BRCA2, TP53, EGFR, KRAS, etc.
- May contain numbers: IL6R, HLA-A, CYP2D6, etc.
- Liberal identification: accept reasonable candidates even if not in known lists

Disease identification rules:
- Include common diseases: cardiovascular disease, diabetes, neurological diseases, cancer, etc.
- Include symptoms and syndromes: hypertension, obesity, depression, etc.
- Liberal identification: accept reasonable disease descriptions even if not standard terminology

Tissue type identification and default rules:
- If tissue is explicitly mentioned: brain→Brain_Cortex, liver→Liver, heart→Heart_Left_Ventricle, lung→Lung, etc.
- If no tissue is explicitly specified, recommend default tissue based on disease type:
  * Cardiovascular diseases (coronary heart disease, hypertension, etc.) → Whole_Blood
  * Neurological diseases (Alzheimer's disease, Parkinson's disease, etc.) → Brain_Cortex
  * Metabolic diseases (diabetes, obesity, etc.) → Whole_Blood
  * Cancer diseases → Based on cancer type: breast cancer→Breast_Mammary_Tissue, lung cancer→Lung, liver cancer→Liver
  * Other diseases → Whole_Blood (most commonly used tissue)

Please return the result in JSON format:
{{
    "gene": "gene symbol or null",
    "disease": "disease name or null",
    "tissue": "recommended tissue type (explicitly specified or intelligently recommended)",
    "confidence": 0.0-1.0 confidence score,
    "missing_params": ["list of missing parameters"],
    "reasoning": "extraction reasoning and tissue selection basis"
}}

Important notes:
- Prioritize liberal identification, don't reject reasonable candidates due to uncertainty
- Must provide a value for the tissue field, cannot be null (unless completely indeterminate)
- If user explicitly specifies tissue, use user's specification; otherwise intelligently recommend based on disease type
- Confidence reflects extraction accuracy
- If some parameters are missing, list them in missing_params
"""
        
        try:
            response = await self.llm_service._generate_text(prompt, max_length=512)
            if not response:
                return None
            
            # 提取JSON部分
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if not json_match:
                return None
            
            result_data = json.loads(json_match.group())
            
            # 验证和清理结果
            gene = result_data.get('gene')
            disease = result_data.get('disease') 
            tissue = result_data.get('tissue')
            confidence = float(result_data.get('confidence', 0.0))
            
            # 组织名称标准化
            if tissue:
                tissue = self._normalize_tissue(tissue)
            
            # 构建结果
            parsed = ParsedParameters(
                gene=gene if gene and gene.lower() != 'null' else None,
                disease=disease if disease and disease.lower() != 'null' else None,
                tissue=tissue if tissue and tissue.lower() != 'null' else None,
                confidence=confidence,
                raw_input=user_input,
                missing_params=result_data.get('missing_params', []),
                suggestions=[]
            )
            
            return parsed
            
        except Exception as e:
            logger.error(f"LLM解析JSON失败: {e}")
            return None
    
    def _parse_with_rules(self, user_input: str, language: str) -> ParsedParameters:
        """使用规则解析用户输入（备用方案）"""
        
        gene = None
        disease = None
        tissue = None
        confidence = 0.5  # 规则解析的基础置信度
        
        # 提取基因
        for pattern in self.gene_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                potential_gene = match.group(1).upper()
                # 验证基因是否有效
                if self.input_validator:
                    is_valid, error_msg = self.input_validator.validate_gene(potential_gene, language)
                    if is_valid:
                        gene = potential_gene
                        confidence += 0.2
                        break
                    else:
                        # 即使验证失败，如果是合理的基因符号格式，也接受
                        if self._is_reasonable_gene_symbol(potential_gene):
                            gene = potential_gene
                            confidence += 0.1
                            break
                else:
                    # 没有验证器时，检查是否是合理的基因符号格式
                    if self._is_reasonable_gene_symbol(potential_gene):
                        gene = potential_gene
                        confidence += 0.1
                        break
        
        # 提取组织
        for pattern in self.tissue_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                tissue_raw = match.group(1).lower()
                tissue = self.tissue_mapping.get(tissue_raw)
                if tissue:
                    confidence += 0.1
                    break
        
        # 提取疾病（使用疾病映射器和规则）
        if self.input_validator and self.input_validator.disease_mapper:
            # 尝试从整个输入中提取疾病
            validation_result = self.input_validator.disease_mapper.validate_input(user_input)
            if validation_result.get('is_valid'):
                disease = validation_result.get('matched_result')
                confidence += 0.2
            else:
                # 如果映射器没找到，尝试规则提取
                disease = self._extract_disease_with_rules(user_input, language)
                if disease:
                    confidence += 0.1
        else:
            # 没有疾病映射器时，使用规则提取
            disease = self._extract_disease_with_rules(user_input, language)
            if disease:
                confidence += 0.1
        
        # 确定缺失参数
        missing_params = []
        if not gene:
            missing_params.append('gene' if language == 'en' else '基因')
        if not disease:
            missing_params.append('disease' if language == 'en' else '疾病')
        
        return ParsedParameters(
            gene=gene,
            disease=disease,
            tissue=tissue or 'Whole_Blood',  # 默认使用全血
            confidence=min(confidence, 1.0),
            raw_input=user_input,
            missing_params=missing_params,
            suggestions=[]
        )
    
    def _normalize_tissue(self, tissue: str) -> str:
        """标准化组织名称"""
        tissue_lower = tissue.lower().strip()
        return self.tissue_mapping.get(tissue_lower, tissue)

    def _is_reasonable_gene_symbol(self, gene_symbol: str) -> bool:
        """检查是否是合理的基因符号格式"""
        if not gene_symbol:
            return False

        gene_clean = gene_symbol.upper().strip()

        # 基因符号的基本格式检查
        # 1. 长度在2-15个字符之间
        if len(gene_clean) < 2 or len(gene_clean) > 15:
            return False

        # 2. 只包含字母、数字、连字符、下划线
        if not re.match(r'^[A-Z0-9_-]+$', gene_clean):
            return False

        # 3. 必须以字母开头
        if not gene_clean[0].isalpha():
            return False

        # 4. 排除明显不是基因的词汇
        excluded_words = {
            'GENE', 'GENES', 'PROTEIN', 'DISEASE', 'TRAIT', 'STUDY', 'DATA',
            'ANALYSIS', 'RESULT', 'METHOD', 'SAMPLE', 'CONTROL', 'CASE'
        }
        if gene_clean in excluded_words:
            return False

        return True

    def _extract_disease_with_rules(self, user_input: str, language: str) -> Optional[str]:
        """使用规则提取疾病名称"""

        # 中文疾病模式
        chinese_disease_patterns = [
            r'(冠心病|心脏病|心血管疾病)',
            r'(糖尿病|2型糖尿病|1型糖尿病)',
            r'(阿尔茨海默病|老年痴呆|痴呆症)',
            r'(高血压|高血压病)',
            r'(中风|脑卒中|脑血管病)',
            r'(癌症|肿瘤|恶性肿瘤)',
            r'(乳腺癌|前列腺癌|肺癌|肝癌)',
            r'(类风湿关节炎|关节炎)',
            r'(肥胖|肥胖症)',
            r'(抑郁症|抑郁|精神疾病)',
            r'(高胆固醇血症|血脂异常)',
            r'(骨质疏松|骨质疏松症)',
            r'([^，。！？\s]{2,8}病)',  # 通用疾病模式：XX病
            r'([^，。！？\s]{2,8}症)',  # 通用症状模式：XX症
        ]

        # 英文疾病模式
        english_disease_patterns = [
            r'(coronary artery disease|coronary heart disease|heart disease)',
            r'(diabetes|type 2 diabetes|type 1 diabetes)',
            r'(alzheimer\'?s disease|dementia)',
            r'(hypertension|high blood pressure)',
            r'(stroke|cerebrovascular disease)',
            r'(cancer|tumor|malignancy)',
            r'(breast cancer|prostate cancer|lung cancer|liver cancer)',
            r'(rheumatoid arthritis|arthritis)',
            r'(obesity)',
            r'(depression|depressive disorder)',
            r'(hypercholesterolemia|dyslipidemia)',
            r'(osteoporosis)',
            r'([a-z\s]{3,30} disease)',  # 通用疾病模式：XX disease
            r'([a-z\s]{3,30} syndrome)',  # 通用综合征模式：XX syndrome
        ]

        patterns = chinese_disease_patterns if language == "zh" else english_disease_patterns

        for pattern in patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                disease_candidate = match.group(1).strip()
                # 简单验证：不能太短或包含明显的非疾病词汇
                if len(disease_candidate) >= 2 and not any(word in disease_candidate.lower() for word in ['基因', 'gene', '分析', 'analysis']):
                    return disease_candidate

        return None
    
    def generate_clarification_prompt(self, parsed: ParsedParameters, language: str = "zh") -> str:
        """生成澄清提示"""
        if language == "zh":
            if not parsed.missing_params:
                return ""
            
            missing_str = "、".join(parsed.missing_params)
            prompt = f"请补充以下信息：{missing_str}"
            
            if not parsed.gene:
                prompt += "\n\n💡 基因示例：PCSK9、APOE、LDLR、LPA等"
            if not parsed.disease:
                prompt += "\n\n💡 疾病示例：冠心病、糖尿病、阿尔茨海默病、高血压等"
            if not parsed.tissue:
                prompt += "\n\n💡 未指定组织时将默认使用全血进行分析"
                
            return prompt
        else:
            if not parsed.missing_params:
                return ""
            
            missing_str = ", ".join(parsed.missing_params)
            prompt = f"Please provide the following information: {missing_str}"
            
            if not parsed.gene:
                prompt += "\n\n💡 Gene examples: PCSK9, APOE, LDLR, LPA, etc."
            if not parsed.disease:
                prompt += "\n\n💡 Disease examples: coronary heart disease, diabetes, Alzheimer's disease, hypertension, etc."
            if not parsed.tissue:
                prompt += "\n\n💡 When tissue is not specified, whole blood will be used by default"
                
            return prompt
