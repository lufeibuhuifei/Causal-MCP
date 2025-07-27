#!/usr/bin/env python3
"""
基于ieugwaspy的疾病映射系统
使用官方推荐的标准方法解决疾病映射问题
"""

import json
import os
import time
import pickle
import re
import sys
import pandas as pd
from typing import List, Dict, Tuple, Optional
from difflib import SequenceMatcher
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# JWT manager integration
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from jwt_manager import jwt_manager
    JWT_MANAGER_AVAILABLE = True
    logger.info("JWT manager loaded")
except ImportError:
    JWT_MANAGER_AVAILABLE = False
    logger.warning("JWT manager not available, using local configuration")

class DiseaseMapper:
    """
    基于ieugwaspy的疾病映射系统

    特性:
    1. 使用ieugwaspy官方包获取GWAS数据
    2. 使用pandas进行标准数据处理
    3. 使用str.contains()进行科学匹配
    4. 完全解决P值相同问题
    5. 向后兼容原有接口
    """

    def __init__(self, validation_data_path: str = None):
        """初始化疾病映射器"""
        # 保持原有的validation_data.pkl支持（向后兼容）
        if validation_data_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.validation_data_path = os.path.join(current_dir, "validation_data.pkl")
        else:
            self.validation_data_path = validation_data_path

        # 原有属性（向后兼容）
        self.gwas_traits = set()
        self.disease_to_studies = {}
        self.study_to_disease = {}

        # ieugwaspy相关属性
        self.ieugwaspy_available = False
        self.gwas_dataframe = None
        self.last_update_time = 0
        self.cache_expiry_hours = 24

        # 缓存文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.cache_dir = os.path.join(current_dir, 'gwas_cache')
        self.cache_file = os.path.join(self.cache_dir, 'gwas_studies_cache.pkl')
        self.cache_meta_file = os.path.join(self.cache_dir, 'cache_metadata.json')

        # 确保缓存目录存在
        os.makedirs(self.cache_dir, exist_ok=True)

        # 初始化数据
        self._load_gwas_data()  # 加载原有validation_data.pkl（向后兼容）
        self._setup_ieugwaspy()  # 设置ieugwaspy

    def _setup_ieugwaspy(self):
        """设置ieugwaspy认证和数据获取"""
        try:
            import ieugwaspy

            # 获取JWT令牌
            jwt_token = self._load_jwt_token()
            if jwt_token:
                # 设置JWT到ieugwaspy配置
                ieugwaspy.config.env["jwt"] = jwt_token
                ieugwaspy.config._save_env()

                # 获取GWAS数据
                self._fetch_gwas_data_with_ieugwaspy()
                self.ieugwaspy_available = True
                logger.info("✅ ieugwaspy设置成功")
            else:
                logger.warning("未找到JWT令牌，ieugwaspy功能不可用")

        except ImportError:
            logger.warning("ieugwaspy包未安装，使用传统方法")
        except Exception as e:
            logger.error(f"设置ieugwaspy失败: {e}")

    def _load_jwt_token(self) -> Optional[str]:
        """加载JWT令牌"""
        # 1. 优先使用集中JWT管理器
        if JWT_MANAGER_AVAILABLE:
            try:
                token = jwt_manager.get_jwt_token()
                if token:
                    logger.info("使用集中JWT管理器中的令牌")
                    return token
            except Exception as e:
                logger.debug(f"集中JWT管理器获取令牌失败: {e}")

        # 2. 从环境变量获取（备用）
        jwt_token = os.environ.get('OPENGWAS_JWT')
        if jwt_token:
            logger.info("使用环境变量中的JWT令牌")
            return jwt_token

        # 3. 从配置文件获取
        config_paths = ['opengwas_config.json']

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        jwt_token = config.get('jwt_token')
                        if jwt_token:
                            logger.info("使用配置文件中的JWT令牌")
                            return jwt_token
                except Exception as e:
                    logger.error(f"读取配置文件失败: {e}")

        logger.warning("未找到有效的JWT令牌配置")
        return None

    def _load_gwas_cache(self) -> bool:
        """加载缓存的GWAS数据"""
        try:
            # 检查缓存文件是否存在
            if not os.path.exists(self.cache_file) or not os.path.exists(self.cache_meta_file):
                logger.info("缓存文件不存在，需要重新获取数据")
                return False

            # 检查缓存元数据
            with open(self.cache_meta_file, 'r', encoding='utf-8') as f:
                cache_meta = json.load(f)

            cache_time = cache_meta.get('timestamp', 0)
            current_time = time.time()

            # 检查缓存是否过期
            if (current_time - cache_time) > (self.cache_expiry_hours * 3600):
                logger.info(f"缓存已过期 ({self.cache_expiry_hours}小时)，需要重新获取数据")
                return False

            # 加载缓存的DataFrame
            logger.info("正在加载缓存的GWAS数据...")
            self.gwas_dataframe = pd.read_pickle(self.cache_file)
            self.last_update_time = cache_time

            logger.info(f"✅ 成功从缓存加载 {len(self.gwas_dataframe)} 个GWAS研究数据")
            logger.info(f"缓存时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_time))}")

            return True

        except Exception as e:
            logger.error(f"加载缓存失败: {e}")
            return False

    def _save_gwas_cache(self):
        """保存GWAS数据到缓存"""
        try:
            if self.gwas_dataframe is None:
                return

            # 保存DataFrame到pickle文件
            self.gwas_dataframe.to_pickle(self.cache_file)

            # 保存元数据
            cache_meta = {
                'timestamp': self.last_update_time,
                'record_count': len(self.gwas_dataframe),
                'columns': list(self.gwas_dataframe.columns),
                'cache_expiry_hours': self.cache_expiry_hours,
                'created_by': 'ieugwaspy_integration'
            }

            with open(self.cache_meta_file, 'w', encoding='utf-8') as f:
                json.dump(cache_meta, f, indent=2, ensure_ascii=False)

            logger.info(f"✅ 缓存已保存: {len(self.gwas_dataframe)} 个研究")
            logger.info(f"缓存位置: {self.cache_file}")

        except Exception as e:
            logger.error(f"保存缓存失败: {e}")

    def _fetch_gwas_data_with_ieugwaspy(self):
        """使用ieugwaspy获取GWAS数据（带缓存）"""
        try:
            # 首先尝试加载缓存
            if self._load_gwas_cache():
                return

            # 缓存不可用，从API获取
            import ieugwaspy

            logger.info("正在从ieugwaspy API获取GWAS数据...")
            logger.info("⚠️ 这可能需要几分钟时间，请耐心等待...")

            # 获取所有GWAS研究元数据
            all_gwas_metadata = ieugwaspy.gwasinfo()

            if isinstance(all_gwas_metadata, dict):
                # 转换为pandas DataFrame格式
                gwas_list = []
                for study_id, study_info in all_gwas_metadata.items():
                    if isinstance(study_info, dict):
                        study_info_copy = study_info.copy()
                        study_info_copy['id'] = study_id
                        gwas_list.append(study_info_copy)

                # 创建DataFrame
                self.gwas_dataframe = pd.DataFrame(gwas_list)

                # 确保trait列为字符串类型
                if 'trait' in self.gwas_dataframe.columns:
                    self.gwas_dataframe['trait'] = self.gwas_dataframe['trait'].astype(str)

                self.last_update_time = time.time()
                logger.info(f"✅ 成功获取 {len(gwas_list)} 个GWAS研究数据")

                # 保存到缓存
                self._save_gwas_cache()

            else:
                logger.error(f"ieugwaspy返回意外数据格式: {type(all_gwas_metadata)}")

        except Exception as e:
            logger.error(f"使用ieugwaspy获取数据失败: {e}")

    def _load_gwas_data(self):
        """加载GWAS验证数据"""
        try:
            with open(self.validation_data_path, 'rb') as f:
                data = pickle.load(f)
            
            self.gwas_traits = data.get('gwas_traits', set())
            logger.info(f"加载了 {len(self.gwas_traits)} 个GWAS数据条目")
            
            # 分离疾病名称和研究ID
            self._build_mappings()
            
        except Exception as e:
            logger.error(f"加载GWAS数据失败: {e}")
            self.gwas_traits = set()
    
    def _build_mappings(self):
        """构建疾病名称和研究ID的双向映射"""
        disease_names = []
        study_ids = []
        
        for item in self.gwas_traits:
            if self._is_study_id(item):
                study_ids.append(item)
            elif not self._is_gene_id(item):
                disease_names.append(item)
        
        logger.info(f"识别出 {len(disease_names)} 个疾病名称，{len(study_ids)} 个研究ID")
        
        # 构建疾病名称到研究ID的映射（基于相似性）
        self._build_disease_study_mapping(disease_names, study_ids)
    
    def _is_study_id(self, item: str) -> bool:
        """判断是否为研究ID"""
        return item.startswith(('ieu-', 'ukb-', 'ebi-', 'finn-', 'bbj-', 'eqtl-', 'prot-', 'met-', 'ubm-'))
    
    def _is_gene_id(self, item: str) -> bool:
        """判断是否为基因ID"""
        return item.startswith('ENSG')
    
    def _build_disease_study_mapping(self, disease_names: List[str], study_ids: List[str]):
        """构建疾病名称到研究ID的映射"""
        # 这里可以实现更复杂的映射逻辑
        # 目前先建立基本的存储结构
        self.disease_names = disease_names
        self.study_ids = study_ids
        
        # 为每个疾病名称找到最相关的研究ID（如果有的话）
        # 这里可以基于疾病名称的关键词匹配来建立映射
        pass
    
    def find_disease_or_study(self, query: str) -> Tuple[Optional[str], str, List[str]]:
        """
        查找疾病或研究ID

        Args:
            query: 用户输入的疾病名称或研究ID

        Returns:
            Tuple[匹配结果, 匹配类型, 推荐列表]
        """
        query_clean = query.strip()
        
        # 1. Exact match
        if query_clean in self.gwas_traits:
            match_type = "study_id" if self._is_study_id(query_clean) else "disease_name"
            return query_clean, f"exact_match_{match_type}", []

        # 2. Case insensitive match
        for item in self.gwas_traits:
            if item.lower() == query_clean.lower():
                match_type = "study_id" if self._is_study_id(item) else "disease_name"
                return item, f"case_insensitive_{match_type}", []
        
        # 3. 模糊匹配和推荐
        recommendations = self._find_similar_diseases(query_clean)
        
        if recommendations:
            # 如果有高相似度匹配（>0.8），返回最佳匹配
            best_match = recommendations[0]
            if best_match[1] > 0.8:
                match_type = "study_id" if self._is_study_id(best_match[0]) else "disease_name"
                return best_match[0], f"fuzzy_match_{match_type}", [r[0] for r in recommendations[1:6]]
        
        # 4. No match found, return recommendations
        return None, "no_match", [r[0] for r in recommendations[:10]]
    
    def _find_similar_diseases(self, query: str) -> List[Tuple[str, float]]:
        """查找相似的疾病名称"""
        query_lower = query.lower()
        similarities = []

        # 在所有GWAS数据中搜索（包括疾病名称和研究ID）
        for item in self.gwas_traits:
            if not isinstance(item, str):
                continue

            item_lower = item.lower()

            # 1. 包含匹配（权重高）
            if query_lower in item_lower:
                similarity = 0.9 + (len(query_lower) / len(item_lower)) * 0.1
                similarities.append((item, similarity))
            elif item_lower in query_lower:
                similarity = 0.8 + (len(item_lower) / len(query_lower)) * 0.1
                similarities.append((item, similarity))
            else:
                # 2. 序列相似度匹配（只对较短的字符串进行，避免性能问题）
                if len(item) < 100 and len(query) < 100:
                    similarity = SequenceMatcher(None, query_lower, item_lower).ratio()
                    if similarity > 0.5:  # 提高相似度阈值
                        similarities.append((item, similarity))

        # 按相似度排序，限制结果数量
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:50]  # 限制返回数量
    
    def get_study_id_for_disease(self, disease_name: str) -> Optional[str]:
        """
        为疾病名称获取对应的研究ID

        使用ieugwaspy的官方推荐方法：
        1. 优先使用ieugwaspy动态查询
        2. 降级到传统方法（向后兼容）

        Args:
            disease_name: 疾病名称或研究ID

        Returns:
            最佳匹配的研究ID，如果没有找到返回None
        """
        # 1. 如果输入本身就是研究ID，直接返回
        if self._is_study_id(disease_name):
            return disease_name

        # 2. 优先使用ieugwaspy方法
        if self.ieugwaspy_available and self.gwas_dataframe is not None:
            ieugwaspy_result = self._get_study_id_with_ieugwaspy(disease_name)
            if ieugwaspy_result:
                return ieugwaspy_result

        # 3. 降级到传统方法（向后兼容）
        return self._get_study_id_traditional(disease_name)

    def _get_study_id_with_ieugwaspy(self, disease_name: str) -> Optional[str]:
        """
        使用ieugwaspy方法获取研究ID
        这是按照官方推荐的标准方法实现的
        """
        try:
            # 检查数据是否需要更新
            if self.gwas_dataframe is None:
                logger.info("GWAS数据未加载，正在获取...")
                self._fetch_gwas_data_with_ieugwaspy()
            else:
                current_time = time.time()
                if (current_time - self.last_update_time) > (self.cache_expiry_hours * 3600):
                    logger.info("GWAS数据缓存过期，重新获取...")
                    self._fetch_gwas_data_with_ieugwaspy()

            if self.gwas_dataframe is None or self.gwas_dataframe.empty:
                logger.warning("GWAS DataFrame为空")
                return None

            # 使用pandas进行筛选（官方推荐方法）
            disease_keyword = disease_name.lower().strip()

            # 使用str.contains进行不区分大小写匹配
            matching_df = self.gwas_dataframe[
                self.gwas_dataframe['trait'].str.contains(disease_keyword, case=False, na=False)
            ].copy()

            if not matching_df.empty:
                # 按样本量排序，选择最大样本量的研究
                if 'sample_size' in matching_df.columns:
                    matching_df = matching_df.sort_values('sample_size', ascending=False, na_position='last')
                elif 'n_total' in matching_df.columns:
                    matching_df = matching_df.sort_values('n_total', ascending=False, na_position='last')

                # 返回最佳匹配的研究ID
                best_match = matching_df.iloc[0]
                best_id = best_match['id']

                logger.info(f"🎯 ieugwaspy匹配: '{disease_name}' → '{best_id}' ({best_match.get('trait', 'N/A')})")
                return best_id

            logger.info(f"ieugwaspy未找到匹配: '{disease_name}'")
            return None

        except Exception as e:
            logger.error(f"ieugwaspy查询失败: {e}")
            return None

    def _get_study_id_traditional(self, disease_name: str) -> Optional[str]:
        """
        传统方法获取研究ID（向后兼容）
        """
        # 如果疾病名称在GWAS数据库中存在，直接使用疾病名称
        if disease_name in self.gwas_traits:
            return disease_name

        # 检查大小写不敏感的疾病名称匹配
        disease_lower = disease_name.lower()
        for trait in self.gwas_traits:
            if isinstance(trait, str) and not self._is_study_id(trait):
                if trait.lower() == disease_lower:
                    return trait

        # 使用保守的同义词映射
        CONSERVATIVE_SYNONYMS = {
            "BMI": "Body mass index",
            "CHD": "Coronary heart disease",
            "T2D": "Type 2 diabetes",
            "T2DM": "Type 2 diabetes",
            "RA": "Rheumatoid arthritis"
        }

        if disease_name in CONSERVATIVE_SYNONYMS:
            synonym = CONSERVATIVE_SYNONYMS[disease_name]
            if synonym in self.gwas_traits:
                return synonym

        return None

    def get_gwas_ids_by_disease_name(self, disease_keyword: str, max_results: int = 10) -> Tuple[List[str], Optional[pd.DataFrame]]:
        """
        根据疾病关键词获取匹配的GWAS研究ID列表
        这是按照官方推荐的标准方法实现的

        Args:
            disease_keyword: 疾病名称或关键词
            max_results: 最大返回结果数

        Returns:
            Tuple[匹配的研究ID列表, 匹配的研究详细信息DataFrame]
        """
        if not self.ieugwaspy_available or self.gwas_dataframe is None:
            logger.warning("ieugwaspy不可用，返回空结果")
            return [], None

        try:
            # 检查数据是否需要更新
            if self.gwas_dataframe is None:
                self._fetch_gwas_data_with_ieugwaspy()
            else:
                current_time = time.time()
                if (current_time - self.last_update_time) > (self.cache_expiry_hours * 3600):
                    self._fetch_gwas_data_with_ieugwaspy()

            if self.gwas_dataframe.empty:
                return [], None

            # 使用pandas进行筛选（官方推荐方法）
            disease_keyword_lower = disease_keyword.lower().strip()

            # 使用str.contains进行不区分大小写匹配
            matching_df = self.gwas_dataframe[
                self.gwas_dataframe['trait'].str.contains(disease_keyword_lower, case=False, na=False)
            ].copy()

            if matching_df.empty:
                logger.info(f"未找到与 '{disease_keyword}' 相关的GWAS研究")
                return [], None

            # 按样本量排序
            if 'sample_size' in matching_df.columns:
                matching_df = matching_df.sort_values('sample_size', ascending=False, na_position='last')
            elif 'n_total' in matching_df.columns:
                matching_df = matching_df.sort_values('n_total', ascending=False, na_position='last')

            # 限制结果数量
            matching_df = matching_df.head(max_results)

            # 提取研究ID列表
            matching_ids = matching_df['id'].tolist()

            logger.info(f"找到与 '{disease_keyword}' 相关的 {len(matching_ids)} 个GWAS研究")

            return matching_ids, matching_df

        except Exception as e:
            logger.error(f"查询GWAS研究失败: {e}")
            return [], None



    def validate_input(self, query: str) -> Dict:
        """
        验证用户输入并提供详细反馈
        
        Args:
            query: 用户输入
            
        Returns:
            验证结果字典
        """
        query_clean = query.strip()

        # 1. 如果是研究ID，直接验证
        if self._is_study_id(query_clean):
            return {
                "is_valid": True,
                "input": query,
                "matched_result": query_clean,
                "match_type": "study_id",
                "recommendations": [],
                "message": f"✅ 有效的研究ID: '{query_clean}'"
            }

        # 2. 优先使用ieugwaspy查找匹配
        if self.ieugwaspy_available:
            matching_ids, matching_df = self.get_gwas_ids_by_disease_name(query_clean, max_results=5)

            if matching_ids:
                # 生成推荐列表
                recommendations = []
                if matching_df is not None:
                    for _, row in matching_df.iterrows():
                        trait = row.get('trait', 'Unknown')
                        study_id = row.get('id', 'Unknown')
                        sample_size = row.get('sample_size', row.get('n_total', 'Unknown'))
                        recommendations.append(f"{trait} (ID: {study_id})")

                return {
                    "is_valid": True,
                    "input": query,
                    "matched_result": matching_ids[0],
                    "match_type": "ieugwaspy_match",
                    "recommendations": recommendations,
                    "message": f"✅ Match found via ieugwaspy: '{matching_ids[0]}'"
                }

        # 3. 降级到传统方法
        result, match_type, traditional_recommendations = self.find_disease_or_study(query_clean)

        validation_result = {
            "is_valid": result is not None,
            "input": query,
            "matched_result": result,
            "match_type": match_type,
            "recommendations": traditional_recommendations,
            "message": ""
        }

        if result:
            if "exact_match" in match_type:
                validation_result["message"] = f"✅ Exact match found: '{result}'"
            elif "case_insensitive" in match_type:
                validation_result["message"] = f"✅ Match found (case insensitive): '{result}'"
            elif "fuzzy_match" in match_type:
                validation_result["message"] = f"✅ Similar match found: '{result}'"
        else:
            validation_result["message"] = f"❌ No match found: '{query}'"
            if traditional_recommendations:
                validation_result["message"] += f"\n💡 Suggestions: {', '.join(traditional_recommendations[:3])}"

        return validation_result

    # ===== 缓存管理方法 =====

    def get_cache_info(self) -> Dict:
        """获取缓存信息"""
        cache_info = {
            'cache_exists': os.path.exists(self.cache_file),
            'cache_file': self.cache_file,
            'cache_size_mb': 0,
            'record_count': 0,
            'last_update': None,
            'expires_in_hours': 0,
            'is_expired': True
        }

        try:
            if os.path.exists(self.cache_file):
                # 获取文件大小
                cache_info['cache_size_mb'] = round(os.path.getsize(self.cache_file) / (1024 * 1024), 2)

            if os.path.exists(self.cache_meta_file):
                with open(self.cache_meta_file, 'r', encoding='utf-8') as f:
                    cache_meta = json.load(f)

                cache_time = cache_meta.get('timestamp', 0)
                current_time = time.time()

                cache_info['record_count'] = cache_meta.get('record_count', 0)
                cache_info['last_update'] = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(cache_time))

                time_diff = current_time - cache_time
                cache_info['expires_in_hours'] = round(self.cache_expiry_hours - (time_diff / 3600), 1)
                cache_info['is_expired'] = time_diff > (self.cache_expiry_hours * 3600)

        except Exception as e:
            logger.error(f"获取缓存信息失败: {e}")

        return cache_info

    def clear_cache(self):
        """清除缓存"""
        try:
            if os.path.exists(self.cache_file):
                os.remove(self.cache_file)
                logger.info("✅ 缓存文件已删除")

            if os.path.exists(self.cache_meta_file):
                os.remove(self.cache_meta_file)
                logger.info("✅ 缓存元数据已删除")

            self.gwas_dataframe = None
            self.last_update_time = 0

        except Exception as e:
            logger.error(f"清除缓存失败: {e}")

    def force_refresh_cache(self):
        """强制刷新缓存"""
        logger.info("强制刷新GWAS数据缓存...")
        self.clear_cache()
        self._fetch_gwas_data_with_ieugwaspy()

    # ===== 辅助方法 =====
