# src/mcp_server_gwas/main.py
from mcp.server.fastmcp import FastMCP
import httpx
import logging

from .models import GWASToolInput, GWASToolOutput, HarmonizedDataPoint
from .harmonize import harmonize_datasets

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 1. 初始化 FastMCP 服务器
mcp = FastMCP(
    name="mcp-server-gwas",
    description="A server providing tools to fetch GWAS outcomes and perform data harmonization."
)

# 导入疾病映射器和JWT管理器
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'client-app'))

# 添加项目根目录到路径以使用集中JWT管理器
project_root_path = os.path.join(os.path.dirname(__file__), '..', '..', '..')
sys.path.append(project_root_path)

try:
    from disease_mapper import DiseaseMapper
    DISEASE_MAPPER = DiseaseMapper()
    DISEASE_MAPPER_AVAILABLE = True
    logger.info("✅ 疾病映射器已加载，支持ieugwaspy动态查询")
except ImportError as e:
    logger.error(f"无法导入疾病映射器: {e}")
    DISEASE_MAPPER = None
    DISEASE_MAPPER_AVAILABLE = False

# 尝试导入集中JWT管理器
try:
    from jwt_manager import jwt_manager
    JWT_MANAGER_AVAILABLE = True
    logger.info("✅ 集中JWT管理器已加载")
except ImportError as e:
    logger.warning(f"集中JWT管理器不可用，使用本地配置: {e}")
    JWT_MANAGER_AVAILABLE = False

# 备用的固定映射（向后兼容）
GWAS_LOOKUP = {
    "ieu-a-7": "Coronary Artery Disease",
    "ieu-a-300": "Body Mass Index",
    "ieu-a-835": "Type 2 Diabetes",
    "ieu-a-89": "LDL Cholesterol",
    # ... 更多GWAS研究的映射
}

# Real GWAS data integration using OpenGWAS API
import httpx
import asyncio
import os
from typing import Dict, List, Optional
import json

class RealGWASDataClient:
    """真实GWAS数据客户端，使用OpenGWAS API"""

    def __init__(self, jwt_token: Optional[str] = None):
        # 使用OpenGWAS API - 专为MR分析设计的高质量数据源
        self.base_url = "https://api.opengwas.io/api"
        self.timeout = 60.0  # 优化超时时间到1分钟

        # JWT令牌获取优先级：
        # 1. 构造函数参数（最高优先级，用于测试和直接调用）
        # 2. 环境变量 OPENGWAS_JWT
        # 3. 配置文件 ~/.opengwas/config.json
        # 4. 当前目录下的 opengwas_config.json
        self.jwt_token = self._get_jwt_token(jwt_token)

        if not self.jwt_token:
            logging.error("OpenGWAS JWT token not available.")
            logging.error("请通过以下方式之一提供有效的OpenGWAS JWT令牌:")
            logging.error("1. 设置环境变量: export OPENGWAS_JWT='your_token_here'")
            logging.error("2. 创建配置文件: opengwas_config.json")
            logging.error("配置文件格式: {\"jwt_token\": \"your_token_here\"}")
            raise ValueError("OpenGWAS JWT token is required for data access")
        else:
            # 隐藏令牌的敏感部分用于日志记录
            token_preview = f"{self.jwt_token[:20]}...{self.jwt_token[-10:]}" if len(self.jwt_token) > 30 else "***"
            logging.info(f"✅ OpenGWAS JWT token configured ({token_preview}). Using OpenGWAS API data.")

        # 数据集映射（用于验证和描述）
        self.dataset_mapping = {
            "ieu-a-7": {"trait": "coronary artery disease", "efo_id": "EFO_0000378"},
            "ieu-a-89": {"trait": "LDL cholesterol", "efo_id": "EFO_0004611"},
            "ieu-a-835": {"trait": "type 2 diabetes", "efo_id": "EFO_0001360"},
            "ieu-a-300": {"trait": "body mass index", "efo_id": "EFO_0004340"},
        }

    def _get_jwt_token(self, provided_token: Optional[str] = None) -> Optional[str]:
        """
        获取JWT令牌，按优先级顺序尝试多种来源

        Args:
            provided_token: 直接提供的令牌（最高优先级）

        Returns:
            JWT令牌字符串，如果未找到则返回None
        """
        # 1. 直接提供的令牌（最高优先级）
        if provided_token:
            logging.info("使用直接提供的JWT令牌")
            return provided_token

        # 2. 集中JWT管理器（新增）
        if JWT_MANAGER_AVAILABLE:
            try:
                token = jwt_manager.get_jwt_token()
                if token:
                    logging.info("使用集中JWT管理器中的令牌")
                    return token
            except Exception as e:
                logging.debug(f"集中JWT管理器获取令牌失败: {e}")

        # 3. 环境变量
        env_token = os.environ.get('OPENGWAS_JWT')
        if env_token:
            logging.info("使用环境变量中的JWT令牌")
            return env_token

        # 4. 配置文件支持
        try:
            import json
            from pathlib import Path

            config_paths = [
                Path('.opengwas') / 'config.json',
                Path('opengwas_config.json')
            ]

            for config_path in config_paths:
                if config_path.exists():
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                        token = config.get('jwt_token')
                        if token:
                            logging.info("使用配置文件中的JWT令牌")
                            return token
        except Exception as e:
            logging.debug(f"读取配置文件失败: {e}")

        logging.warning("未找到有效的JWT令牌配置")
        return None

    def _get_headers(self):
        """获取API请求头"""
        return {
            "Authorization": f"Bearer {self.jwt_token}",
            "Content-Type": "application/json"
        }

    async def get_associations(self, outcome_id: str, snp_list: List[str]) -> Dict[str, Dict]:
        """
        从OpenGWAS API获取SNP-outcome关联数据（严禁使用模拟数据）

        Args:
            outcome_id: GWAS研究ID (如 ieu-a-7)
            snp_list: SNP列表 (如 ['rs123', 'rs456'])

        Returns:
            SNP关联数据字典，只包含真实的OpenGWAS数据
        """
        logging.info(f"🔍 从OpenGWAS API获取真实GWAS数据: {outcome_id}")
        logging.info(f"查询 {len(snp_list)} 个SNP的关联数据")

        # 只使用OpenGWAS API，严禁降级或模拟数据
        return await self._get_opengwas_associations(outcome_id, snp_list)

    def _convert_snp_formats(self, snp_list: List[str]) -> List[str]:
        """转换SNP格式以提高OpenGWAS兼容性"""
        converted_snps = []

        for snp in snp_list:
            # 如果是位置格式 (如 1:55055436)，尝试多种格式
            if ":" in snp and not snp.startswith("rs"):
                parts = snp.split(":")
                if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                    chr_num, pos = parts

                    # 尝试多种可能的格式
                    variants = [
                        snp,                           # 原始格式: 1:55055436
                        f"chr{chr_num}:{pos}",        # chr前缀: chr1:55055436
                        f"{chr_num}_{pos}",           # 下划线: 1_55055436
                        f"chr{chr_num}_{pos}",        # chr下划线: chr1_55055436
                    ]

                    # 只使用第一个格式进行查询，如果失败再尝试其他格式
                    converted_snps.append(variants[0])
                else:
                    converted_snps.append(snp)
            else:
                # rs ID或其他格式直接使用
                converted_snps.append(snp)

        return converted_snps

    async def _get_opengwas_associations(self, outcome_id: str, snp_list: List[str]) -> Dict[str, Dict]:
        """从OpenGWAS API获取关联数据（带重试机制）"""
        # 转换SNP格式以提高兼容性
        converted_snps = self._convert_snp_formats(snp_list)

        # 优化重试机制
        max_retries = 2  # 减少重试次数
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    logging.info(f"🔄 重试第 {attempt} 次...")
                    await asyncio.sleep(1.5 ** attempt)  # 更温和的退避策略

                # 使用更细粒度的超时配置
                timeout_config = httpx.Timeout(
                    connect=20.0,  # 连接超时
                    read=150.0,    # 读取超时 - 增加到2.5分钟以处理复杂查询
                    write=20.0,    # 写入超时
                    pool=15.0      # 连接池超时
                )

                async with httpx.AsyncClient(timeout=timeout_config) as client:
                    # 构建查询参数
                    payload = {
                        "variant": converted_snps,
                        "id": [outcome_id]
                    }

                    # 调用OpenGWAS API
                    response = await client.post(
                        f"{self.base_url}/associations",
                        json=payload,
                        headers=self._get_headers()
                    )

                    if response.status_code == 200:
                        data = response.json()
                        data_count = len(data) if isinstance(data, list) else 0
                        logging.info(f"✅ OpenGWAS API成功返回 {data_count} 个真实关联数据")

                        if data_count == 0:
                            logging.warning(f"OpenGWAS数据库中不存在outcome {outcome_id} 与这些SNP的关联数据")
                            logging.warning("这是数据库内容限制，不是技术错误")

                        return self._parse_opengwas_response(data)
                    elif response.status_code == 401:
                        logging.error("❌ OpenGWAS API认证失败 - 这是技术错误")
                        logging.error("JWT令牌无效或已过期，请检查令牌配置")
                        return {}
                    elif response.status_code == 404:
                        logging.warning(f"⚠️ OpenGWAS数据库中不存在outcome ID: {outcome_id}")
                        logging.warning("这是数据库内容限制，不是技术错误")
                        return {}
                    elif response.status_code == 429:
                        if attempt < max_retries - 1:
                            logging.warning(f"⚠️ OpenGWAS API配额超限，将重试...")
                            continue
                        else:
                            logging.error("❌ OpenGWAS API配额超限 - 重试次数已用完")
                            return {}
                    else:
                        logging.error(f"❌ OpenGWAS API技术错误: HTTP {response.status_code}")
                        logging.error(f"响应内容: {response.text[:200] if hasattr(response, 'text') else 'N/A'}")
                        if attempt < max_retries - 1:
                            logging.info("将重试...")
                            continue
                        else:
                            logging.error("重试次数已用完")
                            return {}

            except httpx.TimeoutException as e:
                if attempt < max_retries - 1:
                    logging.warning(f"⚠️ OpenGWAS API连接超时，将重试... (尝试 {attempt + 1}/{max_retries})")
                    continue
                else:
                    logging.error(f"❌ OpenGWAS API连接超时: {e}")
                    logging.error("这是网络技术问题，不是数据不存在的问题")
                    return {}
            except httpx.ConnectError as e:
                if attempt < max_retries - 1:
                    logging.warning(f"⚠️ OpenGWAS API连接失败，将重试... (尝试 {attempt + 1}/{max_retries})")
                    continue
                else:
                    logging.error(f"❌ OpenGWAS API连接失败: {e}")
                    logging.error("这是网络技术问题，请检查网络连接")
                    return {}
            except Exception as e:
                if attempt < max_retries - 1:
                    logging.warning(f"⚠️ OpenGWAS API调用出错，将重试... (尝试 {attempt + 1}/{max_retries}): {e}")
                    continue
                else:
                    logging.error(f"❌ OpenGWAS API调用技术错误: {e}")
                    logging.error("这是系统技术问题，不是数据不存在的问题")
                    return {}

        # 如果所有重试都失败了
        logging.error("❌ 所有重试尝试都失败了")
        return {}

    # EBI GWAS Catalog降级方案已移除 - 严禁使用非OpenGWAS数据源
    # 所有数据必须来自OpenGWAS API以确保数据质量和一致性

    def _parse_opengwas_response(self, response_data: List[Dict]) -> Dict[str, Dict]:
        """解析OpenGWAS API响应数据"""
        parsed_data = {}

        for item in response_data:
            rsid = item.get('rsid')
            if rsid:
                try:
                    # 安全地转换数值，处理空字符串和None值
                    def safe_float(value, default=None):
                        if value is None or value == '' or value == 'NA':
                            return default
                        try:
                            return float(value)
                        except (ValueError, TypeError):
                            return default

                    # 专门处理P值的函数
                    def safe_pval(value, default=None):
                        if value is None or value == '' or value == 'NA':
                            return default
                        try:
                            result = float(value)
                            # 特殊处理P值为0的情况
                            if result == 0.0 or result < 1e-16:
                                return 1e-16  # 使用1e-16作为极小值下限，与MR分析保持一致
                            return result
                        except (ValueError, TypeError):
                            return default

                    def safe_int(value, default=None):
                        if value is None or value == '' or value == 'NA':
                            return default
                        try:
                            return int(float(value))  # 先转float再转int，处理科学计数法
                        except (ValueError, TypeError):
                            return default

                    parsed_data[rsid] = {
                        "effect_allele": item.get('ea', ''),
                        "other_allele": item.get('nea', ''),
                        "beta": safe_float(item.get('beta')),
                        "se": safe_float(item.get('se')),
                        "pval": safe_pval(item.get('p')),  # 使用专门的P值处理函数
                        "eaf": safe_float(item.get('eaf')),
                        "n": safe_int(item.get('n')),
                        "source": "OpenGWAS_API_Real"
                    }
                except Exception as e:
                    logging.warning(f"解析OpenGWAS数据项失败 {rsid}: {e}")
                    continue

        return parsed_data

    # EBI GWAS Catalog解析函数已移除 - 只使用OpenGWAS数据

    async def validate_outcome_id(self, outcome_id: str) -> bool:
        """验证GWAS研究ID是否在OpenGWAS数据库中有效"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/gwasinfo/{outcome_id}",
                    headers=self._get_headers()
                )

                if response.status_code == 200:
                    logging.info(f"✅ OpenGWAS验证成功: outcome ID {outcome_id} 存在")
                    return True
                elif response.status_code == 404:
                    logging.warning(f"⚠️ OpenGWAS数据库中不存在outcome ID: {outcome_id}")
                    return False
                else:
                    logging.error(f"❌ OpenGWAS验证技术错误: HTTP {response.status_code}")
                    return False
        except Exception as e:
            logging.error(f"❌ OpenGWAS验证调用失败: {e}")
            return False

# 初始化真实GWAS数据客户端
# 注意：JWT令牌将从环境变量或配置文件中自动获取
gwas_client = RealGWASDataClient()



@mcp.tool()
async def fetch_gwas_outcomes(params: GWASToolInput) -> GWASToolOutput:
    """
    Fetches GWAS outcome data for given SNPs and performs data harmonization.
    
    This tool takes a list of SNP instruments from eQTL analysis and fetches
    their effects on a specified outcome from GWAS databases. It then performs
    critical data harmonization to ensure effect estimates are aligned.
    """
    logging.info(f"Received request to fetch GWAS outcomes for {len(params.exposure_instruments)} SNPs")
    logging.info(f"Target outcome: {params.outcome_id}")
    
    # 使用新的疾病映射器验证和解析outcome_id
    outcome_id = params.outcome_id
    outcome_name = None

    if DISEASE_MAPPER_AVAILABLE:
        # 优先使用ieugwaspy方法
        try:
            # 如果输入是疾病名称，获取最佳研究ID
            if not DISEASE_MAPPER._is_study_id(outcome_id):
                logging.info(f"🔍 使用ieugwaspy查找疾病 '{outcome_id}' 的最佳研究ID")
                best_study_id = DISEASE_MAPPER.get_study_id_for_disease(outcome_id)
                if best_study_id:
                    outcome_id = best_study_id
                    logging.info(f"✅ 映射结果: '{params.outcome_id}' → '{outcome_id}'")
                else:
                    raise ValueError(f"未找到疾病 '{params.outcome_id}' 的匹配研究")

            # 获取研究的详细信息
            if DISEASE_MAPPER.gwas_dataframe is not None:
                matching_studies = DISEASE_MAPPER.gwas_dataframe[
                    DISEASE_MAPPER.gwas_dataframe['id'] == outcome_id
                ]
                if not matching_studies.empty:
                    outcome_name = matching_studies.iloc[0].get('trait', outcome_id)

        except Exception as e:
            logging.error(f"ieugwaspy查询失败: {e}")
            # 降级到传统方法
            if outcome_id not in GWAS_LOOKUP:
                raise ValueError(f"Outcome ID '{outcome_id}' is not supported")
            outcome_name = GWAS_LOOKUP[outcome_id]
    else:
        # 使用传统的固定映射
        if outcome_id not in GWAS_LOOKUP:
            raise ValueError(f"Outcome ID '{outcome_id}' is not supported. Available outcomes: {list(GWAS_LOOKUP.keys())}")
        outcome_name = GWAS_LOOKUP[outcome_id]

    logging.info(f"🎯 查询GWAS数据: {outcome_name} (ID: {outcome_id})")
    
    # 验证outcome_id并尝试获取真实数据
    is_valid = await gwas_client.validate_outcome_id(outcome_id)
    data_source = "unknown"

    if is_valid:
        # 提取SNP列表
        snp_list = [getattr(instrument, 'rsid', getattr(instrument, 'snp_id', '')) for instrument in params.exposure_instruments]
        logging.info(f"🔍 从OpenGWAS API获取 {len(snp_list)} 个SNP的真实GWAS数据")

        # 只获取真实OpenGWAS数据，严禁使用模拟数据
        outcome_data = await gwas_client.get_associations(outcome_id, snp_list)

        if outcome_data:
            data_source = "Real_OpenGWAS_Data"
            logging.info(f"✅ 成功从OpenGWAS获取 {len(outcome_data)} 个SNP的真实关联数据")
            # 记录数据来源验证
            for snp_id, data in list(outcome_data.items())[:3]:  # 记录前3个作为验证
                logging.info(f"SNP {snp_id}: beta={data.get('beta', 'N/A')}, p={data.get('pval', 'N/A')}, 来源=OpenGWAS")
        else:
            logging.warning("⚠️ OpenGWAS数据库中未找到相关的GWAS关联数据")
            logging.warning("这是数据库内容限制，不是技术错误")
            outcome_data = {}
            data_source = "No_Data_Available"
    else:
        logging.error(f"❌ Outcome ID '{params.outcome_id}' 在OpenGWAS中无效")
        logging.error("请检查outcome ID是否正确或该研究是否在OpenGWAS数据库中")
        outcome_data = {}
        data_source = "Invalid_Outcome_ID"

    logging.info(f"Retrieved GWAS data for {len(outcome_data)} SNPs from {data_source}")

    # 执行数据和谐化，传递研究ID信息
    harmonized_data, excluded_snps = harmonize_datasets(
        params.exposure_instruments,
        outcome_data,
        outcome_id  # 传递最终使用的研究ID
    )

    # 生成摘要，包含数据来源信息
    total_snps = len(params.exposure_instruments)
    harmonized_count = len(harmonized_data)
    excluded_count = len(excluded_snps)

    # 数据来源说明
    source_description = {
        "Real_OpenGWAS_Data": "authentic OpenGWAS API data",
        "No_Data_Available": "no real data available in OpenGWAS database",
        "Invalid_Outcome_ID": "invalid outcome ID in OpenGWAS database",
        "unknown": "data source unknown"
    }

    summary = f"Successfully harmonized {harmonized_count} out of {total_snps} SNPs for {outcome_name} using {source_description.get(data_source, data_source)}."
    if excluded_count > 0:
        summary += f" {excluded_count} SNPs were excluded due to harmonization issues."

    # 添加数据透明性说明
    if data_source == "Real_OpenGWAS_Data":
        summary += " ✅ This analysis uses only real GWAS data from OpenGWAS API - no simulated data."
    elif data_source == "No_Data_Available":
        summary += " ⚠️ No real GWAS data was available - no simulated data was used as substitute."
    elif data_source == "Invalid_Outcome_ID":
        summary += " ❌ The specified outcome ID is not supported in OpenGWAS database."

    logging.info(summary)

    return GWASToolOutput(
        harmonized_data=harmonized_data,
        summary=summary,
        excluded_snps=excluded_snps
    )

# 3. 配置服务器入口点
def run():
    mcp.run()

if __name__ == "__main__":
    run()
