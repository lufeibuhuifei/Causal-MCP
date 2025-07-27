# src/mcp_server_eqtl/main.py
from mcp.server.fastmcp import FastMCP
from typing import List
import httpx
import logging

from .models import EQTLToolInput, SNPInstrument

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. 初始化 FastMCP 服务器
# FastMCP会自动处理底层的通信协议 (stdio, http, etc.)
mcp = FastMCP(
    name="mcp-server-eqtl",
    description="A server providing tools to query eQTL data for Mendelian Randomization."
)



# Real eQTL data integration using GTEx and eQTLGen
import asyncio
from typing import Dict, List, Optional
import json

class RealEQTLDataClient:
    """真实eQTL数据客户端，连接GTEx和eQTLGen数据库"""

    def __init__(self):
        # 使用GTEx数据源
        self.gtex_url = "https://gtexportal.org/api/v2"
        self.timeout = 60.0  # 增加超时时间

        # 初始化SNP转换器
        try:
            import sys
            import os
            # 添加当前目录到路径
            current_dir = os.path.dirname(__file__)
            if current_dir not in sys.path:
                sys.path.insert(0, current_dir)

            from snp_converter import SNPConverter
            self.snp_converter = SNPConverter()
            logging.info("✅ SNP转换器初始化成功")
        except Exception as e:
            logging.error(f"❌ SNP转换器初始化失败: {e}")
            import traceback
            logging.error(f"详细错误: {traceback.format_exc()}")
            self.snp_converter = None

    async def get_eqtl_instruments(self, gene_symbol: str, tissue: str, significance_threshold: float = 5e-8) -> List[SNPInstrument]:
        """
        从真实eQTL数据库获取基因的工具变量

        Args:
            gene_symbol: 基因符号
            tissue: 组织类型
            significance_threshold: 显著性阈值

        Returns:
            SNP工具变量列表
        """
        logging.info(f"获取基因 {gene_symbol} 在组织 {tissue} 的真实eQTL数据")

        # 只使用GTEx真实数据，不使用任何备用数据源
        gtex_instruments = await self._get_gtex_data(gene_symbol, tissue, significance_threshold)

        if gtex_instruments:
            logging.info(f"从GTEx获取到 {len(gtex_instruments)} 个真实工具变量")
            # 按p值排序，返回最显著的工具变量
            gtex_instruments.sort(key=lambda x: x.p_value)
            return gtex_instruments[:10]
        else:
            logging.warning(f"GTEx中未找到基因 {gene_symbol} 的eQTL数据")
            logging.warning("这是数据库内容限制，不是技术错误")
            return []

    async def _get_gtex_data(self, gene_symbol: str, tissue: str, threshold: float) -> List[SNPInstrument]:
        """从GTEx获取真实的组织特异性eQTL数据"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                # 首先获取基因信息以获得gencodeId
                gene_response = await client.get(
                    f"{self.gtex_url}/reference/gene",
                    params={"geneId": gene_symbol, "format": "json"}
                )

                if gene_response.status_code == 200:
                    gene_data = gene_response.json()
                    logging.info(f"GTEx基因验证成功: {gene_symbol}")

                    if gene_data.get('data'):
                        gene_info = gene_data['data'][0]
                        gencode_id = gene_info.get('gencodeId', gene_symbol)
                        chromosome = gene_info.get('chromosome', 'chr1')

                        logging.info(f"使用gencodeId获取真实eQTL数据: {gencode_id}")

                        # 获取真实的eQTL数据 - 使用正确的GTEx API端点
                        # GTEx API v8的正确端点和参数
                        eqtl_params = {
                            "gencodeId": gencode_id,
                            "format": "json"
                        }

                        # 如果指定了组织，添加组织过滤
                        if tissue and tissue != "any":
                            # 将组织名称映射到GTEx的tissueSiteDetailId
                            tissue_mapping = {
                                "Whole_Blood": "Whole_Blood",
                                "Liver": "Liver",
                                "Brain_Cortex": "Brain_Cortex",
                                "Heart_Left_Ventricle": "Heart_Left_Ventricle",
                                "Muscle_Skeletal": "Muscle_Skeletal"
                            }
                            gtex_tissue = tissue_mapping.get(tissue, tissue)
                            eqtl_params["tissueSiteDetailId"] = gtex_tissue

                        eqtl_response = await client.get(
                            f"{self.gtex_url}/association/singleTissueEqtl",
                            params=eqtl_params
                        )

                        if eqtl_response.status_code == 200:
                            eqtl_data = eqtl_response.json()
                            data_count = len(eqtl_data.get('data', []))
                            logging.info(f"成功获取GTEx eQTL数据: {data_count} 项")

                            if data_count == 0:
                                logging.warning(f"GTEx数据库中不存在基因 {gene_symbol} 在组织 {tissue} 的eQTL数据")
                                logging.warning("这是数据库内容限制，不是技术错误")
                                return []

                            # 解析真实的eQTL数据
                            parsed_instruments = await self._parse_gtex_eqtl_response(eqtl_data, tissue, threshold)
                            if not parsed_instruments:
                                logging.warning(f"基因 {gene_symbol} 在组织 {tissue} 中没有满足显著性阈值 {threshold} 的eQTL")
                                logging.warning("这是统计学筛选结果，不是技术错误")
                            return parsed_instruments
                        elif eqtl_response.status_code == 404:
                            logging.warning(f"GTEx数据库中不存在基因 {gene_symbol} 的eQTL数据 (HTTP 404)")
                            logging.warning("这是数据库内容限制，不是技术错误")
                            return []
                        else:
                            logging.error(f"GTEx API技术错误: HTTP {eqtl_response.status_code}")
                            logging.error(f"这是技术连接问题，不是数据不存在的问题")
                            logging.error(f"响应内容: {eqtl_response.text[:200] if hasattr(eqtl_response, 'text') else 'N/A'}")
                            return []
                    else:
                        logging.warning(f"GTEx数据库中不存在基因: {gene_symbol}")
                        logging.warning("这是数据库内容限制，基因名称可能不正确或不在GTEx数据库中")
                        return []
                elif gene_response.status_code == 404:
                    logging.warning(f"GTEx数据库中不存在基因: {gene_symbol} (HTTP 404)")
                    logging.warning("这是数据库内容限制，不是技术错误")
                    return []
                else:
                    logging.error(f"GTEx基因查询技术错误: HTTP {gene_response.status_code}")
                    logging.error("这是技术连接问题，不是数据不存在的问题")
                    return []

        except httpx.TimeoutException as e:
            logging.error(f"GTEx API连接超时: {e}")
            logging.error("这是网络技术问题，不是数据不存在的问题")
            return []
        except httpx.ConnectError as e:
            logging.error(f"GTEx API连接失败: {e}")
            logging.error("这是网络技术问题，请检查网络连接")
            return []
        except Exception as e:
            logging.error(f"GTEx数据获取技术错误: {e}")
            logging.error("这是系统技术问题，不是数据不存在的问题")
            return []

    async def _parse_gtex_eqtl_response(self, eqtl_data: dict, target_tissue: str, threshold: float) -> List[SNPInstrument]:
        """解析GTEx API返回的真实eQTL数据"""
        instruments = []

        try:
            data_items = eqtl_data.get('data', [])

            # 按组织过滤和按p值排序
            filtered_items = []
            for item in data_items:
                tissue_id = item.get('tissueSiteDetailId', '')
                p_value = float(item.get('pValue', 1.0))

                # 组织匹配逻辑
                tissue_match = False
                if target_tissue.lower() == 'whole_blood' and 'Whole_Blood' in tissue_id:
                    tissue_match = True
                elif target_tissue.lower() == 'liver' and 'Liver' in tissue_id:
                    tissue_match = True
                elif target_tissue.lower() in tissue_id.lower():
                    tissue_match = True
                elif target_tissue == 'any':  # 接受任何组织
                    tissue_match = True

                if tissue_match and p_value <= threshold:
                    filtered_items.append(item)

            # 按p值排序，取最显著的
            filtered_items.sort(key=lambda x: float(x.get('pValue', 1.0)))

            for item in filtered_items[:10]:  # 最多取10个最显著的
                try:
                    # 首先进行业务逻辑验证
                    if not self._validate_eqtl_data(item):
                        logging.warning(f"eQTL数据验证失败，跳过: {item.get('variantId', 'unknown')}")
                        continue

                    variant_id = item.get('variantId', '')
                    p_value = float(item.get('pValue', 1.0))
                    nes = float(item.get('nes', 0.0))  # Normalized Effect Size
                    tissue_id = item.get('tissueSiteDetailId', '')

                    # 记录GTEx响应字段用于调试
                    logging.debug(f"GTEx响应字段: {list(item.keys())}")

                    # 从variantId提取SNP信息 (格式通常是 chr_pos_ref_alt_b38)
                    snp_parts = variant_id.split('_')
                    if len(snp_parts) >= 4:
                        chromosome = snp_parts[0]
                        position = snp_parts[1]
                        ref_allele = snp_parts[2]
                        alt_allele = snp_parts[3]

                        # 生成位置格式并转换为rs ID
                        if chromosome.startswith('chr'):
                            chr_num = chromosome[3:]  # 移除'chr'前缀
                        else:
                            chr_num = chromosome

                        # 生成位置格式
                        position_snp = f"{chr_num}:{position}"

                        # 尝试转换为rs ID格式
                        if self.snp_converter:
                            try:
                                logging.info(f"🔄 尝试转换SNP: {position_snp}")
                                rs_id = await self.snp_converter.convert_position_to_rsid(position_snp)
                                if rs_id:
                                    snp_id = rs_id  # 使用转换后的rs ID
                                    logging.info(f"✅ SNP转换成功: {position_snp} → {rs_id}")
                                else:
                                    snp_id = position_snp  # 转换失败时使用位置格式
                                    logging.warning(f"❌ SNP转换失败，使用位置格式: {position_snp}")
                            except Exception as e:
                                snp_id = position_snp  # 异常时使用位置格式
                                logging.error(f"❌ SNP转换异常，使用位置格式: {position_snp}, 错误: {e}")
                        else:
                            snp_id = position_snp  # 转换器未初始化时使用位置格式
                            logging.warning(f"⚠️ SNP转换器未初始化，使用位置格式: {position_snp}")
                    else:
                        # 如果格式不符合预期，使用原始ID
                        snp_id = variant_id
                        ref_allele = 'A'
                        alt_allele = 'G'

                    # 使用文献标准方法提取效应大小和标准误
                    beta, se = self._extract_gtex_effect_size(nes, p_value)

                    instrument = SNPInstrument(
                        snp_id=snp_id,
                        effect_allele=alt_allele,
                        other_allele=ref_allele,
                        beta=beta,
                        se=se,
                        p_value=p_value,
                        source_db=f"GTEx_{tissue_id}_real"
                    )

                    # 验证生成的工具变量
                    if self._validate_snp_instrument(instrument):
                        instruments.append(instrument)
                        logging.debug(f"✅ SNP工具变量验证通过: {snp_id}")
                    else:
                        logging.warning(f"❌ SNP工具变量验证失败: {snp_id}")

                except (ValueError, KeyError) as e:
                    logging.warning(f"解析GTEx eQTL项目失败: {e}")
                    continue

            # 记录验证统计信息
            total_items = len(filtered_items[:10])
            valid_instruments = len(instruments)
            rejected_items = total_items - valid_instruments

            logging.info(f"✅ eQTL数据处理完成:")
            logging.info(f"   - 处理的eQTL项目: {total_items}")
            logging.info(f"   - 通过验证的工具变量: {valid_instruments}")
            logging.info(f"   - 被拒绝的项目: {rejected_items}")

            if rejected_items > 0:
                rejection_rate = (rejected_items / total_items) * 100
                logging.warning(f"   - 数据拒绝率: {rejection_rate:.1f}%")

                if rejection_rate > 50:
                    logging.error("⚠️ 数据拒绝率过高，可能存在数据质量问题")

            return instruments

        except Exception as e:
            logging.error(f"解析GTEx eQTL响应失败: {e}")
            return []

    def _validate_eqtl_data(self, item: dict) -> bool:
        """
        验证eQTL数据的关键业务逻辑

        Args:
            item: GTEx API返回的单个eQTL数据项

        Returns:
            bool: 数据是否通过验证
        """
        try:
            # 1. 验证必需字段存在
            required_fields = ['variantId', 'pValue', 'nes']
            for field in required_fields:
                if field not in item or item[field] is None:
                    logging.warning(f"缺少必需字段: {field}")
                    return False

            # 2. 验证p值范围 (0 <= p <= 1)
            p_value = float(item.get('pValue', 1.0))
            if not (0 <= p_value <= 1):
                logging.warning(f"p值超出有效范围 [0,1]: {p_value}")
                return False

            # 3. 验证效应大小合理性 (NES通常在-10到10之间)
            nes = float(item.get('nes', 0))
            if abs(nes) > 15:  # 极端大的效应，可能是数据错误
                logging.warning(f"NES效应大小异常: {nes}")
                return False

            # 4. 验证variant ID格式
            variant_id = item.get('variantId', '')
            if not variant_id or len(variant_id) < 5:  # 基本长度检查
                logging.warning(f"variant ID格式异常: {variant_id}")
                return False

            # 5. 验证等位基因信息（如果variant ID包含等位基因）
            if '_' in variant_id:
                parts = variant_id.split('_')
                if len(parts) >= 4:
                    ref_allele = parts[2].upper()
                    alt_allele = parts[3].upper()
                    valid_alleles = {'A', 'T', 'G', 'C', 'I', 'D'}  # 包括插入(I)和删除(D)

                    # 检查等位基因是否有效
                    if ref_allele not in valid_alleles or alt_allele not in valid_alleles:
                        # 对于复杂变异（如多碱基），只记录警告但不排除
                        if len(ref_allele) > 1 or len(alt_allele) > 1:
                            logging.info(f"检测到复杂变异: {ref_allele}/{alt_allele}")
                        else:
                            logging.warning(f"无效的等位基因: {ref_allele}/{alt_allele}")
                            return False

                    # 检查等位基因是否相同（无意义的变异）
                    if ref_allele == alt_allele:
                        logging.warning(f"参考和替代等位基因相同: {ref_allele}")
                        return False

            # 6. 验证染色体信息（如果variant ID包含染色体）
            if '_' in variant_id:
                parts = variant_id.split('_')
                if len(parts) >= 2:
                    chromosome = parts[0]
                    position = parts[1]

                    # 验证染色体格式
                    if chromosome.startswith('chr'):
                        chr_num = chromosome[3:]
                    else:
                        chr_num = chromosome

                    # 验证染色体编号
                    valid_chromosomes = set(map(str, range(1, 23))) | {'X', 'Y', 'MT', 'M'}
                    if chr_num not in valid_chromosomes:
                        logging.warning(f"无效的染色体: {chromosome}")
                        return False

                    # 验证位置是否为正整数
                    try:
                        pos = int(position)
                        if pos <= 0:
                            logging.warning(f"无效的染色体位置: {position}")
                            return False
                    except ValueError:
                        logging.warning(f"染色体位置不是数字: {position}")
                        return False

            # 7. 验证组织信息
            tissue_id = item.get('tissueSiteDetailId', '')
            if tissue_id and len(tissue_id) < 3:  # 基本长度检查
                logging.warning(f"组织ID异常: {tissue_id}")
                return False

            return True

        except (ValueError, TypeError) as e:
            logging.warning(f"数据验证过程中发生错误: {e}")
            return False
        except Exception as e:
            logging.error(f"数据验证异常: {e}")
            return False

    def _validate_snp_instrument(self, instrument: SNPInstrument) -> bool:
        """
        验证生成的SNP工具变量的质量

        Args:
            instrument: 生成的SNP工具变量对象

        Returns:
            bool: 工具变量是否通过验证
        """
        try:
            # 1. 验证SNP ID不为空
            if not instrument.snp_id or len(instrument.snp_id.strip()) == 0:
                logging.warning("SNP ID为空")
                return False

            # 2. 验证等位基因不为空且不相同
            if not instrument.effect_allele or not instrument.other_allele:
                logging.warning("等位基因信息缺失")
                return False

            if instrument.effect_allele == instrument.other_allele:
                logging.warning(f"效应等位基因和参考等位基因相同: {instrument.effect_allele}")
                return False

            # 3. 验证beta值合理性
            if abs(instrument.beta) > 2.0:  # 转换后的beta值通常不会太大
                logging.warning(f"Beta值异常: {instrument.beta}")
                return False

            # 4. 验证标准误为正数
            if instrument.se <= 0:
                logging.warning(f"标准误必须为正数: {instrument.se}")
                return False

            # 5. 验证p值范围
            if not (0 <= instrument.p_value <= 1):
                logging.warning(f"p值超出范围: {instrument.p_value}")
                return False

            # 6. 验证统计一致性 (|beta/se| 应该与p值大致一致)
            if instrument.se > 0:
                z_score = abs(instrument.beta / instrument.se)
                # 对于非常显著的p值，z_score应该比较大
                if instrument.p_value < 1e-10 and z_score < 3:
                    logging.warning(f"统计不一致: p={instrument.p_value:.2e}, z={z_score:.2f}")
                    # 不直接返回False，只记录警告，因为这可能是转换误差

            # 7. 验证数据来源标记
            if not instrument.source_db or 'GTEx' not in instrument.source_db:
                logging.warning(f"数据来源标记异常: {instrument.source_db}")
                return False

            return True

        except Exception as e:
            logging.error(f"SNP工具变量验证异常: {e}")
            return False

    def _extract_gtex_effect_size(self, nes: float, p_value: float) -> tuple:
        """
        从GTEx API响应中提取效应大小和标准误 - 文献标准方法

        文献标准方法：
        - 直接使用GTEx的NES (Normalized Effect Size)
        - 从p值统计学正确地计算标准误
        - 保持beta/se与p值的数学一致性

        参考文献：
        - SMR软件标准做法
        - MR-Base平台方法
        - Hemani et al., eLife 2018

        Args:
            nes: GTEx的标准化效应大小
            p_value: 显著性p值

        Returns:
            tuple: (beta, se)
        """
        try:
            # 文献标准方法：直接使用NES，从p值正确计算SE
            if 0 < p_value < 1:
                from scipy import stats
                z_score = abs(stats.norm.ppf(p_value / 2))
                if z_score > 0:
                    beta = nes  # 直接使用GTEx的NES
                    se = abs(nes / z_score)  # 从z-score统计学正确地计算标准误
                    logging.info(f"📊 文献标准方法: nes={beta:.6f}, 计算se={se:.6f}, z_score={z_score:.2f}")
                    return beta, se

            # 异常情况的保守处理
            logging.warning(f"⚠️ p值异常，使用保守处理: p_value={p_value}")
            return nes, 0.05

        except Exception as e:
            logging.error(f"提取GTEx效应大小失败: {e}")
            # 异常时使用保守的默认值
            return nes, 0.05





    def _parse_gtex_response(self, data: Dict, tissue: str) -> List[SNPInstrument]:
        """解析GTEx API响应"""
        instruments = []

        # GTEx API返回的数据在'data'字段中
        for item in data.get('data', []):
            # 只处理指定组织的数据
            if item.get('tissueSiteDetailId') == tissue:
                # 解析variant ID获取等位基因信息
                variant_id = item.get('variantId', '')
                effect_allele = 'A'  # 默认值
                other_allele = 'G'   # 默认值

                # 从variant ID解析等位基因 (格式: chr19_44799247_G_A_b38)
                if '_' in variant_id:
                    parts = variant_id.split('_')
                    if len(parts) >= 4:
                        other_allele = parts[2]  # 参考等位基因
                        effect_allele = parts[3]  # 替代等位基因

                # GTEx使用NES (Normalized Effect Size)作为效应大小
                nes = item.get('nes', 0)
                p_value = item.get('pValue', 1)

                # 估算标准误 (基于NES和p值)
                if p_value > 0 and nes != 0:
                    import math
                    z_score = abs(nes)
                    se = abs(nes / z_score) if z_score > 0 else 0.1
                else:
                    se = 0.1

                instruments.append(SNPInstrument(
                    snp_id=item.get('snpId', variant_id),
                    effect_allele=effect_allele,
                    other_allele=other_allele,
                    beta=float(nes),
                    se=float(se),
                    p_value=float(p_value),
                    source_db=f"GTEx_{tissue}_real"
                ))

        return instruments

    def _parse_eqtlgen_response(self, data: Dict) -> List[SNPInstrument]:
        """解析eQTLGen API响应"""
        instruments = []

        for item in data.get('eqtls', []):
            instruments.append(SNPInstrument(
                snp_id=item.get('SNP', ''),
                effect_allele=item.get('AssessedAllele', ''),
                other_allele=item.get('OtherAllele', ''),
                beta=float(item.get('Zscore', 0)) * 0.1,  # 转换Z分数为效应大小
                se=0.1,  # 估计标准误
                p_value=float(item.get('Pvalue', 1)),
                source_db="eQTLGen"
            ))

        return instruments

    def _deduplicate_instruments(self, instruments: List[SNPInstrument]) -> List[SNPInstrument]:
        """去除重复的SNP工具变量"""
        seen_snps = set()
        unique_instruments = []

        for instrument in instruments:
            if instrument.snp_id not in seen_snps:
                seen_snps.add(instrument.snp_id)
                unique_instruments.append(instrument)

        return unique_instruments

# 初始化真实eQTL数据客户端
eqtl_client = RealEQTLDataClient()




# 2. 使用装饰器定义工具
@mcp.tool()
async def find_eqtl_instruments(params: EQTLToolInput) -> List[SNPInstrument]:
    """
    Finds significant eQTLs for a given gene to be used as instrumental variables.

    This tool queries a public API (emulating the IEU OpenGWAS database) to retrieve
    SNPs associated with the expression of a specified gene in a specific tissue.
    """
    logging.info(f"Received request to find instruments for gene: {params.gene_symbol}")
    
    # 使用真实eQTL数据
    logging.info(f"Fetching real eQTL data for {params.gene_symbol} in {params.tissue}")

    try:
        # 只获取真实eQTL工具变量，严禁使用任何模拟数据
        instruments = await eqtl_client.get_eqtl_instruments(
            gene_symbol=params.gene_symbol,
            tissue=params.tissue,
            significance_threshold=params.significance_threshold
        )

        if instruments:
            logging.info(f"✅ 成功从GTEx获取 {len(instruments)} 个真实eQTL工具变量")
            logging.info(f"基因: {params.gene_symbol}, 组织: {params.tissue}")
            # 记录数据来源验证
            for i, inst in enumerate(instruments[:3]):  # 记录前3个作为验证
                logging.info(f"工具变量 {i+1}: {inst.snp_id}, beta={inst.beta:.3f}, p={inst.p_value:.2e}, 来源={inst.source_db}")
            return instruments
        else:
            logging.warning(f"⚠️ GTEx数据库中未找到基因 {params.gene_symbol} 在组织 {params.tissue} 的eQTL数据")
            logging.warning("原因可能是: 1) 基因名称不正确 2) 该基因在此组织中无显著eQTL 3) 显著性阈值过严格")
            return []

    except Exception as e:
        logging.error(f"❌ eQTL数据获取技术错误: {e}")
        logging.error("这是系统技术问题，不是数据不存在的问题")
        return []

# 3. 配置服务器入口点
# 这使得包可以通过 `python -m mcp_server_eqtl` 运行
def run():
    mcp.run()

if __name__ == "__main__":
    run()
