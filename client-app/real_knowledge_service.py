#!/usr/bin/env python3
"""
真实知识服务 - 直接集成到客户端
使用真实的生物学数据库API获取基因注释和药物信息
"""

import logging
import asyncio
from typing import Dict, Any, List, Optional
import httpx
import time

logger = logging.getLogger(__name__)

class RealKnowledgeService:
    """
    真实知识服务，直接调用生物学数据库API
    """
    
    def __init__(self):
        self.timeout = 30.0
        self.last_request_time = 0
        self.min_request_interval = 0.5  # 500ms between requests
        
        # 初始化bioservices（如果可用）
        self.bioservices_available = False
        try:
            import bioservices
            self.bioservices_available = True
            logger.info("✅ bioservices库可用，将使用真实生物学数据库")
        except ImportError:
            logger.warning("⚠️ bioservices库不可用，将使用REST API备用方案")
    
    async def _rate_limit(self):
        """实施速率限制"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            await asyncio.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    async def get_gene_annotation(self, gene_symbol: str) -> Dict[str, Any]:
        """
        获取基因注释信息 - 使用真实API数据
        """
        logger.info(f"🧬 获取基因 {gene_symbol} 的真实注释信息")
        
        try:
            # 方法1: 使用bioservices获取UniProt信息
            if self.bioservices_available:
                uniprot_data = await self._get_uniprot_data(gene_symbol)
                if uniprot_data:
                    return {
                        "gene_info": uniprot_data,
                        "data_source": "Real_UniProt_API",
                        "gene_symbol": gene_symbol
                    }
            
            # 方法2: 使用REST API备用方案
            rest_data = await self._get_gene_info_rest(gene_symbol)
            if rest_data:
                return {
                    "gene_info": rest_data,
                    "data_source": "Real_REST_API",
                    "gene_symbol": gene_symbol
                }
            
            # 如果都失败，返回数据不可用
            return {
                "error": f"No real data available for gene {gene_symbol}",
                "data_source": "Real_API_No_Data",
                "message": f"真实生物学数据库中未找到基因 {gene_symbol} 的注释信息"
            }
            
        except Exception as e:
            logger.error(f"基因注释获取失败: {e}")
            return {
                "error": f"Gene annotation failed: {str(e)}",
                "data_source": "Service_Error",
                "message": "生物学数据库服务调用失败"
            }
    
    async def _get_uniprot_data(self, gene_symbol: str) -> Optional[Dict[str, Any]]:
        """使用REST API直接获取UniProt数据，避免bioservices的问题"""
        try:
            await self._rate_limit()

            # 使用UniProt REST API直接查询
            logger.info(f"🔍 通过REST API在UniProt中搜索基因: {gene_symbol}")

            import httpx

            # 构建查询URL
            base_url = "https://rest.uniprot.org/uniprotkb/search"
            params = {
                "query": f"gene:{gene_symbol} AND organism_id:9606",
                "format": "json",
                "size": 5
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(base_url, params=params)

                if response.status_code == 200:
                    data = response.json()

                    if data.get("results") and len(data["results"]) > 0:
                        entry = data["results"][0]

                        # 提取基本信息
                        protein_name = "Unknown"
                        if "proteinDescription" in entry:
                            rec_name = entry["proteinDescription"].get("recommendedName", {})
                            if "fullName" in rec_name:
                                protein_name = rec_name["fullName"]["value"]

                        gene_names = []
                        if "genes" in entry:
                            for gene in entry["genes"]:
                                if "geneName" in gene:
                                    gene_names.append(gene["geneName"]["value"])

                        return {
                            "uniprot_id": entry.get("primaryAccession", "Unknown"),
                            "protein_name": protein_name,
                            "gene_name": gene_names[0] if gene_names else gene_symbol,
                            "organism": "Homo sapiens",
                            "source": "UniProt_REST_API"
                        }
                    else:
                        logger.info(f"UniProt REST API中未找到基因 {gene_symbol}")
                        return None
                else:
                    logger.warning(f"UniProt REST API请求失败: {response.status_code}")
                    return None

        except Exception as e:
            logger.warning(f"UniProt REST API查询失败: {e}")
            return None
            
        except Exception as e:
            logger.error(f"UniProt查询失败: {e}")
            return None
    
    async def _get_gene_info_rest(self, gene_symbol: str) -> Optional[Dict[str, Any]]:
        """使用REST API获取基因信息"""
        try:
            await self._rate_limit()
            
            # 使用Ensembl REST API
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                url = f"https://rest.ensembl.org/lookup/symbol/homo_sapiens/{gene_symbol}"
                headers = {"Content-Type": "application/json"}
                
                logger.info(f"🔍 Calling Ensembl API: {url}")
                response = await client.get(url, headers=headers)
                
                if response.status_code == 200:
                    data = response.json()
                    return {
                        "ensembl_id": data.get("id"),
                        "gene_name": data.get("display_name", gene_symbol),
                        "description": data.get("description", ""),
                        "chromosome": data.get("seq_region_name"),
                        "start": data.get("start"),
                        "end": data.get("end"),
                        "strand": data.get("strand"),
                        "biotype": data.get("biotype"),
                        "source": "Ensembl_REST_API"
                    }
                else:
                    logger.warning(f"Ensembl API返回错误: {response.status_code}")
                    return None
                    
        except Exception as e:
            logger.error(f"Ensembl REST API调用失败: {e}")
            return None
    
    async def get_drug_targets(self, gene_symbol: str) -> Dict[str, Any]:
        """
        获取药物靶点信息 - 使用真实API数据
        """
        logger.info(f"💊 获取基因 {gene_symbol} 的真实药物靶点信息")
        
        try:
            # 方法1: 使用bioservices获取ChEMBL数据
            if self.bioservices_available:
                chembl_data = await self._get_chembl_data(gene_symbol)
                if chembl_data:
                    return {
                        "drug_targets": chembl_data,
                        "data_source": "Real_ChEMBL_API",
                        "gene_symbol": gene_symbol
                    }
            

            
            # 如果都失败，返回数据不可用
            return {
                "error": f"No real drug data available for gene {gene_symbol}",
                "data_source": "Real_API_No_Data",
                "message": f"真实药物数据库中未找到基因 {gene_symbol} 的靶点信息"
            }
            
        except Exception as e:
            logger.error(f"药物靶点获取失败: {e}")
            return {
                "error": f"Drug target analysis failed: {str(e)}",
                "data_source": "Service_Error",
                "message": "药物数据库服务调用失败"
            }
    
    async def _get_chembl_data(self, gene_symbol: str) -> Optional[List[Dict[str, Any]]]:
        """使用ChEMBL REST API获取药物数据"""
        try:
            await self._rate_limit()

            logger.info(f"🔍 通过REST API在ChEMBL中搜索靶点: {gene_symbol}")

            import httpx

            # 使用ChEMBL REST API搜索靶点
            base_url = "https://www.ebi.ac.uk/chembl/api/data/target/search"
            params = {
                "q": gene_symbol,
                "format": "json",
                "limit": 5
            }

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(base_url, params=params)

                if response.status_code == 200:
                    data = response.json()
                    targets = data.get("targets", [])
            
                    if targets:
                        drug_data = []
                        for target in targets[:3]:  # 限制前3个结果
                            target_id = target.get("target_chembl_id")
                            if target_id:
                                # 获取与该靶点相关的活性数据
                                activity_url = f"https://www.ebi.ac.uk/chembl/api/data/activity"
                                activity_params = {
                                    "target_chembl_id": target_id,
                                    "format": "json",
                                    "limit": 5
                                }

                                activity_response = await client.get(activity_url, params=activity_params)
                                if activity_response.status_code == 200:
                                    activities = activity_response.json().get("activities", [])

                                    for activity in activities:
                                        compound_id = activity.get('molecule_chembl_id')
                                        activity_value = activity.get('standard_value')

                                        # 跳过无效数据：没有化合物ID或活性值为None/空
                                        if not compound_id or activity_value is None:
                                            continue

                                        # 跳过活性值为0或负数的无意义数据
                                        try:
                                            if float(activity_value) <= 0:
                                                continue
                                        except (ValueError, TypeError):
                                            continue

                                        # 直接使用ChEMBL ID作为显示名称（更可靠）
                                        compound_name = compound_id

                                        drug_data.append({
                                            "target_id": target_id,
                                            "target_name": target.get('pref_name', 'Unknown'),
                                            "compound_id": compound_id,
                                            "compound_name": compound_name,
                                            "activity_type": activity.get('standard_type'),
                                            "activity_value": activity_value,
                                            "activity_units": activity.get('standard_units'),
                                            "source": "ChEMBL_REST_API"
                                        })

                        # 去重：基于化合物ID、活性类型和数值的组合
                        if drug_data:
                            unique_drugs = []
                            seen_combinations = set()

                            for drug in drug_data:
                                # 创建唯一标识符
                                key = (
                                    drug['compound_id'],
                                    drug['activity_type'],
                                    str(drug['activity_value']),
                                    drug['activity_units']
                                )

                                if key not in seen_combinations:
                                    seen_combinations.add(key)
                                    unique_drugs.append(drug)

                            logger.info(f"过滤前: {len(drug_data)} 条记录，去重后: {len(unique_drugs)} 条记录")
                            return unique_drugs

                        return None
                    else:
                        logger.info(f"ChEMBL REST API中未找到基因 {gene_symbol} 的靶点信息")
                        return None
                else:
                    logger.warning(f"ChEMBL REST API请求失败: {response.status_code}")
                    return None

        except Exception as e:
            logger.error(f"ChEMBL查询失败: {e}")
            return None

    

