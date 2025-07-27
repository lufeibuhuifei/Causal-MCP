#!/usr/bin/env python3
"""
🔄 SNP位置→rs ID转换服务
将GTEx的位置格式(1:55055436)转换为OpenGWAS兼容的rs ID格式
"""

import asyncio
import httpx
import logging
from typing import List, Dict, Optional
import json
import time

class SNPConverter:
    """SNP位置到rs ID转换器"""
    
    def __init__(self):
        self.cache = {}  # 缓存转换结果
        self.timeout = 30.0
        
        # 多个转换服务的URL
        self.conversion_services = [
            {
                'name': 'Ensembl REST API',
                'url': 'https://rest.ensembl.org/variation/human',
                'method': 'ensembl'
            }
        ]
    
    async def convert_position_to_rsid(self, position_snp: str) -> Optional[str]:
        """
        将位置格式SNP转换为rs ID
        
        Args:
            position_snp: 位置格式SNP，如 "1:55055436"
            
        Returns:
            rs ID字符串，如 "rs11591147"，如果转换失败返回None
        """
        
        # 检查缓存
        if position_snp in self.cache:
            return self.cache[position_snp]
        
        # 解析位置信息
        if ':' not in position_snp:
            logging.warning(f"无效的位置格式: {position_snp}")
            return None
        
        try:
            chr_num, pos = position_snp.split(':')
            chr_num = chr_num.strip()
            pos = pos.strip()
            
            # 使用Ensembl转换
            rs_id = await self._convert_with_ensembl(chr_num, pos)
            if rs_id:
                # 缓存结果
                self.cache[position_snp] = rs_id
                logging.info(f"SNP转换成功: {position_snp} → {rs_id}")
                return rs_id
            
            logging.warning(f"SNP转换失败: {position_snp}")
            return None
            
        except Exception as e:
            logging.error(f"SNP转换异常: {position_snp}, {e}")
            return None
    
    async def _convert_with_ensembl(self, chr_num: str, pos: str) -> Optional[str]:
        """使用Ensembl REST API转换"""
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            
            # Ensembl API查询位置的变异
            url = f"https://rest.ensembl.org/overlap/region/human/{chr_num}:{pos}-{pos}"
            
            headers = {
                'Content-Type': 'application/json'
            }
            
            params = {
                'feature': 'variation'
            }
            
            response = await client.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                
                if isinstance(data, list) and len(data) > 0:
                    for variant in data:
                        variant_name = variant.get('id', '')
                        if variant_name.startswith('rs'):
                            return variant_name
            
            return None
    
    async def convert_batch(self, position_snps: List[str]) -> Dict[str, Optional[str]]:
        """
        批量转换SNP位置到rs ID
        
        Args:
            position_snps: 位置格式SNP列表
            
        Returns:
            转换结果字典 {position_snp: rs_id}
        """
        
        results = {}
        
        # 并发转换，但限制并发数量避免API限制
        semaphore = asyncio.Semaphore(3)  # 最多3个并发请求
        
        async def convert_single(pos_snp):
            async with semaphore:
                rs_id = await self.convert_position_to_rsid(pos_snp)
                results[pos_snp] = rs_id
                # 添加延迟避免API限制
                await asyncio.sleep(0.5)
        
        # 创建并发任务
        tasks = [convert_single(snp) for snp in position_snps]
        
        # 等待所有转换完成
        await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    def get_conversion_stats(self) -> Dict:
        """获取转换统计信息"""
        total = len(self.cache)
        successful = sum(1 for v in self.cache.values() if v is not None)
        
        return {
            'total_conversions': total,
            'successful_conversions': successful,
            'success_rate': successful / total * 100 if total > 0 else 0,
            'cache_size': total
        }
