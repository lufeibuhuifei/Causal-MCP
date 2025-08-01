#!/usr/bin/env python3
"""
集中JWT令牌管理系统
统一管理OpenGWAS API的JWT令牌配置
"""

import os
import json
import logging
import asyncio
import httpx
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from datetime import datetime, timezone
try:
    import jwt
    JWT_DECODE_AVAILABLE = True
except ImportError:
    JWT_DECODE_AVAILABLE = False

logger = logging.getLogger(__name__)

class JWTManager:
    """JWT令牌集中管理器"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_file = self.project_root / "jwt_config.json"
        self.legacy_config_files = [
            self.project_root / "opengwas_config.json",
            self.project_root / "mcp-server-gwas" / "src" / "opengwas_config.json",
            # ieugwaspy配置文件
            self.project_root / ".ieugwaspy.json",
            # 注意：只有GWAS服务器真正需要JWT令牌
            # eQTL、MR、Knowledge服务器不需要OpenGWAS JWT
        ]
        
    def get_jwt_token(self) -> Optional[str]:
        """
        获取JWT令牌，按优先级顺序：
        1. 环境变量 OPENGWAS_JWT
        2. 集中配置文件 jwt_config.json
        3. 遗留配置文件
        """
        # 1. 环境变量（最高优先级）
        env_token = os.environ.get('OPENGWAS_JWT')
        if env_token:
            logger.info("使用环境变量中的JWT令牌")
            return env_token
        
        # 2. 集中配置文件
        token = self._load_from_central_config()
        if token:
            logger.info("使用集中配置文件中的JWT令牌")
            return token
        
        # 3. 遗留配置文件
        token = self._load_from_legacy_configs()
        if token:
            logger.info("使用遗留配置文件中的JWT令牌")
            return token
        
        logger.warning("未找到有效的JWT令牌")
        return None
    
    def _load_from_central_config(self) -> Optional[str]:
        """从集中配置文件加载JWT令牌"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config.get('jwt_token')
        except Exception as e:
            logger.debug(f"读取集中配置文件失败: {e}")
        return None
    
    def _load_from_legacy_configs(self) -> Optional[str]:
        """从遗留配置文件加载JWT令牌"""
        for config_path in self.legacy_config_files:
            try:
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)

                        # 处理不同的配置文件格式
                        token = None
                        if config_path.name == '.ieugwaspy.json':
                            # ieugwaspy配置文件格式: {"jwt": "token"}
                            token = config.get('jwt')
                        else:
                            # 标准配置文件格式: {"jwt_token": "token"}
                            token = config.get('jwt_token')

                        if token:
                            logger.info(f"从遗留配置文件加载令牌: {config_path}")
                            return token
            except Exception as e:
                logger.debug(f"读取遗留配置文件失败 {config_path}: {e}")
        return None
    
    def save_jwt_token(self, jwt_token: str, description: str = "") -> bool:
        """
        保存JWT令牌到集中配置文件
        
        Args:
            jwt_token: JWT令牌字符串
            description: 令牌描述
            
        Returns:
            保存是否成功
        """
        try:
            config = {
                "jwt_token": jwt_token,
                "description": description or "OpenGWAS API JWT token for Causal-MCP",
                "configured_at": datetime.now(timezone.utc).isoformat(),
                "configured_by": "user"
            }
            
            # 确保目录存在
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存到集中配置文件
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            # 设置环境变量
            os.environ['OPENGWAS_JWT'] = jwt_token
            
            logger.info(f"JWT令牌已保存到集中配置文件: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"保存JWT令牌失败: {e}")
            return False
    
    def is_token_configured(self) -> bool:
        """检查是否已配置JWT令牌"""
        return self.get_jwt_token() is not None

    def is_token_expired(self) -> bool:
        """检查JWT令牌是否已过期"""
        jwt_token = self.get_jwt_token()
        if not jwt_token:
            return True  # 没有令牌视为过期

        if JWT_DECODE_AVAILABLE:
            try:
                # 解码JWT令牌检查过期时间
                decoded = jwt.decode(jwt_token, options={"verify_signature": False})
                exp = decoded.get('exp')
                if exp:
                    exp_time = datetime.fromtimestamp(exp, tz=timezone.utc)
                    now = datetime.now(timezone.utc)
                    return exp_time <= now
            except jwt.InvalidTokenError:
                return True  # 令牌格式无效视为过期

        # 如果无法解码，返回False（假设令牌有效，让API调用来验证）
        return False

    def get_token_expiry_days(self) -> int:
        """获取JWT令牌剩余有效天数"""
        jwt_token = self.get_jwt_token()
        if not jwt_token:
            return -1

        if JWT_DECODE_AVAILABLE:
            try:
                decoded = jwt.decode(jwt_token, options={"verify_signature": False})
                exp = decoded.get('exp')
                if exp:
                    exp_time = datetime.fromtimestamp(exp, tz=timezone.utc)
                    now = datetime.now(timezone.utc)
                    days_remaining = (exp_time - now).days
                    return max(0, days_remaining)
            except jwt.InvalidTokenError:
                return -1

        return -1  # 无法确定过期时间

    def is_token_expiring_soon(self, days_threshold: int = 3) -> bool:
        """检查JWT令牌是否即将过期"""
        days_remaining = self.get_token_expiry_days()
        if days_remaining == -1:
            return False  # 无法确定过期时间
        return 0 <= days_remaining <= days_threshold

    async def is_token_valid(self) -> bool:
        """异步检查JWT令牌是否有效（包括网络验证）"""
        jwt_token = self.get_jwt_token()
        if not jwt_token:
            return False

        # 首先检查本地过期时间
        if self.is_token_expired():
            return False

        # 然后进行网络验证
        try:
            is_valid, _ = await self.test_jwt_token(jwt_token)
            return is_valid
        except Exception:
            return False
    
    async def test_jwt_token(self, jwt_token: Optional[str] = None) -> Tuple[bool, str]:
        """
        测试JWT令牌是否有效
        
        Args:
            jwt_token: 要测试的令牌，如果为None则使用当前配置的令牌
            
        Returns:
            (是否有效, 测试结果消息)
        """
        if jwt_token is None:
            jwt_token = self.get_jwt_token()
        
        if not jwt_token:
            return False, "未找到JWT令牌"
        
        try:
            # 首先检查令牌格式和过期时间（如果JWT库可用）
            if JWT_DECODE_AVAILABLE:
                try:
                    # 解码JWT令牌（不验证签名，只检查格式和过期时间）
                    decoded = jwt.decode(jwt_token, options={"verify_signature": False})
                    exp = decoded.get('exp')
                    if exp:
                        exp_time = datetime.fromtimestamp(exp, tz=timezone.utc)
                        now = datetime.now(timezone.utc)
                        if exp_time <= now:
                            return False, f"JWT令牌已过期 (过期时间: {exp_time.strftime('%Y-%m-%d %H:%M:%S UTC')})"
                except jwt.InvalidTokenError as e:
                    return False, f"JWT令牌格式无效: {e}"
            
            # 使用官方推荐的 /user 端点验证JWT令牌
            headers = {"Authorization": f"Bearer {jwt_token}"}

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    "https://api.opengwas.io/api/user",
                    headers=headers
                )

                if response.status_code == 200:
                    try:
                        data = response.json()
                        # 根据实际API结构提取用户信息
                        user_info = data.get('user', {})
                        user_email = user_info.get('uid', 'Unknown')
                        user_name = f"{user_info.get('first_name', '')} {user_info.get('last_name', '')}".strip()
                        jwt_valid_until = user_info.get('jwt_valid_until', '')

                        # 构建显示信息
                        if user_email != 'Unknown':
                            message = f"✅ JWT token verification successful! User: {user_email}"
                            if user_name:
                                message += f" ({user_name})"
                            if jwt_valid_until:
                                message += f", Valid until: {jwt_valid_until}"
                            return True, message
                        else:
                            return True, "✅ JWT token verification successful, can access OpenGWAS API"
                    except:
                        return True, "✅ JWT token verification successful, can access OpenGWAS API"
                elif response.status_code == 401:
                    return False, "❌ JWT token invalid or expired"
                elif response.status_code == 403:
                    return False, "❌ JWT token insufficient permissions"
                elif response.status_code == 429:
                    return False, "❌ API request rate limit exceeded, please try again later"
                else:
                    return False, f"❌ API test failed, status code: {response.status_code}"

        except httpx.TimeoutException:
            return False, "❌ API connection timeout, please check network connection"
        except Exception as e:
            return False, f"❌ Test failed: {str(e)}"
    
    def get_token_info(self) -> Dict[str, Any]:
        """获取当前令牌的详细信息"""
        jwt_token = self.get_jwt_token()
        if not jwt_token:
            return {"configured": False}
        
        info = {"configured": True, "token_length": len(jwt_token)}
        
        if JWT_DECODE_AVAILABLE:
            try:
                # 解码JWT令牌获取信息
                decoded = jwt.decode(jwt_token, options={"verify_signature": False})

                # 提取有用信息
                if 'exp' in decoded:
                    exp_time = datetime.fromtimestamp(decoded['exp'], tz=timezone.utc)
                    info['expires_at'] = exp_time.isoformat()
                    info['expires_in_days'] = (exp_time - datetime.now(timezone.utc)).days

                if 'iat' in decoded:
                    issued_time = datetime.fromtimestamp(decoded['iat'], tz=timezone.utc)
                    info['issued_at'] = issued_time.isoformat()

                if 'sub' in decoded:
                    info['subject'] = decoded['sub']

            except Exception as e:
                logger.debug(f"解析JWT令牌信息失败: {e}")
                info['parse_error'] = str(e)
        else:
            info['jwt_decode_unavailable'] = "JWT解码库不可用，无法解析令牌详情"
        
        return info
    
    def migrate_legacy_configs(self) -> bool:
        """迁移遗留配置文件到集中配置"""
        try:
            # 查找第一个有效的遗留配置
            for config_path in self.legacy_config_files:
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = json.load(f)
                        token = config.get('jwt_token')
                        if token:
                            description = config.get('description', 'Migrated from legacy config')
                            if self.save_jwt_token(token, description):
                                logger.info(f"成功迁移JWT令牌从 {config_path}")
                                return True
            
            logger.warning("未找到可迁移的遗留配置")
            return False
            
        except Exception as e:
            logger.error(f"迁移遗留配置失败: {e}")
            return False

# 全局JWT管理器实例
jwt_manager = JWTManager()

def get_jwt_token() -> Optional[str]:
    """获取JWT令牌的便捷函数"""
    return jwt_manager.get_jwt_token()

def is_jwt_configured() -> bool:
    """检查JWT是否已配置的便捷函数"""
    return jwt_manager.is_token_configured()

async def test_jwt_token(token: Optional[str] = None) -> Tuple[bool, str]:
    """测试JWT令牌的便捷函数"""
    return await jwt_manager.test_jwt_token(token)
