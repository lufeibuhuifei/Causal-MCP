# client-app/llm_service.py
"""
LLM服务管理器，支持多种LLM提供商（Ollama本地模型和DeepSeek API）
"""

import logging
from typing import Optional, Dict, Any, List
import asyncio
import requests
import json
import re
import os
from pathlib import Path
from abc import ABC, abstractmethod
try:
    from .i18n import get_text
except ImportError:
    from i18n import get_text

logger = logging.getLogger(__name__)

class LLMProvider(ABC):
    """LLM提供商抽象基类"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.is_available = False

    @abstractmethod
    async def initialize(self) -> bool:
        """初始化LLM连接"""
        pass

    @abstractmethod
    async def generate_text(self, prompt: str, max_length: int = 1024) -> Optional[str]:
        """生成文本"""
        pass

    @abstractmethod
    async def test_connection(self) -> tuple[bool, str, List[str]]:
        """测试连接"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """获取提供商名称"""
        pass

class OllamaProvider(LLMProvider):
    """Ollama本地LLM提供商"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = config.get("base_url", "http://localhost:11434")
        self.model_name = config.get("model_name", "deepseek-r1:1.5b")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2048)
        self.timeout = config.get("timeout", 60)

    def get_provider_name(self) -> str:
        return "Ollama"

    async def initialize(self) -> bool:
        """初始化Ollama连接"""
        try:
            logger.info(f"正在初始化Ollama服务，模型: {self.model_name}")

            # 测试Ollama连接
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                logger.warning(f"Ollama API不可用: {response.status_code}")
                self.is_available = False
                return False

            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]

            if self.model_name not in available_models:
                logger.warning(f"模型 {self.model_name} 不可用。可用模型: {available_models}")
                self.is_available = False
                return False

            # 测试生成
            test_payload = {
                "model": self.model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 20
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=test_payload,
                timeout=min(30, self.timeout)
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('response'):
                    self.is_available = True
                    logger.info("✅ Ollama服务初始化成功")
                    return True

            logger.warning("❌ Ollama测试生成失败")
            self.is_available = False
            return False

        except Exception as e:
            logger.warning(f"⚠️ Ollama初始化失败: {e}")
            self.is_available = False
            return False

    async def generate_text(self, prompt: str, max_length: int = 1024) -> Optional[str]:
        """使用Ollama生成文本"""
        if not self.is_available:
            return None

        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": min(max_length, self.max_tokens)
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '')
                if generated_text:
                    return generated_text.strip()

            logger.error(f"Ollama API调用失败: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"Ollama文本生成失败: {e}")
            return None

    async def test_connection(self) -> tuple[bool, str, List[str]]:
        """测试Ollama连接"""
        try:
            # 获取可用模型列表
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code != 200:
                return False, f"无法连接到Ollama服务 (状态码: {response.status_code})", []

            models_data = response.json()
            available_models = [model['name'] for model in models_data.get('models', [])]

            if not available_models:
                return False, "Ollama服务运行正常，但没有可用模型", []

            # 检查指定模型是否可用
            if self.model_name not in available_models:
                return False, f"指定模型 {self.model_name} 不可用", available_models

            # 测试模型生成
            test_payload = {
                "model": self.model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "temperature": self.temperature,
                    "num_predict": 20
                }
            }

            response = requests.post(
                f"{self.base_url}/api/generate",
                json=test_payload,
                timeout=min(30, self.timeout)
            )

            if response.status_code == 200:
                result = response.json()
                if result.get('response'):
                    self.is_available = True
                    return True, "连接测试成功", available_models
                else:
                    self.is_available = False
                    return False, "模型响应为空", available_models
            else:
                self.is_available = False
                return False, f"模型测试失败 (状态码: {response.status_code})", available_models

        except requests.exceptions.Timeout:
            self.is_available = False
            return False, "连接超时，请检查Ollama服务是否运行", []
        except requests.exceptions.ConnectionError:
            self.is_available = False
            return False, "无法连接到Ollama服务，请检查服务地址和端口", []
        except Exception as e:
            self.is_available = False
            return False, f"连接测试失败: {str(e)}", []

class GeminiProvider(LLMProvider):
    """Google Gemini API提供商"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://generativelanguage.googleapis.com")
        self.model_name = config.get("model_name", "gemini-1.5-flash")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2048)
        self.timeout = config.get("timeout", 60)

    def get_provider_name(self) -> str:
        return "Gemini"

    async def initialize(self) -> bool:
        """初始化Gemini API连接"""
        try:
            logger.info(f"正在初始化Gemini API服务，模型: {self.model_name}")

            if not self.api_key:
                logger.warning("Gemini API密钥未配置")
                self.is_available = False
                return False

            # 测试API连接
            success, message, _ = await self.test_connection()
            if success:
                self.is_available = True
                logger.info("✅ Gemini API服务初始化成功")
                return True
            else:
                logger.warning(f"❌ Gemini API初始化失败: {message}")
                self.is_available = False
                return False

        except Exception as e:
            logger.warning(f"⚠️ Gemini API初始化失败: {e}")
            self.is_available = False
            return False

    async def generate_text(self, prompt: str, max_length: int = 1024) -> Optional[str]:
        """使用Gemini API生成文本"""
        if not self.is_available or not self.api_key:
            return None

        try:
            headers = {
                "Content-Type": "application/json"
            }

            # Gemini API使用不同的请求格式
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": self.temperature,
                    "maxOutputTokens": min(max_length, self.max_tokens),
                    "candidateCount": 1
                }
            }

            # Gemini API URL格式
            url = f"{self.base_url}/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                candidates = result.get('candidates', [])
                if candidates and len(candidates) > 0:
                    content = candidates[0].get('content', {})
                    parts = content.get('parts', [])
                    if parts and len(parts) > 0:
                        text = parts[0].get('text', '')
                        if text:
                            return text.strip()

            logger.error(f"Gemini API调用失败: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"Gemini API文本生成失败: {e}")
            return None

    async def test_connection(self) -> tuple[bool, str, List[str]]:
        """测试Gemini API连接"""
        try:
            if not self.api_key:
                return False, "API密钥未配置", []

            headers = {
                "Content-Type": "application/json"
            }

            # 使用简单的测试请求
            test_payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": "Hello"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 10,
                    "candidateCount": 1
                }
            }

            url = f"{self.base_url}/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"

            response = requests.post(
                url,
                headers=headers,
                json=test_payload,
                timeout=min(30, self.timeout)
            )

            if response.status_code == 200:
                result = response.json()
                candidates = result.get('candidates', [])
                if candidates and len(candidates) > 0:
                    self.is_available = True
                    # Gemini常用模型列表
                    available_models = [
                        self.model_name,
                        "gemini-1.5-flash",
                        "gemini-1.5-pro",
                        "gemini-1.0-pro"
                    ]
                    return True, "连接测试成功", available_models
                else:
                    self.is_available = False
                    return False, "API响应格式异常", []
            elif response.status_code == 400:
                self.is_available = False
                error_detail = response.json().get('error', {}).get('message', response.text)
                return False, f"请求格式错误: {error_detail}", []
            elif response.status_code == 403:
                self.is_available = False
                return False, "API密钥无效或权限不足", []
            elif response.status_code == 429:
                self.is_available = False
                return False, "API调用频率超限，请稍后重试", []
            else:
                self.is_available = False
                return False, f"API调用失败 (状态码: {response.status_code}): {response.text}", []

        except requests.exceptions.Timeout:
            self.is_available = False
            return False, "连接超时，请检查网络连接", []
        except requests.exceptions.ConnectionError:
            self.is_available = False
            return False, "无法连接到Gemini API服务", []
        except Exception as e:
            self.is_available = False
            return False, f"连接测试失败: {str(e)}", []

class DeepSeekProvider(LLMProvider):
    """DeepSeek API提供商"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.base_url = config.get("base_url", "https://api.deepseek.com")
        self.model_name = config.get("model_name", "deepseek-chat")
        self.temperature = config.get("temperature", 0.3)
        self.max_tokens = config.get("max_tokens", 2048)
        self.timeout = config.get("timeout", 60)

    def get_provider_name(self) -> str:
        return "DeepSeek"

    async def initialize(self) -> bool:
        """初始化DeepSeek API连接"""
        try:
            logger.info(f"正在初始化DeepSeek API服务，模型: {self.model_name}")

            if not self.api_key:
                logger.warning("DeepSeek API密钥未配置")
                self.is_available = False
                return False

            # 测试API连接
            success, message, _ = await self.test_connection()
            if success:
                self.is_available = True
                logger.info("✅ DeepSeek API服务初始化成功")
                return True
            else:
                logger.warning(f"❌ DeepSeek API初始化失败: {message}")
                self.is_available = False
                return False

        except Exception as e:
            logger.warning(f"⚠️ DeepSeek API初始化失败: {e}")
            self.is_available = False
            return False

    async def generate_text(self, prompt: str, max_length: int = 1024) -> Optional[str]:
        """使用DeepSeek API生成文本"""
        if not self.is_available or not self.api_key:
            return None

        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": min(max_length, self.max_tokens),
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )

            if response.status_code == 200:
                result = response.json()
                choices = result.get('choices', [])
                if choices and len(choices) > 0:
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    if content:
                        return content.strip()

            logger.error(f"DeepSeek API调用失败: {response.status_code} - {response.text}")
            return None

        except Exception as e:
            logger.error(f"DeepSeek API文本生成失败: {e}")
            return None

    async def test_connection(self) -> tuple[bool, str, List[str]]:
        """测试DeepSeek API连接"""
        try:
            if not self.api_key:
                return False, "API密钥未配置", []

            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }

            # 使用简单的测试请求
            test_payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "temperature": 0.1,
                "max_tokens": 10,
                "stream": False
            }

            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=test_payload,
                timeout=min(30, self.timeout)
            )

            if response.status_code == 200:
                result = response.json()
                choices = result.get('choices', [])
                if choices and len(choices) > 0:
                    self.is_available = True
                    # DeepSeek API不提供模型列表，返回当前配置的模型
                    available_models = [self.model_name, "deepseek-chat", "deepseek-reasoner"]
                    return True, "连接测试成功", available_models
                else:
                    self.is_available = False
                    return False, "API响应格式异常", []
            elif response.status_code == 401:
                self.is_available = False
                return False, "API密钥无效或已过期", []
            elif response.status_code == 429:
                self.is_available = False
                return False, "API调用频率超限，请稍后重试", []
            else:
                self.is_available = False
                return False, f"API调用失败 (状态码: {response.status_code}): {response.text}", []

        except requests.exceptions.Timeout:
            self.is_available = False
            return False, "连接超时，请检查网络连接", []
        except requests.exceptions.ConnectionError:
            self.is_available = False
            return False, "无法连接到DeepSeek API服务", []
        except Exception as e:
            self.is_available = False
            return False, f"连接测试失败: {str(e)}", []

class LLMConfig:
    """LLM配置管理类"""

    def __init__(self):
        self.config_file = Path("llm_config.json")
        self.default_config = {
            "provider": "ollama",  # 默认使用ollama
            "enabled": True,
            "ollama": {
                "base_url": "http://localhost:11434",
                "model_name": "deepseek-r1:1.5b",
                "temperature": 0.3,
                "max_tokens": 2048,
                "timeout": 60
            },
            "deepseek": {
                "api_key": "",
                "base_url": "https://api.deepseek.com",
                "model_name": "deepseek-chat",
                "temperature": 0.3,
                "max_tokens": 2048,
                "timeout": 60
            },
            "gemini": {
                "api_key": "",
                "base_url": "https://generativelanguage.googleapis.com",
                "model_name": "gemini-1.5-flash",
                "temperature": 0.3,
                "max_tokens": 2048,
                "timeout": 60
            }
        }

    def load_config(self) -> Dict[str, Any]:
        """加载LLM配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    # 检查是否是旧格式配置，如果是则转换为新格式
                    if "provider" not in config and "base_url" in config:
                        # 旧格式转换为新格式
                        old_config = config.copy()
                        config = self.default_config.copy()
                        config["provider"] = "ollama"
                        config["ollama"].update({
                            "base_url": old_config.get("base_url", "http://localhost:11434"),
                            "model_name": old_config.get("model_name", "deepseek-r1:1.5b"),
                            "temperature": old_config.get("temperature", 0.3),
                            "max_tokens": old_config.get("max_tokens", 2048),
                            "timeout": old_config.get("timeout", 60)
                        })
                        config["enabled"] = old_config.get("enabled", True)
                        # 保存转换后的配置
                        self.save_config(config)
                        logger.info("已将旧格式配置转换为新格式")
                    else:
                        # 合并默认配置，确保所有必需字段存在
                        merged_config = self.default_config.copy()
                        self._deep_update(merged_config, config)
                        config = merged_config
                    return config
        except Exception as e:
            logger.warning(f"加载LLM配置失败: {e}")

        return self.default_config.copy()

    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]):
        """深度更新字典"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value

    def save_config(self, config: Dict[str, Any]) -> bool:
        """保存LLM配置"""
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            logger.info(f"LLM配置已保存到: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"保存LLM配置失败: {e}")
            return False

    def validate_config(self, config: Dict[str, Any]) -> tuple[bool, str]:
        """验证配置有效性"""
        # 检查基本字段
        if "provider" not in config:
            return False, "缺少必需字段: provider"

        provider = config["provider"]
        if provider not in ["ollama", "deepseek", "gemini"]:
            return False, f"不支持的提供商: {provider}"

        # 检查提供商特定配置
        if provider not in config:
            return False, f"缺少提供商配置: {provider}"

        provider_config = config[provider]

        if provider == "ollama":
            required_fields = ["base_url", "model_name"]
            for field in required_fields:
                if field not in provider_config or not provider_config[field]:
                    return False, f"Ollama配置缺少必需字段: {field}"
        elif provider == "deepseek":
            required_fields = ["api_key", "model_name"]
            for field in required_fields:
                if field not in provider_config or not provider_config[field]:
                    return False, f"DeepSeek配置缺少必需字段: {field}"
        elif provider == "gemini":
            required_fields = ["api_key", "model_name"]
            for field in required_fields:
                if field not in provider_config or not provider_config[field]:
                    return False, f"Gemini配置缺少必需字段: {field}"

        # 验证URL格式
        base_url = provider_config.get("base_url", "")
        if base_url and not base_url.startswith(("http://", "https://")):
            return False, "服务地址必须以 http:// 或 https:// 开头"

        # 验证数值参数
        try:
            if "temperature" in provider_config:
                temp = float(provider_config["temperature"])
                if not 0 <= temp <= 2:
                    return False, "温度参数必须在 0-2 之间"

            if "max_tokens" in provider_config:
                max_tokens = int(provider_config["max_tokens"])
                if max_tokens <= 0:
                    return False, "最大令牌数必须大于 0"

            if "timeout" in provider_config:
                timeout = int(provider_config["timeout"])
                if timeout <= 0:
                    return False, "超时时间必须大于 0"

        except (ValueError, TypeError):
            return False, "数值参数格式错误"

        return True, "配置验证通过"

class LLMService:
    """
    LLM服务管理器，提供智能分析和文本生成功能
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化LLM服务

        Args:
            config: LLM配置字典，如果为None则从配置文件加载
        """
        self.config_manager = LLMConfig()

        if config is None:
            self.config = self.config_manager.load_config()
        else:
            self.config = config

        self.enabled = self.config.get("enabled", True)
        self.provider_name = self.config.get("provider", "ollama")

        # 创建对应的提供商实例
        self.provider = self._create_provider()
        self.is_available = False

    def _create_provider(self) -> LLMProvider:
        """创建LLM提供商实例"""
        if self.provider_name == "ollama":
            provider_config = self.config.get("ollama", {})
            return OllamaProvider(provider_config)
        elif self.provider_name == "deepseek":
            provider_config = self.config.get("deepseek", {})
            return DeepSeekProvider(provider_config)
        elif self.provider_name == "gemini":
            provider_config = self.config.get("gemini", {})
            return GeminiProvider(provider_config)
        else:
            # 默认使用Ollama
            logger.warning(f"未知的提供商: {self.provider_name}，使用默认Ollama")
            provider_config = self.config.get("ollama", {})
            return OllamaProvider(provider_config)
        
    async def initialize(self) -> bool:
        """
        初始化LLM连接

        Returns:
            bool: 是否初始化成功
        """
        # 检查是否启用LLM服务
        if not self.enabled:
            logger.info("LLM服务已禁用，将使用基于规则的智能生成")
            self.is_available = False
            return True

        try:
            # 使用提供商进行初始化
            success = await self.provider.initialize()
            if success:
                self.is_available = True
                logger.info(f"✅ {self.provider.get_provider_name()} LLM服务初始化成功")
                return True
            else:
                logger.info("将使用增强的基于规则的智能生成")
                self.is_available = False
                return True  # 仍然返回True，使用备用方法

        except Exception as e:
            logger.warning(f"⚠️ LLM初始化失败: {e}")
            logger.info("将使用增强的基于规则的智能生成")
            self.is_available = False
            return True  # 仍然返回True，使用备用方法

    async def _generate_text(self, prompt: str, max_length: int = 1024) -> Optional[str]:
        """
        生成文本的基础方法

        Args:
            prompt: 输入提示
            max_length: 最大输出长度

        Returns:
            生成的文本或None
        """
        if not self.is_available:
            logger.warning("LLM服务不可用，使用备用方法")
            return None

        try:
            # 使用提供商生成文本
            generated_text = await self.provider.generate_text(prompt, max_length)
            if generated_text:
                # 清理思考过程标签
                cleaned_text = self._clean_thinking_process(generated_text)
                return cleaned_text.strip()
            return None

        except Exception as e:
            logger.error(f"LLM文本生成失败: {e}")
            return None

    def _clean_thinking_process(self, text: str, show_thinking: bool = False) -> str:
        """
        清理LLM输出中的思考过程标签

        Args:
            text: 原始LLM输出
            show_thinking: 是否保留思考过程

        Returns:
            清理后的文本
        """
        if not text:
            return text

        if not show_thinking:
            # 移除 <think>...</think> 标签及其内容
            import re
            # 匹配 <think> 到 </think> 之间的所有内容（包括换行）
            pattern = r'<think>.*?</think>'
            cleaned = re.sub(pattern, '', text, flags=re.DOTALL)

            # 清理多余的空行
            lines = cleaned.split('\n')
            cleaned_lines = []
            prev_empty = False

            for line in lines:
                line = line.strip()
                if line:
                    cleaned_lines.append(line)
                    prev_empty = False
                elif not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True

            # 移除开头和结尾的空行
            while cleaned_lines and not cleaned_lines[0]:
                cleaned_lines.pop(0)
            while cleaned_lines and not cleaned_lines[-1]:
                cleaned_lines.pop()

            return '\n'.join(cleaned_lines)
        else:
            # 保留思考过程，但用特殊格式标记
            import re
            def replace_think_tags(match):
                content = match.group(1)
                return f"\n💭 **思考过程**:\n```\n{content.strip()}\n```\n"

            pattern = r'<think>(.*?)</think>'
            return re.sub(pattern, replace_think_tags, text, flags=re.DOTALL)
    
    async def generate_analysis_interpretation(
        self,
        exposure_gene: str,
        outcome_trait: str,
        mr_results: Dict[str, Any],
        language: str = "en",
        show_thinking: bool = False
    ) -> str:
        """
        生成MR分析结果的智能解释
        
        Args:
            exposure_gene: 暴露基因
            outcome_trait: 结局性状
            mr_results: MR分析结果
            language: 语言 ("zh" 或 "en")
            
        Returns:
            智能生成的分析解释
        """
        
        # 构建提示模板
        if language == "zh":
            prompt_template = """
你是一位专业的生物统计学家和遗传流行病学专家。请基于以下孟德尔随机化(MR)分析结果，生成一份专业且易懂的分析报告。

## 分析信息
- 暴露变量: {exposure_gene} 基因表达
- 结局变量: {outcome_trait}
- 分析方法: 孟德尔随机化

## MR分析结果
{mr_results_summary}

## 请提供以下内容：

### 1. 主要发现
简明扼要地总结因果关系的主要发现。

### 2. 统计学解释
解释各种MR方法的结果及其统计显著性。

### 3. 生物学意义
从生物学角度解释这种因果关系的可能机制。

### 4. 临床意义
讨论这一发现对临床实践和公共卫生的潜在影响。

### 5. 研究局限性
指出分析的局限性和需要注意的问题。

请用专业但易懂的语言撰写，适合科研人员和临床医生阅读。
"""
        else:
            prompt_template = """
You are a professional biostatistician and genetic epidemiologist. Please generate a professional and accessible analysis report based on the following Mendelian Randomization (MR) analysis results.

## Analysis Information
- Exposure: {exposure_gene} gene expression
- Outcome: {outcome_trait}
- Method: Mendelian Randomization

## MR Analysis Results
{mr_results_summary}

## Please provide the following content:

### 1. Key Findings
Summarize the main findings regarding the causal relationship.

### 2. Statistical Interpretation
Explain the results from different MR methods and their statistical significance.

### 3. Biological Significance
Interpret the potential biological mechanisms underlying this causal relationship.

### 4. Clinical Implications
Discuss the potential impact of these findings on clinical practice and public health.

### 5. Study Limitations
Point out the limitations of the analysis and important considerations.

Please write in professional but accessible language, suitable for researchers and clinicians.
"""
        
        # 准备MR结果摘要
        mr_summary = self._format_mr_results(mr_results, language)
        
        # 构建完整提示
        prompt = prompt_template.format(
            exposure_gene=exposure_gene,
            outcome_trait=outcome_trait,
            mr_results_summary=mr_summary
        )
        
        # 生成解释
        interpretation = await self._generate_text(prompt, max_length=2048)

        if interpretation:
            # 清理思考过程
            cleaned_interpretation = self._clean_thinking_process(interpretation, show_thinking)
            return cleaned_interpretation
        else:
            # 备用方案：基于规则的解释
            return self._generate_fallback_interpretation(exposure_gene, outcome_trait, mr_results, language)
    
    def _format_mr_results(self, mr_results: Dict[str, Any], language: str = "en") -> str:
        """
        格式化MR结果为文本摘要
        """
        if not mr_results or "results" not in mr_results:
            return "无可用的MR分析结果" if language == "zh" else "No MR analysis results available"
        
        summary_parts = []
        
        for result in mr_results["results"]:
            method = result.get("method", "Unknown")
            estimate = result.get("estimate", 0)
            p_value = result.get("p_value", 1)
            ci_lower = result.get("ci_lower", 0)
            ci_upper = result.get("ci_upper", 0)
            
            if language == "zh":
                summary_parts.append(
                    f"- {method}: 因果效应 = {estimate:.3f} "
                    f"(95% CI: {ci_lower:.3f} to {ci_upper:.3f}), P = {p_value:.2e}"
                )
            else:
                summary_parts.append(
                    f"- {method}: Causal effect = {estimate:.3f} "
                    f"(95% CI: {ci_lower:.3f} to {ci_upper:.3f}), P = {p_value:.2e}"
                )
        
        return "\n".join(summary_parts)
    
    def _generate_fallback_interpretation(
        self,
        exposure_gene: str,
        outcome_trait: str,
        mr_results: Dict[str, Any],
        language: str = "en"
    ) -> str:
        """
        增强的基于规则的解释生成
        """
        # 分析MR结果
        mr_summary = self._analyze_mr_results(mr_results)

        # 使用国际化文本构建报告
        report_parts = [
            f"## {get_text('mr_analysis_report', language)}",
            "",
            f"### {get_text('key_findings', language)}",
            get_text('mr_study_description', language).format(exposure_gene, outcome_trait),
            "",
            mr_summary[language],
            "",
            f"### {get_text('methodological_advantages', language)}",
            f"- {get_text('genetic_variants_iv', language)}",
            f"- {get_text('multiple_mr_validation', language)}",
            f"- {get_text('sensitivity_analysis', language)}",
            "",
            f"### {get_text('biological_significance', language)}",
            get_text('expression_mechanisms', language).format(exposure_gene, outcome_trait),
            get_text('direct_regulation', language),
            get_text('protein_function', language),
            get_text('gene_networks', language),
            "",
            f"### {get_text('clinical_translation', language)}",
            get_text('drug_target_identification', language).format(exposure_gene, outcome_trait),
            get_text('personalized_medicine', language),
            get_text('prevention_strategies', language),
            "",
            f"### {get_text('study_limitations', language)}",
            get_text('population_specific', language),
            get_text('larger_samples', language),
            get_text('functional_validation', language),
            "",
            f"### {get_text('future_research', language)}",
            get_text('validate_cohorts', language),
            get_text('functional_experiments', language),
            get_text('clinical_feasibility', language),
            get_text('drug_validation', language),
            "",
            get_text('enhanced_system_note', language)
        ]

        return "\n".join(report_parts)

    def _analyze_mr_results(self, mr_results: Dict[str, Any]) -> Dict[str, str]:
        """分析MR结果并生成智能总结"""
        if not mr_results or "results" not in mr_results:
            return {
                "zh": get_text("no_mr_results", "zh"),
                "en": get_text("no_mr_results", "en")
            }

        results = mr_results["results"]

        # 找到IVW结果
        ivw_result = None
        for result in results:
            if "Inverse Variance" in result.get("method", ""):
                ivw_result = result
                break

        if not ivw_result:
            ivw_result = results[0] if results else None

        if not ivw_result:
            return {
                "zh": get_text("no_valid_results", "zh"),
                "en": get_text("no_valid_results", "en")
            }

        estimate = ivw_result.get("estimate", 0)
        p_value = ivw_result.get("p_value", 1)
        ci_lower = ivw_result.get("ci_lower", 0)
        ci_upper = ivw_result.get("ci_upper", 0)

        # 判断显著性和效应方向
        is_significant = p_value < 0.05
        effect_direction = "正向" if estimate > 0 else "负向"
        effect_direction_en = "positive" if estimate > 0 else "negative"

        if is_significant:
            zh_parts = [
                get_text("significant_causal_effect", "zh"),
                get_text("causal_effect_estimate", "zh").format(estimate, ci_lower, ci_upper),
                get_text("statistical_significance", "zh").format(p_value),
                get_text("effect_assessment_significant", "zh"),
                "",
                get_text("conclusion_significant", "zh")
            ]

            en_parts = [
                get_text("significant_causal_effect", "en"),
                get_text("causal_effect_estimate", "en").format(estimate, ci_lower, ci_upper),
                get_text("statistical_significance", "en").format(p_value),
                get_text("effect_assessment_significant", "en"),
                "",
                get_text("conclusion_significant", "en")
            ]
        else:
            zh_parts = [
                get_text("no_significant_effect_detected", "zh"),
                get_text("causal_effect_estimate", "zh").format(estimate, ci_lower, ci_upper),
                get_text("statistical_significance", "zh").format(p_value),
                get_text("effect_assessment_nonsignificant", "zh"),
                "",
                get_text("conclusion_nonsignificant", "zh")
            ]

            en_parts = [
                get_text("no_significant_effect_detected", "en"),
                get_text("causal_effect_estimate", "en").format(estimate, ci_lower, ci_upper),
                get_text("statistical_significance", "en").format(p_value),
                get_text("effect_assessment_nonsignificant", "en"),
                "",
                get_text("conclusion_nonsignificant", "en")
            ]

        return {
            "zh": "\n".join(zh_parts),
            "en": "\n".join(en_parts)
        }

    def _format_gene_data(self, gene_info: Dict[str, Any], language: str = "en") -> str:
        """
        格式化基因数据为文本摘要
        """
        try:
            summary_parts = []

            if language == "zh":
                if gene_info.get("protein_name"):
                    summary_parts.append(f"蛋白质名称: {gene_info['protein_name']}")

                if gene_info.get("function"):
                    function_text = gene_info["function"]
                    if len(function_text) > 300:
                        function_text = function_text[:300] + "..."
                    summary_parts.append(f"分子功能: {function_text}")

                if gene_info.get("subcellular_location"):
                    summary_parts.append(f"亚细胞定位: {gene_info['subcellular_location']}")

                if gene_info.get("description"):
                    desc_text = gene_info["description"]
                    if len(desc_text) > 200:
                        desc_text = desc_text[:200] + "..."
                    summary_parts.append(f"基因描述: {desc_text}")

            else:  # English
                if gene_info.get("protein_name"):
                    summary_parts.append(f"Protein name: {gene_info['protein_name']}")

                if gene_info.get("function"):
                    function_text = gene_info["function"]
                    if len(function_text) > 300:
                        function_text = function_text[:300] + "..."
                    summary_parts.append(f"Molecular function: {function_text}")

                if gene_info.get("subcellular_location"):
                    summary_parts.append(f"Subcellular location: {gene_info['subcellular_location']}")

                if gene_info.get("description"):
                    desc_text = gene_info["description"]
                    if len(desc_text) > 200:
                        desc_text = desc_text[:200] + "..."
                    summary_parts.append(f"Gene description: {desc_text}")

            return "\n".join(summary_parts) if summary_parts else "No detailed information available"

        except Exception as e:
            logger.error(f"基因数据格式化失败: {e}")
            return "Data formatting error"

    def _format_drug_data(self, drug_targets: List[Dict[str, Any]], language: str = "en") -> str:
        """
        格式化药物数据为文本摘要
        """
        try:
            if not drug_targets:
                if language == "zh":
                    return "目前没有已知药物直接靶向该基因"
                else:
                    return "No known drugs currently target this gene"

            summary_parts = []

            if language == "zh":
                summary_parts.append(f"发现 {len(drug_targets)} 个相关药物靶点:")

                for i, drug in enumerate(drug_targets[:5]):  # 最多显示5个
                    drug_name = drug.get("drug_name", drug.get("name", f"药物 {i+1}"))
                    drug_type = drug.get("drug_type", drug.get("type", ""))
                    mechanism = drug.get("mechanism", drug.get("action_type", ""))

                    drug_info = f"- {drug_name}"
                    if drug_type:
                        drug_info += f" ({drug_type})"
                    if mechanism:
                        drug_info += f" - {mechanism}"

                    summary_parts.append(drug_info)

                if len(drug_targets) > 5:
                    summary_parts.append(f"- 以及其他 {len(drug_targets) - 5} 个药物...")

            else:  # English
                summary_parts.append(f"Found {len(drug_targets)} related drug targets:")

                for i, drug in enumerate(drug_targets[:5]):
                    drug_name = drug.get("drug_name", drug.get("name", f"Drug {i+1}"))
                    drug_type = drug.get("drug_type", drug.get("type", ""))
                    mechanism = drug.get("mechanism", drug.get("action_type", ""))

                    drug_info = f"- {drug_name}"
                    if drug_type:
                        drug_info += f" ({drug_type})"
                    if mechanism:
                        drug_info += f" - {mechanism}"

                    summary_parts.append(drug_info)

                if len(drug_targets) > 5:
                    summary_parts.append(f"- And {len(drug_targets) - 5} other drugs...")

            return "\n".join(summary_parts)

        except Exception as e:
            logger.error(f"药物数据格式化失败: {e}")
            if language == "zh":
                return "数据格式化错误"
            else:
                return "Data formatting error"

    async def generate_recommendations(
        self,
        analysis_result: Dict[str, Any],
        language: str = "en",
        show_thinking: bool = False
    ) -> List[str]:
        """
        生成智能化的研究建议
        """
        if language == "zh":
            prompt = f"""
基于以下孟德尔随机化分析结果，请生成5-7条具体的后续研究建议：

分析结果摘要：
{analysis_result.get('summary', {})}

请提供具体、可操作的建议，包括：
1. 验证实验设计
2. 扩展研究方向
3. 临床转化建议
4. 方法学改进
5. 数据收集建议

每条建议应该简洁明确，适合研究人员参考。
"""
        else:
            prompt = f"""
Based on the following Mendelian randomization analysis results, please generate 5-7 specific follow-up research recommendations:

Analysis summary:
{analysis_result.get('summary', {})}

Please provide specific, actionable recommendations including:
1. Validation experiment design
2. Extended research directions
3. Clinical translation suggestions
4. Methodological improvements
5. Data collection recommendations

Each recommendation should be concise and clear, suitable for researchers' reference.
"""
        
        recommendations_text = await self._generate_text(prompt, max_length=1024)

        if recommendations_text:
            # 清理思考过程
            cleaned_text = self._clean_thinking_process(recommendations_text, show_thinking)

            # 解析生成的建议
            recommendations = []
            lines = cleaned_text.split('\n')
            for line in lines:
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or re.match(r'^\d+\.', line)):
                    # 清理格式
                    clean_line = re.sub(r'^[-•\d\.]\s*', '', line)
                    if clean_line:
                        recommendations.append(clean_line)
            
            return recommendations[:7]  # 最多返回7条建议
        else:
            # 备用建议
            if language == "zh":
                return [
                    "在独立队列中验证因果关系",
                    "进行功能实验验证分子机制",
                    "评估临床应用的可行性",
                    "扩大样本量提高统计功效",
                    "考虑人群分层分析"
                ]
            else:
                return [
                    "Validate causal relationship in independent cohorts",
                    "Conduct functional experiments to verify molecular mechanisms",
                    "Assess feasibility of clinical applications",
                    "Increase sample size to improve statistical power",
                    "Consider population stratification analysis"
                ]

    async def generate_gene_interpretation(
        self,
        gene_symbol: str,
        gene_info: Dict[str, Any],
        language: str = "en",
        show_thinking: bool = False
    ) -> str:
        """
        生成基因功能的智能解释

        Args:
            gene_symbol: 基因符号
            gene_info: 基因信息数据
            language: 语言 ("zh" 或 "en")
            show_thinking: 是否显示思考过程

        Returns:
            智能生成的基因功能解释
        """

        # 构建提示模板
        if language == "zh":
            prompt_template = """
你是一位专业的分子生物学家和基因组学专家。请基于以下基因信息，生成一份专业且易懂的基因功能分析报告。

## 基因信息
- 基因符号: {gene_symbol}
- 基因数据: {gene_data}

## 请提供以下内容：

### 1. 基因概述
简明扼要地介绍该基因的基本信息和主要功能。

### 2. 分子功能
详细解释该基因编码蛋白质的分子功能和生物学作用。

### 3. 生物学通路
描述该基因参与的主要生物学通路和调控网络。

### 4. 疾病关联
总结该基因与人类疾病的关联性和临床意义。

### 5. 研究价值
评估该基因在科学研究和临床应用中的价值。

请用专业但易懂的语言撰写，适合科研人员和临床医生阅读。
"""
        else:
            prompt_template = """
You are a professional molecular biologist and genomics expert. Please generate a professional and accessible gene function analysis report based on the following gene information.

## Gene Information
- Gene Symbol: {gene_symbol}
- Gene Data: {gene_data}

## Please provide the following content:

### 1. Gene Overview
Provide a concise introduction to the basic information and main functions of this gene.

### 2. Molecular Function
Explain in detail the molecular function and biological role of the protein encoded by this gene.

### 3. Biological Pathways
Describe the main biological pathways and regulatory networks this gene participates in.

### 4. Disease Associations
Summarize the associations between this gene and human diseases, and its clinical significance.

### 5. Research Value
Evaluate the value of this gene in scientific research and clinical applications.

Please write in professional but accessible language, suitable for researchers and clinicians.
"""

        # 准备基因数据摘要
        gene_data_summary = self._format_gene_data(gene_info, language)

        # 构建完整提示
        prompt = prompt_template.format(
            gene_symbol=gene_symbol,
            gene_data=gene_data_summary
        )

        # 生成解释
        interpretation = await self._generate_text(prompt, max_length=1536)

        if interpretation:
            # 清理思考过程
            cleaned_interpretation = self._clean_thinking_process(interpretation, show_thinking)
            return cleaned_interpretation
        else:
            # 备用方案：返回空字符串，让调用方使用基于规则的方法
            return ""

    async def generate_drug_interpretation(
        self,
        gene_symbol: str,
        drug_targets: List[Dict[str, Any]],
        language: str = "en",
        show_thinking: bool = False
    ) -> str:
        """
        生成药物靶点的智能解释

        Args:
            gene_symbol: 基因符号
            drug_targets: 药物靶点数据
            language: 语言 ("zh" 或 "en")
            show_thinking: 是否显示思考过程

        Returns:
            智能生成的药物靶点解释
        """

        # 构建提示模板
        if language == "zh":
            prompt_template = """
你是一位专业的药物化学家和临床药理学专家。请基于以下药物靶点信息，生成一份专业且易懂的治疗意义分析报告。

## 靶点信息
- 靶点基因: {gene_symbol}
- 相关药物: {drug_data}

## 请提供以下内容：

### 1. 靶点概述
简明扼要地介绍该基因作为药物靶点的基本特征。

### 2. 现有药物
总结目前已知靶向该基因的药物及其作用机制。

### 3. 治疗应用
描述这些药物在临床治疗中的应用和效果。

### 4. 开发前景
评估该靶点在新药开发中的潜力和机会。

### 5. 临床意义
分析该靶点在精准医疗和个体化治疗中的价值。

请用专业但易懂的语言撰写，适合科研人员和临床医生阅读。
"""
        else:
            prompt_template = """
You are a professional medicinal chemist and clinical pharmacology expert. Please generate a professional and accessible therapeutic significance analysis report based on the following drug target information.

## Target Information
- Target Gene: {gene_symbol}
- Related Drugs: {drug_data}

## Please provide the following content:

### 1. Target Overview
Provide a concise introduction to the basic characteristics of this gene as a drug target.

### 2. Existing Drugs
Summarize currently known drugs targeting this gene and their mechanisms of action.

### 3. Therapeutic Applications
Describe the clinical applications and effects of these drugs in treatment.

### 4. Development Prospects
Evaluate the potential and opportunities of this target in new drug development.

### 5. Clinical Significance
Analyze the value of this target in precision medicine and personalized therapy.

Please write in professional but accessible language, suitable for researchers and clinicians.
"""

        # 准备药物数据摘要
        drug_data_summary = self._format_drug_data(drug_targets, language)

        # 构建完整提示
        prompt = prompt_template.format(
            gene_symbol=gene_symbol,
            drug_data=drug_data_summary
        )

        # 生成解释
        interpretation = await self._generate_text(prompt, max_length=1536)

        if interpretation:
            # 清理思考过程
            cleaned_interpretation = self._clean_thinking_process(interpretation, show_thinking)
            return cleaned_interpretation
        else:
            # 备用方案：返回空字符串，让调用方使用基于规则的方法
            return ""

    def update_config(self, new_config: Dict[str, Any]) -> tuple[bool, str]:
        """
        更新LLM配置

        Args:
            new_config: 新的配置字典

        Returns:
            tuple[bool, str]: (是否成功, 消息)
        """
        # 验证配置
        is_valid, message = self.config_manager.validate_config(new_config)
        if not is_valid:
            return False, message

        # 保存配置
        if not self.config_manager.save_config(new_config):
            return False, "保存配置失败"

        # 检查是否有影响连接的关键参数变更
        old_provider = self.provider_name
        old_enabled = self.enabled

        new_provider = new_config.get("provider", "ollama")
        new_enabled = new_config.get("enabled", True)

        # 检查关键连接参数是否发生变化
        connection_params_changed = (
            old_provider != new_provider or
            old_enabled != new_enabled
        )

        # 更新当前配置
        self.config = new_config
        self.provider_name = new_provider
        self.enabled = new_enabled

        # 重新创建提供商实例
        if connection_params_changed:
            self.provider = self._create_provider()
            self.is_available = False

        return True, "配置更新成功"

    async def test_connection(self, skip_model_check: bool = False) -> tuple[bool, str, List[str]]:
        """
        测试LLM连接

        Args:
            skip_model_check: 是否跳过指定模型检查（用于快速设置）

        Returns:
            tuple[bool, str, List[str]]: (是否成功, 消息, 可用模型列表)
        """
        if not self.enabled:
            self.is_available = False
            return False, "LLM服务已禁用", []

        try:
            # 使用提供商测试连接
            success, message, models = await self.provider.test_connection()
            if success:
                self.is_available = True
            else:
                self.is_available = False
            return success, message, models

        except Exception as e:
            self.is_available = False
            return False, f"连接测试失败: {str(e)}", []

    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self.config.copy()

    def get_status(self) -> Dict[str, Any]:
        """
        获取LLM服务状态
        """
        provider_config = self.config.get(self.provider_name, {})
        return {
            "available": self.is_available,
            "enabled": self.enabled,
            "provider": self.provider_name,
            "provider_name": self.provider.get_provider_name() if self.provider else "Unknown",
            "model": provider_config.get("model_name", ""),
            "base_url": provider_config.get("base_url", ""),
            "temperature": provider_config.get("temperature", 0.3),
            "max_tokens": provider_config.get("max_tokens", 2048),
            "timeout": provider_config.get("timeout", 60)
        }
