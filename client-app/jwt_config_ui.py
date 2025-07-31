# client-app/jwt_config_ui.py
"""
OpenGWAS JWT令牌配置用户界面组件
"""

import streamlit as st
import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import asyncio
import httpx

logger = logging.getLogger(__name__)

class JWTConfigManager:
    """JWT配置管理器"""
    
    def __init__(self):
        self.config_file = Path("opengwas_config.json")
        
    def load_config(self) -> Optional[Dict[str, Any]]:
        """加载JWT配置"""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                    return config
        except Exception as e:
            logger.warning(f"加载JWT配置失败: {e}")
        return None
    
    def save_config(self, jwt_token: str, description: str = "") -> bool:
        """保存JWT配置"""
        try:
            config = {
                "jwt_token": jwt_token,
                "description": description or "OpenGWAS API JWT token for Causal-MCP",
                "configured_at": "2025-01-11",
                "configured_by": "user"
            }
            
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
            
            logger.info(f"JWT配置已保存到: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"保存JWT配置失败: {e}")
            return False
    
    def get_jwt_token(self) -> Optional[str]:
        """获取JWT令牌"""
        config = self.load_config()
        if config:
            return config.get("jwt_token")
        return None
    
    def validate_jwt_format(self, token: str) -> tuple[bool, str]:
        """验证JWT令牌格式"""
        if not token or not token.strip():
            return False, "JWT令牌不能为空"
        
        token = token.strip()
        
        # 基本格式检查
        if not token.startswith("eyJ"):
            return False, "JWT令牌格式不正确（应以'eyJ'开头）"
        
        # 检查是否包含三个部分（用.分隔）
        parts = token.split('.')
        if len(parts) != 3:
            return False, "JWT令牌格式不正确（应包含三个部分）"
        
        # 长度检查
        if len(token) < 100:
            return False, "JWT令牌长度过短"
        
        return True, "JWT令牌格式验证通过"

async def test_jwt_token(jwt_token: str) -> tuple[bool, str]:
    """测试JWT令牌是否有效"""
    try:
        headers = {"Authorization": f"Bearer {jwt_token}"}

        async with httpx.AsyncClient(timeout=30.0) as client:
            # 使用官方推荐的 /user 端点验证JWT令牌
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
                        message = f"✅ JWT令牌验证成功！用户: {user_email}"
                        if user_name:
                            message += f" ({user_name})"
                        if jwt_valid_until:
                            message += f", 有效期至: {jwt_valid_until}"
                        return True, message
                    else:
                        return True, "✅ JWT令牌验证成功，可以访问OpenGWAS API"
                except:
                    return True, "✅ JWT令牌验证成功，可以访问OpenGWAS API"
            elif response.status_code == 401:
                return False, "❌ JWT令牌无效或已过期"
            elif response.status_code == 403:
                return False, "❌ JWT令牌权限不足"
            elif response.status_code == 429:
                return False, "❌ API请求频率超限，请稍后重试"
            else:
                return False, f"❌ API测试失败，状态码: {response.status_code}"

    except httpx.TimeoutException:
        return False, "❌ API连接超时，请检查网络连接"
    except Exception as e:
        return False, f"❌ 测试失败: {str(e)}"

def render_jwt_setup_guide(language: str = "en") -> None:
    """渲染JWT设置指南"""
    
    texts = {
        "zh": {
            "title": "🔑 OpenGWAS JWT令牌配置",
            "description": "为了访问真实的GWAS数据，您需要配置OpenGWAS API的JWT令牌。",
            "step1_title": "步骤1: 获取JWT令牌",
            "step1_desc": "访问OpenGWAS网站获取您的个人JWT令牌：",
            "step2_title": "步骤2: 注册账户",
            "step2_desc": "如果您还没有账户，请先注册：",
            "step3_title": "步骤3: 复制令牌",
            "step3_desc": "登录后，在个人资料页面找到并复制您的JWT令牌",
            "important_note": "重要提示",
            "note_content": "JWT令牌是您的个人凭证，请妥善保管，不要分享给他人。",
            "continue_button": "我已获取JWT令牌，继续配置"
        },
        "en": {
            "title": "🔑 OpenGWAS JWT Token Configuration",
            "description": "To access real GWAS data, you need to configure your OpenGWAS API JWT token.",
            "step1_title": "Step 1: Get JWT Token",
            "step1_desc": "Visit OpenGWAS website to get your personal JWT token:",
            "step2_title": "Step 2: Register Account",
            "step2_desc": "If you don't have an account yet, please register first:",
            "step3_title": "Step 3: Copy Token",
            "step3_desc": "After login, find and copy your JWT token from the profile page",
            "important_note": "Important Note",
            "note_content": "JWT token is your personal credential. Please keep it safe and do not share with others.",
            "continue_button": "I have obtained JWT token, continue setup"
        }
    }
    
    t = texts.get(language, texts["en"])
    
    st.subheader(t["title"])
    st.write(t["description"])
    
    # 步骤指南
    with st.expander(t["step1_title"], expanded=True):
        st.write(t["step1_desc"])
        st.code("https://api.opengwas.io/profile/")
        
        col1, col2 = st.columns(2)
        with col1:
            button1_text = "🌐 Open OpenGWAS Website"
        if st.button(button1_text):
                st.markdown('[OpenGWAS Profile](https://api.opengwas.io/profile/)')

        with col2:
            button2_text = "📚 View API Documentation"
            if st.button(button2_text):
                st.markdown('[OpenGWAS API Docs](https://api.opengwas.io/docs/)')
    
    with st.expander(t["step2_title"]):
        st.write(t["step2_desc"])
        st.code("https://api.opengwas.io/")
    
    with st.expander(t["step3_title"]):
        st.write(t["step3_desc"])
        info_text = "💡 JWT tokens usually start with 'eyJ' and contain three dot-separated parts"
        st.info(info_text)
    
    # 重要提示
    st.warning(f"⚠️ **{t['important_note']}**: {t['note_content']}")

def render_jwt_config_form(language: str = "en") -> Optional[str]:
    """渲染JWT配置表单"""
    
    texts = {
        "zh": {
            "form_title": "配置JWT令牌",
            "token_input": "请输入您的OpenGWAS JWT令牌",
            "token_placeholder": "eyJhbGciOiJSUzI1NiIsImtpZCI6...",
            "description_input": "描述（可选）",
            "description_placeholder": "例如：我的OpenGWAS API令牌",
            "test_button": "测试令牌",
            "save_button": "保存配置",
            "testing": "正在测试令牌...",
            "test_success": "✅ JWT令牌测试成功！",
            "test_failed": "❌ JWT令牌测试失败",
            "save_success": "✅ JWT配置保存成功！",
            "save_failed": "❌ 保存配置失败",
            "format_error": "❌ JWT令牌格式错误"
        },
        "en": {
            "form_title": "Configure JWT Token",
            "token_input": "Please enter your OpenGWAS JWT token",
            "token_placeholder": "eyJhbGciOiJSUzI1NiIsImtpZCI6...",
            "description_input": "Description (optional)",
            "description_placeholder": "e.g., My OpenGWAS API token",
            "test_button": "Test Token",
            "save_button": "Save Configuration",
            "testing": "Testing token...",
            "test_success": "✅ JWT token test successful!",
            "test_failed": "❌ JWT token test failed",
            "save_success": "✅ JWT configuration saved successfully!",
            "save_failed": "❌ Failed to save configuration",
            "format_error": "❌ JWT token format error"
        }
    }
    
    t = texts.get(language, texts["en"])
    
    st.subheader(t["form_title"])
    
    with st.form("jwt_config_form"):
        # JWT令牌输入
        jwt_token = st.text_area(
            t["token_input"],
            placeholder=t["token_placeholder"],
            height=100,
            help="Copy the complete JWT token from OpenGWAS profile page"
        )
        
        # 描述输入
        description = st.text_input(
            t["description_input"],
            placeholder=t["description_placeholder"]
        )
        
        # 按钮
        col1, col2 = st.columns(2)
        with col1:
            test_clicked = st.form_submit_button(t["test_button"], type="secondary")
        with col2:
            save_clicked = st.form_submit_button(t["save_button"], type="primary")
    
    # 处理测试按钮
    if test_clicked:
        if jwt_token:
            # 格式验证
            config_manager = JWTConfigManager()
            is_valid, message = config_manager.validate_jwt_format(jwt_token)
            
            if is_valid:
                with st.spinner(t["testing"]):
                    # 异步测试令牌
                    test_success, test_message = asyncio.run(test_jwt_token(jwt_token))
                    
                    if test_success:
                        st.success(f"{t['test_success']} {test_message}")
                    else:
                        st.error(f"{t['test_failed']}: {test_message}")
            else:
                st.error(f"{t['format_error']}: {message}")
        else:
            warning_text = "请先输入JWT令牌" if language == "zh" else "Please enter JWT token first"
            st.warning(warning_text)
    
    # 处理保存按钮
    if save_clicked:
        if jwt_token:
            config_manager = JWTConfigManager()
            
            # 格式验证
            is_valid, message = config_manager.validate_jwt_format(jwt_token)
            
            if is_valid:
                # 保存配置
                if config_manager.save_config(jwt_token, description):
                    st.success(t["save_success"])
                    st.balloons()
                    return jwt_token
                else:
                    st.error(t["save_failed"])
            else:
                st.error(f"{t['format_error']}: {message}")
        else:
            warning_text = "请先输入JWT令牌" if language == "zh" else "Please enter JWT token first"
            st.warning(warning_text)
    
    return None

def check_jwt_availability() -> tuple[bool, Optional[str]]:
    """检查JWT令牌是否可用"""
    config_manager = JWTConfigManager()
    jwt_token = config_manager.get_jwt_token()
    
    if jwt_token:
        # 验证格式
        is_valid, _ = config_manager.validate_jwt_format(jwt_token)
        return is_valid, jwt_token
    
    return False, None
