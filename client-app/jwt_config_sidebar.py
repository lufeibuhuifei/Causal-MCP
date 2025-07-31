# client-app/jwt_config_sidebar.py
"""
JWT配置侧边栏组件
"""

import streamlit as st
import asyncio
import logging
from typing import Dict, Any, Optional
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jwt_manager import jwt_manager

logger = logging.getLogger(__name__)

def render_jwt_config_section(language: str = "zh") -> bool:
    """
    渲染简化的JWT配置界面（仅显示状态和设置按钮）

    Args:
        language: 界面语言

    Returns:
        bool: 是否需要在主页面显示详细配置
    """
    # 检查是否需要在主页面显示详细配置
    if 'show_main_jwt_config' not in st.session_state:
        st.session_state.show_main_jwt_config = False

    # 获取JWT令牌状态
    jwt_token = jwt_manager.get_jwt_token()
    token_info = jwt_manager.get_token_info()
    
    # 显示JWT状态信息
    if jwt_token:
        # 检查令牌是否即将过期
        days_remaining = jwt_manager.get_token_expiry_days()
        is_expiring_soon = jwt_manager.is_token_expiring_soon()

        if is_expiring_soon and days_remaining >= 0:
            if days_remaining == 0:
                st.error("🔴 JWT Token: Expires Today!")
            elif days_remaining == 1:
                st.warning("🟡 JWT Token: Expires Tomorrow")
            else:
                st.warning(f"🟡 JWT Token: Expires in {days_remaining} days")
        else:
            st.success("🟢 JWT Token: Configured")

        # 显示用户信息（如果可用）
        if token_info.get('configured'):
            # 尝试获取用户信息（简化版，避免在侧边栏进行网络请求）
            try:
                # 只在没有缓存时才进行验证
                if 'jwt_user_info' not in st.session_state:
                    st.session_state.jwt_user_info = {
                        'valid': True,  # 假设已配置的令牌是有效的
                        'message': 'Token configured',
                        'needs_verification': True
                    }

                # 显示简化的状态信息
                user_info = st.session_state.get('jwt_user_info', {})
                if user_info.get('valid') and not user_info.get('needs_verification'):
                    # 提取用户邮箱（如果消息中包含）
                    message = user_info.get('message', '')
                    if '用户:' in message:
                        user_part = message.split('用户:')[1].split(',')[0].strip()
                        if '@' in user_part:
                            # 只显示邮箱的前几个字符
                            user_display = user_part[:3] + "***@" + user_part.split('@')[1]
                            st.info(f"👤 {user_display}")
                        else:
                            st.info(f"👤 {user_part}")
                    else:
                        st.info("✅ Token Valid")
                else:
                    st.info("🔑 Token Configured")

                # 显示过期时间信息
                if days_remaining >= 0:
                    if days_remaining <= 7:  # 一周内过期显示详细信息
                        if days_remaining == 0:
                            st.error("⏰ Expires today!")
                        elif days_remaining == 1:
                            st.warning("⏰ Expires tomorrow")
                        else:
                            st.info(f"⏰ {days_remaining} days left")

            except Exception as e:
                logger.debug(f"显示JWT状态失败: {e}")
                st.info("🔑 Token Configured")
    else:
        st.error("🔴 JWT Token: Not Configured")
        st.warning("⚠️ GWAS data unavailable")

    # JWT设置按钮
    if st.button("🔑 JWT Settings", key="open_jwt_config", use_container_width=True):
        st.session_state.show_main_jwt_config = True
        # 清除缓存的用户信息，强制重新验证
        if 'jwt_user_info' in st.session_state:
            del st.session_state.jwt_user_info
        st.rerun()

    return st.session_state.show_main_jwt_config

def render_jwt_config_modal(language: str = "zh"):
    """
    渲染JWT配置模态框
    
    Args:
        language: 界面语言
    """
    if not st.session_state.get('show_main_jwt_config', False):
        return
    
    # 创建模态框样式的容器
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin: 10px 0;">
        <h3 style="color: #1f77b4; margin-top: 0;">🔑 JWT Token Configuration</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # 获取当前JWT状态
    current_token = jwt_manager.get_jwt_token()
    token_info = jwt_manager.get_token_info()
    
    # 显示当前状态
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if current_token:
            st.success("✅ JWT Token is currently configured")
            # 显示令牌信息
            if token_info.get('configured'):
                token_length = token_info.get('token_length', 0)
                st.info(f"Token length: {token_length} characters")
                
                # 显示过期信息（如果可用）
                expires_in_days = token_info.get('expires_in_days')
                if expires_in_days is not None:
                    if expires_in_days > 7:
                        st.success(f"⏰ Expires in {expires_in_days} days")
                    elif expires_in_days > 0:
                        st.warning(f"⚠️ Expires in {expires_in_days} days")
                    else:
                        st.error("❌ Token has expired")
        else:
            st.warning("⚠️ No JWT token configured")
            st.info("You need to configure a JWT token to access GWAS data.")
    
    with col2:
        # 关闭按钮
        if st.button("❌ Close", key="close_jwt_config"):
            st.session_state.show_main_jwt_config = False
            st.rerun()
    
    # JWT配置表单
    with st.form("jwt_config_form"):
        st.markdown("#### Configure JWT Token")
        
        # JWT令牌输入
        jwt_input = st.text_area(
            "JWT Token",
            value="",
            height=100,
            placeholder="Paste your OpenGWAS JWT token here...",
            help="Get your JWT token from https://api.opengwas.io/profile/"
        )
        
        # 描述输入
        description = st.text_input(
            "Description (optional)",
            value="OpenGWAS API JWT token for Causal-MCP",
            help="Optional description for this token"
        )
        
        # 按钮行
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            test_button = st.form_submit_button("🧪 Test Token", use_container_width=True)
        
        with col2:
            save_button = st.form_submit_button("💾 Save Token", use_container_width=True)
        
        with col3:
            clear_button = st.form_submit_button("🗑️ Clear Token", use_container_width=True)
    
    # 处理按钮操作
    if test_button and jwt_input.strip():
        with st.spinner("Testing JWT token..."):
            try:
                async def test_token():
                    return await jwt_manager.test_jwt_token(jwt_input.strip())

                is_valid, message = asyncio.run(test_token())

                if is_valid:
                    st.success(f"✅ {message}")
                    # 缓存测试结果
                    st.session_state.jwt_test_result = {
                        'valid': True,
                        'message': message,
                        'token': jwt_input.strip()
                    }
                else:
                    st.error(f"❌ {message}")
                    st.session_state.jwt_test_result = {
                        'valid': False,
                        'message': message,
                        'token': jwt_input.strip()
                    }

            except Exception as e:
                error_msg = f"Test failed: {str(e)}"
                st.error(f"❌ {error_msg}")
                st.session_state.jwt_test_result = {
                    'valid': False,
                    'message': error_msg,
                    'token': jwt_input.strip()
                }

    elif save_button and jwt_input.strip():
        with st.spinner("Saving JWT token..."):
            try:
                # 检查是否已经测试过这个令牌
                test_result = st.session_state.get('jwt_test_result', {})
                if (test_result.get('token') == jwt_input.strip() and
                    test_result.get('valid')):
                    # 使用缓存的测试结果
                    success = jwt_manager.save_jwt_token(jwt_input.strip(), description)
                    if success:
                        st.success("✅ JWT token saved successfully!")
                        st.info(f"Token info: {test_result.get('message', 'Token configured')}")
                        # 更新用户信息缓存
                        st.session_state.jwt_user_info = {
                            'valid': True,
                            'message': test_result.get('message', 'Token configured'),
                            'needs_verification': False
                        }
                        # 清除JWT过期状态
                        st.session_state.jwt_expired = False
                        st.session_state.jwt_validity_checked = False
                        st.session_state.jwt_config_success = True
                    else:
                        st.error("❌ Failed to save token to file")
                else:
                    # 需要先测试令牌
                    async def test_and_save():
                        is_valid, test_message = await jwt_manager.test_jwt_token(jwt_input.strip())
                        if is_valid:
                            success = jwt_manager.save_jwt_token(jwt_input.strip(), description)
                            return success, test_message
                        else:
                            return False, test_message

                    success, message = asyncio.run(test_and_save())

                    if success:
                        st.success("✅ JWT token saved successfully!")
                        st.info(f"Token info: {message}")
                        # 更新用户信息缓存
                        st.session_state.jwt_user_info = {
                            'valid': True,
                            'message': message,
                            'needs_verification': False
                        }
                        # 清除JWT过期状态
                        st.session_state.jwt_expired = False
                        st.session_state.jwt_validity_checked = False
                        st.session_state.jwt_config_success = True
                    else:
                        st.error(f"❌ Failed to save token: {message}")

            except Exception as e:
                st.error(f"❌ Save failed: {str(e)}")

    elif clear_button:
        # 清除令牌（删除配置文件）
        try:
            if jwt_manager.config_file.exists():
                jwt_manager.config_file.unlink()
                st.success("✅ JWT token cleared successfully!")
                # 清除所有相关缓存
                for key in ['jwt_user_info', 'jwt_test_result']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.jwt_config_success = True
            else:
                st.info("ℹ️ No JWT token to clear")
        except Exception as e:
            st.error(f"❌ Failed to clear token: {str(e)}")
    
    # 如果操作成功，延迟关闭界面
    if st.session_state.get('jwt_config_success', False):
        st.session_state.jwt_config_success = False
        st.session_state.show_main_jwt_config = False
        st.rerun()
    
    # 帮助信息
    with st.expander("ℹ️ How to get JWT Token"):
        st.markdown("""
        **Steps to get your JWT token:**
        
        1. Visit [OpenGWAS Profile Page](https://api.opengwas.io/profile/)
        2. Register for an account or log in
        3. Generate a new JWT token
        4. Copy the token and paste it above
        5. Click "Test Token" to verify, then "Save Token"
        
        **Note:** JWT tokens are valid for 14 days and need to be renewed periodically.
        """)
    
    st.markdown("---")
