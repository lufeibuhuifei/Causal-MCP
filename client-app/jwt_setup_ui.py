#!/usr/bin/env python3
"""
JWT令牌配置界面
为用户提供友好的JWT令牌配置和测试功能
"""

import streamlit as st
import asyncio
import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jwt_manager import jwt_manager

def show_jwt_setup_guide():
    """显示JWT令牌配置指南"""
    st.markdown("### 📋 OpenGWAS JWT Token Configuration Guide")

    with st.expander("🔍 What is a JWT Token?", expanded=False):
        st.markdown("""
        **JWT (JSON Web Token)** is an authentication token for OpenGWAS API, used for:
        - Accessing research data in GWAS databases
        - Retrieving SNP-trait association information
        - Ensuring API access security and rate limiting

        **Why do you need a JWT token?**
        - OpenGWAS API requires authentication to access complete data
        - Token validity period is 14 days, requires periodic renewal
        - Different user levels have different access restrictions
        """)

    with st.expander("🔧 How to obtain a JWT Token?", expanded=True):
        st.markdown("""
        **Step 1: Register OpenGWAS Account**
        1. Visit [OpenGWAS Official Website](https://api.opengwas.io/)
        2. Click "Sign up" in the top right corner to register an account
        3. Verify your account using email

        **Step 2: Obtain JWT Token**
        1. After login, visit [Profile Page](https://api.opengwas.io/profile/)
        2. Find "JWT Token" in the "API Access" section
        3. Click "Generate Token" to create a new token
        4. Copy the complete JWT token string

        **Step 3: Configure in System**
        1. Paste the token into the input box below
        2. Click "Test Token" to verify validity
        3. Click "Save Configuration" to complete setup
        """)

    with st.expander("⚠️ Important Notes", expanded=False):
        st.markdown("""
        **Token Security**
        - JWT token is sensitive information, do not share with others
        - Token validity period is 14 days, needs regeneration after expiration
        - System will securely store your token

        **Access Limitations**
        - Trial account: 100 requests/10 minutes
        - Standard account: 100,000 requests/10 minutes
        - Exceeding limits will cause API call failures

        **Troubleshooting**
        - If token test fails, please check network connection
        - Ensure token is copied completely without extra spaces
        - Expired tokens need to be regenerated
        """)

def show_jwt_configuration():
    """显示JWT令牌配置界面"""
    st.markdown("### 🔑 JWT Token Configuration")

    # 检查当前配置状态
    token_info = jwt_manager.get_token_info()

    if token_info.get('configured'):
        st.success("✅ JWT Token Configured")

        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**Token Length**: {token_info.get('token_length', 'Unknown')} characters")

        with col2:
            if 'expires_in_days' in token_info:
                days_left = token_info['expires_in_days']
                if days_left > 7:
                    st.success(f"**Validity**: {days_left} days remaining")
                elif days_left > 0:
                    st.warning(f"**Validity**: {days_left} days remaining (expiring soon)")
                else:
                    st.error("**Validity**: Expired, needs update")

        # 显示详细信息
        if st.checkbox("Show Details"):
            st.json(token_info)

        # 测试当前令牌
        if st.button("🧪 Test Current Token", key="test_current"):
            with st.spinner("Testing JWT token..."):
                try:
                    is_valid, message = asyncio.run(jwt_manager.test_jwt_token())
                    if is_valid:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"❌ Test failed: {e}")
    else:
        st.warning("⚠️ JWT token not configured, configuration required to access GWAS data")

    st.markdown("---")

    # JWT令牌输入
    st.markdown("#### Configure New JWT Token")

    jwt_token = st.text_area(
        "JWT Token",
        height=100,
        placeholder="Please paste the complete JWT token obtained from OpenGWAS...",
        help="Get JWT token from https://api.opengwas.io/profile/"
    )

    description = st.text_input(
        "Description (optional)",
        value="OpenGWAS API JWT token for Causal-MCP",
        help="Add description information for this token"
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🧪 Test Token", disabled=not jwt_token.strip()):
            if jwt_token.strip():
                with st.spinner("Testing JWT token..."):
                    try:
                        is_valid, message = asyncio.run(jwt_manager.test_jwt_token(jwt_token.strip()))
                        if is_valid:
                            st.success(f"✅ {message}")
                            st.session_state.jwt_test_passed = True
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.jwt_test_passed = False
                    except Exception as e:
                        st.error(f"❌ Test failed: {e}")
                        st.session_state.jwt_test_passed = False

    with col2:
        test_passed = st.session_state.get('jwt_test_passed', False)
        if st.button("💾 Save Configuration", disabled=not jwt_token.strip()):
            if jwt_token.strip():
                with st.spinner("Saving JWT token..."):
                    try:
                        success = jwt_manager.save_jwt_token(jwt_token.strip(), description.strip())
                        if success:
                            st.success("✅ JWT token configuration saved")
                            st.session_state.jwt_configured = True
                            # 清除测试状态
                            if 'jwt_test_passed' in st.session_state:
                                del st.session_state.jwt_test_passed
                            st.rerun()
                        else:
                            st.error("❌ Failed to save JWT token")
                    except Exception as e:
                        st.error(f"❌ Save failed: {e}")



def show_jwt_setup_page():
    """显示完整的JWT设置页面"""
    st.title("🔑 OpenGWAS JWT Token Configuration")

    # 检查是否需要显示设置
    if not jwt_manager.is_token_configured():
        st.warning("⚠️ System requires OpenGWAS JWT token configuration to work properly")

    # 配置指南
    show_jwt_setup_guide()

    st.markdown("---")

    # JWT配置
    show_jwt_configuration()



def check_jwt_requirement():
    """检查JWT令牌要求，如果未配置则显示配置界面"""
    if not jwt_manager.is_token_configured():
        st.error("❌ OpenGWAS JWT token not configured")
        st.markdown("System requires JWT token to access GWAS data, please complete configuration first.")

        if st.button("🔧 Configure JWT Token"):
            st.session_state.show_jwt_setup = True
            st.rerun()

        return False

    return True

def jwt_status_indicator():
    """显示JWT状态指示器"""
    if jwt_manager.is_token_configured():
        token_info = jwt_manager.get_token_info()

        if 'expires_in_days' in token_info:
            days_left = token_info['expires_in_days']
            if days_left > 7:
                st.success(f"🔑 JWT token normal ({days_left} days remaining)")
            elif days_left > 0:
                st.warning(f"🔑 JWT token expiring soon ({days_left} days remaining)")
            else:
                st.error("🔑 JWT token expired, needs update")
        else:
            st.info("🔑 JWT token configured")
    else:
        st.error("🔑 JWT token not configured")
