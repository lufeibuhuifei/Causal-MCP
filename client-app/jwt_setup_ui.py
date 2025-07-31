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
    st.markdown("### 📋 OpenGWAS JWT令牌配置指南")
    
    with st.expander("🔍 什么是JWT令牌？", expanded=False):
        st.markdown("""
        **JWT (JSON Web Token)** 是OpenGWAS API的认证令牌，用于：
        - 访问GWAS数据库中的研究数据
        - 获取SNP-性状关联信息
        - 确保API访问的安全性和速率限制
        
        **为什么需要JWT令牌？**
        - OpenGWAS API需要认证才能访问完整数据
        - 令牌有效期为14天，需要定期更新
        - 不同用户级别有不同的访问限制
        """)
    
    with st.expander("🔧 如何获取JWT令牌？", expanded=True):
        st.markdown("""
        **步骤1: 注册OpenGWAS账户**
        1. 访问 [OpenGWAS官网](https://api.opengwas.io/)
        2. 点击右上角 "Sign up" 注册账户
        3. 使用邮箱验证账户
        
        **步骤2: 获取JWT令牌**
        1. 登录后访问 [个人资料页面](https://api.opengwas.io/profile/)
        2. 在 "API Access" 部分找到 "JWT Token"
        3. 点击 "Generate Token" 生成新令牌
        4. 复制完整的JWT令牌字符串
        
        **步骤3: 配置到系统**
        1. 将令牌粘贴到下方的输入框
        2. 点击 "测试令牌" 验证有效性
        3. 点击 "保存配置" 完成设置
        """)
    
    with st.expander("⚠️ 注意事项", expanded=False):
        st.markdown("""
        **令牌安全**
        - JWT令牌是敏感信息，请勿分享给他人
        - 令牌有效期为14天，过期后需要重新生成
        - 系统会安全地存储您的令牌
        
        **访问限制**
        - Trial账户：100次请求/10分钟
        - Standard账户：100,000次请求/10分钟
        - 超出限制会导致API调用失败
        
        **故障排除**
        - 如果令牌测试失败，请检查网络连接
        - 确保令牌完整复制，没有多余的空格
        - 过期令牌需要重新生成
        """)

def show_jwt_configuration():
    """显示JWT令牌配置界面"""
    st.markdown("### 🔑 JWT令牌配置")
    
    # 检查当前配置状态
    token_info = jwt_manager.get_token_info()
    
    if token_info.get('configured'):
        st.success("✅ JWT令牌已配置")
        
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"**令牌长度**: {token_info.get('token_length', 'Unknown')} 字符")
        
        with col2:
            if 'expires_in_days' in token_info:
                days_left = token_info['expires_in_days']
                if days_left > 7:
                    st.success(f"**有效期**: 还有 {days_left} 天")
                elif days_left > 0:
                    st.warning(f"**有效期**: 还有 {days_left} 天 (即将过期)")
                else:
                    st.error("**有效期**: 已过期，需要更新")
        
        # 显示详细信息
        if st.checkbox("显示详细信息"):
            st.json(token_info)
        
        # 测试当前令牌
        if st.button("🧪 测试当前令牌", key="test_current"):
            with st.spinner("正在测试JWT令牌..."):
                try:
                    is_valid, message = asyncio.run(jwt_manager.test_jwt_token())
                    if is_valid:
                        st.success(f"✅ {message}")
                    else:
                        st.error(f"❌ {message}")
                except Exception as e:
                    st.error(f"❌ 测试失败: {e}")
    else:
        st.warning("⚠️ 未配置JWT令牌，需要配置后才能访问GWAS数据")
    
    st.markdown("---")
    
    # JWT令牌输入
    st.markdown("#### 配置新的JWT令牌")
    
    jwt_token = st.text_area(
        "JWT令牌",
        height=100,
        placeholder="请粘贴从OpenGWAS获取的完整JWT令牌...",
        help="从 https://api.opengwas.io/profile/ 获取JWT令牌"
    )
    
    description = st.text_input(
        "描述 (可选)",
        value="OpenGWAS API JWT token for Causal-MCP",
        help="为此令牌添加描述信息"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🧪 测试令牌", disabled=not jwt_token.strip()):
            if jwt_token.strip():
                with st.spinner("正在测试JWT令牌..."):
                    try:
                        is_valid, message = asyncio.run(jwt_manager.test_jwt_token(jwt_token.strip()))
                        if is_valid:
                            st.success(f"✅ {message}")
                            st.session_state.jwt_test_passed = True
                        else:
                            st.error(f"❌ {message}")
                            st.session_state.jwt_test_passed = False
                    except Exception as e:
                        st.error(f"❌ 测试失败: {e}")
                        st.session_state.jwt_test_passed = False
    
    with col2:
        test_passed = st.session_state.get('jwt_test_passed', False)
        if st.button("💾 保存配置", disabled=not jwt_token.strip()):
            if jwt_token.strip():
                with st.spinner("正在保存JWT令牌..."):
                    try:
                        success = jwt_manager.save_jwt_token(jwt_token.strip(), description.strip())
                        if success:
                            st.success("✅ JWT令牌配置已保存")
                            st.session_state.jwt_configured = True
                            # 清除测试状态
                            if 'jwt_test_passed' in st.session_state:
                                del st.session_state.jwt_test_passed
                            st.rerun()
                        else:
                            st.error("❌ 保存JWT令牌失败")
                    except Exception as e:
                        st.error(f"❌ 保存失败: {e}")



def show_jwt_setup_page():
    """显示完整的JWT设置页面"""
    st.title("🔑 OpenGWAS JWT令牌配置")
    
    # 检查是否需要显示设置
    if not jwt_manager.is_token_configured():
        st.warning("⚠️ 系统需要配置OpenGWAS JWT令牌才能正常工作")
    
    # 配置指南
    show_jwt_setup_guide()
    
    st.markdown("---")
    
    # JWT配置
    show_jwt_configuration()
    


def check_jwt_requirement():
    """检查JWT令牌要求，如果未配置则显示配置界面"""
    if not jwt_manager.is_token_configured():
        st.error("❌ 未配置OpenGWAS JWT令牌")
        st.markdown("系统需要JWT令牌才能访问GWAS数据，请先完成配置。")
        
        if st.button("🔧 配置JWT令牌"):
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
                st.success(f"🔑 JWT令牌正常 (还有 {days_left} 天)")
            elif days_left > 0:
                st.warning(f"🔑 JWT令牌即将过期 (还有 {days_left} 天)")
            else:
                st.error("🔑 JWT令牌已过期，需要更新")
        else:
            st.info("🔑 JWT令牌已配置")
    else:
        st.error("🔑 未配置JWT令牌")
