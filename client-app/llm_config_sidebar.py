# client-app/llm_config_sidebar.py
"""
LLM配置侧边栏组件
"""

import streamlit as st
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional
from llm_service import LLMService, LLMConfig

logger = logging.getLogger(__name__)

def render_llm_config_section(llm_service: LLMService, language: str = "zh") -> bool:
    """
    渲染简化的LLM配置界面（仅显示状态和设置按钮）

    Args:
        llm_service: LLM服务实例
        language: 界面语言

    Returns:
        bool: 是否需要在主页面显示详细配置
    """
    # 获取当前配置
    current_config = llm_service.get_config()
    status = llm_service.get_status()

    # 检查是否需要在主页面显示详细配置
    if 'show_main_llm_config' not in st.session_state:
        st.session_state.show_main_llm_config = False

    # 显示LLM状态信息
    if status['enabled']:
        if status['available']:
            st.success("🟢 LLM Service: Online")
        else:
            st.warning("🟡 LLM Service: Offline")

        # 显示当前提供商
        provider_name = status.get('provider_name', 'Unknown')
        st.info(f"📡 Provider: {provider_name}")
    else:
        st.info("⚪ LLM Service: Disabled")

    # LLM设置按钮
    if st.button("⚙️ LLM Settings", key="open_llm_config", use_container_width=True):
        st.session_state.show_main_llm_config = True
        st.rerun()

    return st.session_state.get('show_main_llm_config', False)
