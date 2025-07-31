# client-app/llm_first_setup.py
"""
LLM首次设置界面
"""

import streamlit as st
import asyncio
from typing import Dict, Any, Optional
from llm_service import LLMService

def render_llm_first_time_setup(language: str = "en") -> Optional[Dict[str, Any]]:
    """
    渲染LLM首次设置界面，支持选择提供商
    
    Args:
        language: 界面语言
        
    Returns:
        Dict[str, Any]: 如果用户完成设置，返回配置字典，否则返回None
    """
    texts = {
        "zh": {
            "title": "🤖 LLM 智能分析设置",
            "welcome": "欢迎使用因果分析系统！",
            "description": "为了提供更智能的分析解释，请选择并配置您的 LLM 服务：",
            "provider_selection": "选择 LLM 提供商",
            "ollama_option": "🏠 Ollama (本地部署)",
            "deepseek_option": "🌐 DeepSeek API (官方服务)",
            "gemini_option": "🧠 Google Gemini API",
            "skip_option": "⏭️ 跳过设置 (使用基础分析)",
            "ollama_desc": "• 完全本地运行，保护数据隐私\n• 需要本地安装和配置\n• 适合有技术背景的用户",
            "deepseek_desc": "• 官方 API 服务，即开即用\n• 需要 API 密钥，按使用量付费\n• 响应速度快，分析质量高",
            "gemini_desc": "• Google 最新 AI 技术\n• 支持多种模型选择\n• 强大的分析和理解能力",
            "skip_desc": "• 使用基础的规则分析\n• 后续可在设置中配置 LLM",
            "continue_btn": "继续配置",
            "skip_btn": "跳过设置",
            "ollama_setup_title": "配置 Ollama 本地服务",
            "deepseek_setup_title": "配置 DeepSeek API",
            "gemini_setup_title": "配置 Google Gemini API",
            "server_url": "服务地址",
            "api_key": "API 密钥",
            "api_key_help": "从 https://platform.deepseek.com/api_keys 获取",
            "gemini_api_key_help": "从 https://aistudio.google.com/app/apikey 获取",
            "model_selection": "选择模型",
            "test_connection": "测试连接",
            "complete_setup": "完成设置",
            "back": "返回",
            "testing": "正在测试连接...",
            "test_success": "✅ 连接成功！",
            "test_failed": "❌ 连接失败：",
            "setup_complete": "🎉 LLM 设置完成！",
            "getting_models": "正在获取模型列表...",
            "no_models": "未找到可用模型"
        },
        "en": {
            "title": "🤖 LLM Intelligent Analysis Setup",
            "welcome": "Welcome to the Causal Analysis System!",
            "description": "To provide smarter analysis explanations, please select and configure your LLM service:",
            "provider_selection": "Choose LLM Provider",
            "ollama_option": "🏠 Ollama (Local Deployment)",
            "deepseek_option": "🌐 DeepSeek API (Official Service)",
            "gemini_option": "🧠 Google Gemini API",
            "skip_option": "⏭️ Skip Setup (Basic Analysis)",
            "ollama_desc": "• Runs completely locally, protects data privacy\n• Requires local installation and configuration\n• Suitable for technical users",
            "deepseek_desc": "• Official API service, ready to use\n• Requires API key, pay-per-use\n• Fast response, high analysis quality",
            "gemini_desc": "• Google's latest AI technology\n• Multiple model options available\n• Powerful analysis and understanding capabilities",
            "skip_desc": "• Use basic rule-based analysis\n• Can configure LLM later in settings",
            "continue_btn": "Continue Setup",
            "skip_btn": "Skip Setup",
            "ollama_setup_title": "Configure Ollama Local Service",
            "deepseek_setup_title": "Configure DeepSeek API",
            "gemini_setup_title": "Configure Google Gemini API",
            "server_url": "Service URL",
            "api_key": "API Key",
            "api_key_help": "Get from https://platform.deepseek.com/api_keys",
            "gemini_api_key_help": "Get from https://aistudio.google.com/app/apikey",
            "model_selection": "Select Model",
            "test_connection": "Test Connection",
            "complete_setup": "Complete Setup",
            "back": "Back",
            "testing": "Testing connection...",
            "test_success": "✅ Connection successful!",
            "test_failed": "❌ Connection failed:",
            "setup_complete": "🎉 LLM setup completed!",
            "getting_models": "Getting model list...",
            "no_models": "No available models found"
        }
    }
    
    t = texts.get(language, texts["zh"])
    
    # 使用session state来管理设置流程状态
    if 'first_setup_step' not in st.session_state:
        st.session_state.first_setup_step = 'provider_selection'
    if 'first_setup_provider' not in st.session_state:
        st.session_state.first_setup_provider = None
    
    # 创建一个容器来显示设置界面
    with st.container():
        st.markdown(f"### {t['title']}")
        st.markdown(f"**{t['welcome']}**")
        st.write(t["description"])
        
        # 步骤1：选择提供商
        if st.session_state.first_setup_step == 'provider_selection':
            st.markdown(f"#### {t['provider_selection']}")
            
            # 创建四列布局
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"**{t['ollama_option']}**")
                st.markdown(t['ollama_desc'])
                if st.button("Choose Ollama",
                           key="choose_ollama", use_container_width=True):
                    st.session_state.first_setup_provider = 'ollama'
                    st.session_state.first_setup_step = 'configure'
                    st.rerun()

            with col2:
                st.markdown(f"**{t['deepseek_option']}**")
                st.markdown(t['deepseek_desc'])
                if st.button("Choose DeepSeek",
                           key="choose_deepseek", use_container_width=True):
                    st.session_state.first_setup_provider = 'deepseek'
                    st.session_state.first_setup_step = 'configure'
                    st.rerun()

            with col3:
                st.markdown(f"**{t['gemini_option']}**")
                st.markdown(t['gemini_desc'])
                if st.button("Choose Gemini",
                           key="choose_gemini", use_container_width=True):
                    st.session_state.first_setup_provider = 'gemini'
                    st.session_state.first_setup_step = 'configure'
                    st.rerun()

            with col4:
                st.markdown(f"**{t['skip_option']}**")
                st.markdown(t['skip_desc'])
                if st.button(t['skip_btn'], key="skip_setup", use_container_width=True):
                    # 返回禁用LLM的配置
                    _cleanup_first_setup_state()
                    return {
                        "provider": "ollama",
                        "enabled": False,
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
        
        # 步骤2：配置选定的提供商
        elif st.session_state.first_setup_step == 'configure':
            provider = st.session_state.first_setup_provider
            
            if provider == 'ollama':
                return _render_ollama_setup(t, language)
            elif provider == 'deepseek':
                return _render_deepseek_setup(t, language)
            elif provider == 'gemini':
                return _render_gemini_setup(t, language)
    
    return None

def _render_ollama_setup(t: dict, language: str) -> Optional[Dict[str, Any]]:
    """渲染 Ollama 配置界面"""
    st.markdown(f"#### {t['ollama_setup_title']}")

    # 初始化 session state
    if 'first_setup_ollama_base_url' not in st.session_state:
        st.session_state.first_setup_ollama_base_url = "http://localhost:11434"
    if 'first_setup_ollama_models' not in st.session_state:
        st.session_state.first_setup_ollama_models = []
    if 'first_setup_ollama_models_error' not in st.session_state:
        st.session_state.first_setup_ollama_models_error = None
    if 'first_setup_ollama_manual_config' not in st.session_state:
        st.session_state.first_setup_ollama_manual_config = False

    # Ollama 服务配置
    st.markdown("##### Ollama Service Configuration")

    # 服务地址输入框（独占一行）
    base_url = st.text_input(
        t["server_url"],
        value=st.session_state.first_setup_ollama_base_url,
        placeholder="http://localhost:11434",
        help="Please enter the correct Ollama service address first",
        key="first_setup_ollama_url_input"
    )
    # 更新 session state
    st.session_state.first_setup_ollama_base_url = base_url

    # 获取模型列表按钮（与左侧设置保持一致的布局）
    col1, col2 = st.columns([1, 3])
    with col1:
        get_models_clicked = st.button("📋 Get Model List", help="Get available model list from Ollama service", key="first_setup_get_models")

    with col2:
        if st.session_state.first_setup_ollama_models:
            st.success(f"✅ Retrieved {len(st.session_state.first_setup_ollama_models)} models")
        elif st.session_state.first_setup_ollama_models_error:
            st.error(f"❌ {st.session_state.first_setup_ollama_models_error}")

    # 处理获取模型列表
    if get_models_clicked:
        with st.spinner("Getting model list..."):
            try:
                import requests
                import time
                response = requests.get(f"{base_url}/api/tags", timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    models = [model['name'] for model in data.get('models', [])]
                    if models:
                        st.session_state.first_setup_ollama_models = models
                        st.session_state.first_setup_ollama_models_error = None
                    else:
                        st.session_state.first_setup_ollama_models = []
                        st.session_state.first_setup_ollama_models_error = "No available models found"
                else:
                    st.session_state.first_setup_ollama_models = []
                    st.session_state.first_setup_ollama_models_error = f"HTTP {response.status_code}"
            except Exception as e:
                st.session_state.first_setup_ollama_models = []
                st.session_state.first_setup_ollama_models_error = str(e)

            # 强制重新运行以更新状态显示
            time.sleep(0.5)
            st.rerun()

    # 检查是否应该显示配置表单
    show_ollama_form = False
    if st.session_state.first_setup_ollama_models:
        show_ollama_form = True
    elif st.session_state.first_setup_ollama_models_error:
        # 获取失败，显示手动配置选项
        st.warning("⚠️ Unable to automatically get model list, you can manually enter model name")
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🔧 Manual Model Config", help="If unable to get model list, you can manually enter configuration", key="first_setup_manual_config"):
                st.session_state.first_setup_ollama_manual_config = True

        if st.session_state.first_setup_ollama_manual_config:
            show_ollama_form = True
    else:
        # 还没有尝试获取模型列表
        st.info("💡 Please click the '📋 Get Model List' button above to get available models")

    # 初始化Ollama测试状态
    if 'first_setup_ollama_test_success' not in st.session_state:
        st.session_state.first_setup_ollama_test_success = False
    if 'first_setup_ollama_test_config' not in st.session_state:
        st.session_state.first_setup_ollama_test_config = None

    # 详细配置表单
    if show_ollama_form:
        with st.form("ollama_setup_form"):
            # 显示模型选择
            if st.session_state.first_setup_ollama_models:
                # 有可用模型，显示选择框
                selected_model = st.selectbox(
                    t["model_selection"],
                    options=st.session_state.first_setup_ollama_models,
                    index=0,
                    help="Select an available model from Ollama service"
                )
            else:
                # 手动配置模式，显示文本输入框
                st.info("🔧 Manual configuration mode: Please ensure you have installed the corresponding model in Ollama")
                selected_model = st.text_input(
                    t["model_selection"],
                    value="",
                    placeholder="e.g.: deepseek-r1:1.5b, llama2:7b, qwen:7b",
                    help="Please enter the model name installed in Ollama, you can use 'ollama list' command to view installed models"
                )

                # 提供一些常用模型的建议
                st.markdown("**Common Model Suggestions:**")
                st.markdown("- `deepseek-r1:1.5b` - DeepSeek R1 1.5B model")
                st.markdown("- `llama2:7b` - Llama 2 7B model")
                st.markdown("- `qwen:7b` - Qwen 7B model")
                st.markdown("- `mistral:7b` - Mistral 7B model")

            col1, col2 = st.columns(2)
            with col1:
                back_clicked = st.form_submit_button(t["back"])
            with col2:
                test_clicked = st.form_submit_button(t["test_connection"], type="primary")

            if back_clicked:
                st.session_state.first_setup_step = 'provider_selection'
                # 清理测试状态
                st.session_state.first_setup_ollama_test_success = False
                st.session_state.first_setup_ollama_test_config = None
                st.rerun()

            if test_clicked:
                if not selected_model:
                    st.error("Please select or enter model name")
                else:
                    # 测试配置
                    test_config = {
                        "provider": "ollama",
                        "enabled": True,
                        "ollama": {
                            "base_url": base_url,
                            "model_name": selected_model,
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

                    with st.spinner(t["testing"]):
                        temp_service = LLMService(test_config)
                        success, message, models = asyncio.run(temp_service.test_connection())

                        if success:
                            st.success(t["test_success"])
                            # 保存测试成功状态和配置
                            st.session_state.first_setup_ollama_test_success = True
                            st.session_state.first_setup_ollama_test_config = test_config
                        else:
                            st.error(f"{t['test_failed']} {message}")
                            st.session_state.first_setup_ollama_test_success = False
                            st.session_state.first_setup_ollama_test_config = None

        # 在表单外显示完成设置按钮（只有测试成功后才显示）
        if st.session_state.first_setup_ollama_test_success and st.session_state.first_setup_ollama_test_config:
            st.markdown("---")
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button(t["complete_setup"], type="primary", use_container_width=True, key="ollama_complete_setup"):
                    # 获取配置并清理 session state
                    config_to_return = st.session_state.first_setup_ollama_test_config
                    _cleanup_first_setup_state()
                    return config_to_return
    else:
        # 没有表单时，显示返回按钮
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button(t["back"], key="first_setup_ollama_back", use_container_width=True):
                st.session_state.first_setup_step = 'provider_selection'
                st.rerun()

    return None

def _render_deepseek_setup(t: dict, language: str) -> Optional[Dict[str, Any]]:
    """渲染 DeepSeek 配置界面"""
    st.markdown(f"#### {t['deepseek_setup_title']}")

    # 初始化session state来保存测试状态
    if 'first_setup_deepseek_test_success' not in st.session_state:
        st.session_state.first_setup_deepseek_test_success = False
    if 'first_setup_deepseek_test_config' not in st.session_state:
        st.session_state.first_setup_deepseek_test_config = None

    with st.form("deepseek_setup_form"):
        api_key = st.text_input(
            t["api_key"],
            type="password",
            help=t["api_key_help"]
        )

        # 固定使用 deepseek-chat，无需用户选择
        selected_model = "deepseek-chat"
        st.info("💡 Will use DeepSeek Chat model, suitable for various analysis tasks")

        col1, col2 = st.columns(2)
        with col1:
            back_clicked = st.form_submit_button(t["back"])
        with col2:
            test_clicked = st.form_submit_button(t["test_connection"], type="primary")

        if back_clicked:
            st.session_state.first_setup_step = 'provider_selection'
            # 清理测试状态
            st.session_state.first_setup_deepseek_test_success = False
            st.session_state.first_setup_deepseek_test_config = None
            st.rerun()

        if test_clicked and api_key:
            with st.spinner(t["testing"]):
                # 测试 DeepSeek 连接
                test_config = {
                    "provider": "deepseek",
                    "enabled": True,
                    "ollama": {
                        "base_url": "http://localhost:11434",
                        "model_name": "deepseek-r1:1.5b",
                        "temperature": 0.3,
                        "max_tokens": 2048,
                        "timeout": 60
                    },
                    "deepseek": {
                        "api_key": api_key,
                        "base_url": "https://api.deepseek.com",
                        "model_name": selected_model,
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

                temp_service = LLMService(test_config)
                success, message, models = asyncio.run(temp_service.test_connection())

                if success:
                    st.success(t["test_success"])
                    # 保存测试成功状态和配置
                    st.session_state.first_setup_deepseek_test_success = True
                    st.session_state.first_setup_deepseek_test_config = test_config
                else:
                    st.error(f"{t['test_failed']} {message}")
                    st.session_state.first_setup_deepseek_test_success = False
                    st.session_state.first_setup_deepseek_test_config = None
        elif test_clicked and not api_key:
            st.error("Please enter API key")

    # 在表单外显示完成设置按钮（只有测试成功后才显示）
    if st.session_state.first_setup_deepseek_test_success and st.session_state.first_setup_deepseek_test_config:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(t["complete_setup"], type="primary", use_container_width=True, key="deepseek_complete_setup"):
                # 获取配置并清理 session state
                config_to_return = st.session_state.first_setup_deepseek_test_config
                _cleanup_first_setup_state()
                return config_to_return

    return None

def _render_gemini_setup(t: dict, language: str) -> Optional[Dict[str, Any]]:
    """渲染 Gemini 配置界面"""
    st.markdown(f"#### {t['gemini_setup_title']}")

    # 初始化 session state
    if 'first_setup_gemini_test_success' not in st.session_state:
        st.session_state.first_setup_gemini_test_success = False
    if 'first_setup_gemini_api_key' not in st.session_state:
        st.session_state.first_setup_gemini_api_key = ""
    if 'first_setup_gemini_model' not in st.session_state:
        st.session_state.first_setup_gemini_model = "gemini-1.5-flash"
    if 'first_setup_gemini_test_config' not in st.session_state:
        st.session_state.first_setup_gemini_test_config = None

    with st.form("gemini_setup_form"):
        api_key = st.text_input(
            t["api_key"],
            value=st.session_state.first_setup_gemini_api_key,
            type="password",
            help=t["gemini_api_key_help"]
        )
        # 更新 session state
        st.session_state.first_setup_gemini_api_key = api_key

        # Gemini模型选择
        gemini_models = [
            "gemini-1.5-flash",
            "gemini-1.5-pro",
            "gemini-1.0-pro"
        ]

        current_model_index = 0
        if st.session_state.first_setup_gemini_model in gemini_models:
            current_model_index = gemini_models.index(st.session_state.first_setup_gemini_model)

        selected_model = st.selectbox(
            t["model_selection"],
            options=gemini_models,
            index=current_model_index,
            help="Choose the Gemini model for analysis"
        )
        # 更新 session state
        st.session_state.first_setup_gemini_model = selected_model

        st.info("💡 Gemini-1.5-flash is recommended for fast response and good quality")

        col1, col2 = st.columns(2)
        with col1:
            back_clicked = st.form_submit_button(t["back"])
        with col2:
            test_clicked = st.form_submit_button(t["test_connection"], type="primary")

        if back_clicked:
            st.session_state.first_setup_step = 'provider_selection'
            st.rerun()

        if test_clicked and api_key:
            with st.spinner(t["testing"]):
                # 测试 Gemini 连接
                test_config = {
                    "provider": "gemini",
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
                        "api_key": api_key,
                        "base_url": "https://generativelanguage.googleapis.com",
                        "model_name": selected_model,
                        "temperature": 0.3,
                        "max_tokens": 2048,
                        "timeout": 60
                    }
                }

                temp_service = LLMService(test_config)
                success, message, models = asyncio.run(temp_service.test_connection())

                if success:
                    st.success(t["test_success"])
                    # 设置测试成功状态
                    st.session_state.first_setup_gemini_test_success = True
                    st.session_state.first_setup_gemini_test_config = test_config
                else:
                    st.error(f"{t['test_failed']} {message}")
                    st.session_state.first_setup_gemini_test_success = False
        elif test_clicked and not api_key:
            st.error("Please enter API key")
            st.session_state.first_setup_gemini_test_success = False

    # 在表单外显示完成设置按钮（只有测试成功后才显示）
    if st.session_state.first_setup_gemini_test_success and st.session_state.first_setup_gemini_test_config:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button(t["complete_setup"], type="primary", use_container_width=True, key="gemini_complete_setup"):
                # 获取配置并清理 session state
                config_to_return = st.session_state.first_setup_gemini_test_config
                _cleanup_first_setup_state()
                return config_to_return

    return None

def _cleanup_first_setup_state():
    """清理首次设置的 session state"""
    keys_to_remove = [
        'first_setup_step',
        'first_setup_provider',
        'first_setup_ollama_base_url',
        'first_setup_ollama_models',
        'first_setup_ollama_models_error',
        'first_setup_ollama_manual_config',
        'first_setup_ollama_test_success',
        'first_setup_ollama_test_config',
        'first_setup_deepseek_test_success',
        'first_setup_deepseek_test_config',
        'first_setup_gemini_test_success',
        'first_setup_gemini_api_key',
        'first_setup_gemini_model',
        'first_setup_gemini_test_config'
    ]
    for key in keys_to_remove:
        try:
            if hasattr(st.session_state, key):
                delattr(st.session_state, key)
            elif hasattr(st.session_state, 'data') and key in st.session_state.data:
                del st.session_state.data[key]
        except (AttributeError, KeyError):
            pass  # 忽略不存在的键

# 保持向后兼容
def render_llm_quick_setup(language: str = "en") -> Optional[Dict[str, Any]]:
    """渲染LLM快速设置界面（保持向后兼容）"""
    return render_llm_first_time_setup(language)
