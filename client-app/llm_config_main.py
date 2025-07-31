#!/usr/bin/env python3
"""
主页面LLM配置组件
"""

import streamlit as st
import asyncio
import time
from llm_service import LLMService


def render_main_llm_config(llm_service: LLMService, language: str = "en") -> bool:
    """
    在主页面渲染详细的LLM配置界面

    Args:
        llm_service: LLM服务实例
        language: 界面语言

    Returns:
        bool: 配置是否有变更
    """
    # 初始化返回值
    config_changed = False

    # 获取当前配置
    current_config = llm_service.get_config()
    status = llm_service.get_status()

    # 初始化 session state
    if 'config_use_llm' not in st.session_state:
        st.session_state.config_use_llm = current_config.get("enabled", True)
    if 'config_provider' not in st.session_state:
        st.session_state.config_provider = current_config.get("provider", "ollama")

    # 界面文本
    texts = {
        "zh": {
            "llm_config_title": "LLM 配置",
            "enable_llm": "启用 LLM 增强分析",
            "use_basic_analysis": "使用基础分析",
            "provider": "LLM 提供商",
            "server_url": "服务地址",
            "model_name": "模型",
            "api_key": "API 密钥",
            "test_connection": "测试连接",
            "save_config": "保存配置",
            "close_config": "关闭配置",
            "testing": "测试中...",
            "test_success": "✅ 连接成功",
            "test_failed": "❌ 连接失败",
            "available_models": "可用模型:",
            "config_saved": "✅ 配置保存成功",
            "config_save_failed": "❌ 配置保存失败",
            "ollama_desc": "本地 Ollama 服务",
            "deepseek_desc": "DeepSeek 官方 API",
            "gemini_desc": "Google Gemini API",
            "api_key_placeholder": "请输入 DeepSeek API 密钥",
            "api_key_help": "从 https://platform.deepseek.com/api_keys 获取",
            "gemini_api_key_placeholder": "请输入 Gemini API 密钥",
            "gemini_api_key_help": "从 https://aistudio.google.com/app/apikey 获取",
            "basic_analysis_warning": "⚠️ 选择基础分析将清空当前 LLM 配置",
            "confirm_basic_analysis": "确认使用基础分析",
            "llm_config_cleared": "✅ LLM 配置已清空，将使用基础分析"
        },
        "en": {
            "llm_config_title": "LLM Configuration",
            "enable_llm": "Enable LLM Analysis",
            "use_basic_analysis": "Use Basic Analysis",
            "provider": "LLM Provider",
            "server_url": "Server URL",
            "model_name": "Model",
            "api_key": "API Key",
            "test_connection": "Test Connection",
            "save_config": "Save Config",
            "close_config": "Close Config",
            "testing": "Testing...",
            "test_success": "✅ Connected",
            "test_failed": "❌ Failed",
            "available_models": "Available models:",
            "config_saved": "✅ Configuration saved",
            "config_save_failed": "❌ Failed to save",
            "ollama_desc": "Local Ollama Service",
            "deepseek_desc": "DeepSeek Official API",
            "gemini_desc": "Google Gemini API",
            "api_key_placeholder": "Enter DeepSeek API Key",
            "api_key_help": "Get from https://platform.deepseek.com/api_keys",
            "gemini_api_key_placeholder": "Enter Gemini API Key",
            "gemini_api_key_help": "Get from https://aistudio.google.com/app/apikey",
            "basic_analysis_warning": "⚠️ Choosing basic analysis will clear current LLM configuration",
            "confirm_basic_analysis": "Confirm Basic Analysis",
            "llm_config_cleared": "✅ LLM configuration cleared, using basic analysis"
        }
    }

    t = texts.get(language, texts["en"])

    # 配置界面标题和关闭按钮
    col1, col2 = st.columns([4, 1])
    with col1:
        st.header(t["llm_config_title"])
    with col2:
        if st.button("✖️ Close", key="close_config_header", help="Close LLM configuration"):
            st.session_state.show_main_llm_config = False
            st.rerun()

    # 分析模式选择（在表单外，可以立即响应）
    col1, col2 = st.columns(2)

    with col1:
        # 使用独立的 key 来避免状态冲突
        analysis_key = "llm_analysis_mode_selector"

        # 初始化独立键值（只在第一次时设置）
        if analysis_key not in st.session_state:
            st.session_state[analysis_key] = st.session_state.config_use_llm

        use_llm = st.radio(
            "Analysis Mode",
            options=[True, False],
            format_func=lambda x: t["enable_llm"] if x else t["use_basic_analysis"],
            key=analysis_key
        )

        # 更新配置状态
        st.session_state.config_use_llm = use_llm

    with col2:
        if not use_llm:
            st.warning(t["basic_analysis_warning"])
            if st.button(t["confirm_basic_analysis"], type="primary", use_container_width=True):
                # 确认选择基础分析
                config = _build_config("ollama", False, "", "", "", 0.3, 2048, 60, current_config)
                success, message = llm_service.update_config(config)

                if success:
                    # 标记用户主动选择基础分析
                    st.session_state.user_chose_basic_analysis = True
                    st.session_state.llm_configured_via_sidebar = False
                    st.session_state.llm_setup_completed = True
                    st.session_state.show_main_llm_config = False

                    st.success(t["llm_config_cleared"])
                    time.sleep(1.5)
                    st.rerun()
                else:
                    st.error(f"{t['config_save_failed']} {message}")

    # LLM 提供商选择（在表单外）
    provider = "ollama"  # 默认值
    if use_llm:
        provider_options = {
            "ollama": t["ollama_desc"],
            "deepseek": t["deepseek_desc"],
            "gemini": t["gemini_desc"]
        }

        # 使用独立的 key 来避免状态冲突
        provider_key = "llm_provider_selector"

        # 初始化独立键值（只在第一次时设置）
        if provider_key not in st.session_state:
            st.session_state[provider_key] = st.session_state.config_provider

        provider = st.selectbox(
            t["provider"],
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            key=provider_key
        )

        # 更新配置状态
        st.session_state.config_provider = provider

    # Ollama 模型获取（在表单外）
    if use_llm and provider == "ollama":
        # 初始化 session state 存储模型列表
        if 'ollama_models' not in st.session_state:
            st.session_state.ollama_models = []
        if 'ollama_models_error' not in st.session_state:
            st.session_state.ollama_models_error = None
        if 'ollama_base_url' not in st.session_state:
            st.session_state.ollama_base_url = current_config.get("ollama", {}).get("base_url", "http://localhost:11434")

        # 服务地址输入（在表单外，可以立即响应）
        st.subheader("🔧 Ollama Service Configuration")

        base_url = st.text_input(
            t["server_url"],
            value=st.session_state.ollama_base_url,
            placeholder="http://localhost:11434",
            help="Please enter the correct Ollama service address first",
            key="ollama_base_url_input"
        )

        # 更新 session state
        st.session_state.ollama_base_url = base_url

        # 获取模型列表按钮
        col1, col2 = st.columns([1, 3])
        with col1:
            get_models_clicked = st.button("📋 Get Model List", help="Get available models from Ollama service")

        with col2:
            if st.session_state.ollama_models:
                st.success(f"✅ Retrieved {len(st.session_state.ollama_models)} models")
            elif st.session_state.ollama_models_error:
                st.error(f"❌ {st.session_state.ollama_models_error}")

        # 处理获取模型列表
        if get_models_clicked:
            with st.spinner("Getting model list..."):
                try:
                    import requests
                    response = requests.get(f"{base_url}/api/tags", timeout=10)
                    if response.status_code == 200:
                        data = response.json()
                        models = [model['name'] for model in data.get('models', [])]
                        if models:
                            st.session_state.ollama_models = models
                            st.session_state.ollama_models_error = None
                            st.success(f"✅ Successfully retrieved {len(models)} models")
                        else:
                            st.session_state.ollama_models = []
                            st.session_state.ollama_models_error = "No available models found"
                            st.warning("⚠️ No available models found, please ensure models are installed in Ollama")
                    else:
                        st.session_state.ollama_models = []
                        st.session_state.ollama_models_error = f"HTTP {response.status_code}"
                        st.error(f"❌ Failed to get model list: HTTP {response.status_code}")
                except Exception as e:
                    st.session_state.ollama_models = []
                    st.session_state.ollama_models_error = str(e)
                    st.error(f"❌ Failed to connect to Ollama service: {e}")

                # 强制重新运行以更新状态显示
                time.sleep(0.5)
                st.rerun()

    # 检查是否应该显示配置表单
    show_ollama_form = False
    if use_llm and provider == "ollama":
        # 初始化手动配置状态
        if 'ollama_manual_config' not in st.session_state:
            st.session_state.ollama_manual_config = False

        # 只有在获取到模型列表或用户明确要求手动配置时才显示表单
        if st.session_state.ollama_models:
            show_ollama_form = True
        elif st.session_state.ollama_models_error:
            # 获取失败，显示手动配置选项
            st.warning("⚠️ Unable to automatically get model list, you can manually enter model name")
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🔧 Manual Model Config", help="If unable to get model list, you can manually enter configuration"):
                    st.session_state.ollama_manual_config = True

            if st.session_state.ollama_manual_config:
                show_ollama_form = True
        else:
            # 还没有尝试获取模型列表
            st.info("💡 Please click the '📋 Get Model List' button above to get available models")

    # 详细配置表单
    if show_ollama_form or (use_llm and provider in ["deepseek", "gemini"]):
        with st.form("main_llm_config_form", clear_on_submit=False):
            # 根据选择的提供商和模式显示相应配置
            if use_llm and provider == "ollama" and show_ollama_form:
                ollama_config = current_config.get("ollama", {})

                # 使用 session state 中的地址
                base_url = st.session_state.ollama_base_url

                # 显示模型选择
                if st.session_state.ollama_models:
                    # 有可用模型，显示选择框
                    current_model = ollama_config.get("model_name", "")
                    if current_model not in st.session_state.ollama_models and st.session_state.ollama_models:
                        current_model = st.session_state.ollama_models[0]

                    model_name = st.selectbox(
                        t["model_name"],
                        options=st.session_state.ollama_models,
                        index=st.session_state.ollama_models.index(current_model) if current_model in st.session_state.ollama_models else 0,
                        help="Select an available model from Ollama service"
                    )
                else:
                    # 手动配置模式，显示文本输入框
                    st.info("🔧 Manual configuration mode: Please ensure you have installed the corresponding model in Ollama")
                    model_name = st.text_input(
                        t["model_name"],
                        value=ollama_config.get("model_name", ""),
                        placeholder="e.g.: deepseek-r1:1.5b, llama2:7b, qwen:7b",
                        help="Please enter the model name installed in Ollama, you can use 'ollama list' command to view installed models"
                    )

                    # 提供一些常用模型的建议
                    st.markdown("**Common Model Suggestions:**")
                    st.markdown("- `deepseek-r1:1.5b` - DeepSeek R1 1.5B model")
                    st.markdown("- `llama2:7b` - Llama 2 7B model")
                    st.markdown("- `qwen:7b` - Qwen 7B model")
                    st.markdown("- `mistral:7b` - Mistral 7B model")

                temperature = ollama_config.get("temperature", 0.3)
                max_tokens = ollama_config.get("max_tokens", 2048)
                timeout = ollama_config.get("timeout", 60)
                api_key = ""

            elif use_llm and provider == "deepseek":
                deepseek_config = current_config.get("deepseek", {})

                api_key = st.text_input(
                    t["api_key"],
                    value=deepseek_config.get("api_key", ""),
                    placeholder=t["api_key_placeholder"],
                    type="password",
                    help=t["api_key_help"]
                )

                base_url = deepseek_config.get("base_url", "https://api.deepseek.com")
                model_name = "deepseek-chat"  # 固定使用 deepseek-chat

                temperature = deepseek_config.get("temperature", 0.3)
                max_tokens = deepseek_config.get("max_tokens", 2048)
                timeout = deepseek_config.get("timeout", 60)

            elif use_llm and provider == "gemini":
                gemini_config = current_config.get("gemini", {})

                api_key = st.text_input(
                    t["api_key"],
                    value=gemini_config.get("api_key", ""),
                    placeholder=t["gemini_api_key_placeholder"],
                    type="password",
                    help=t["gemini_api_key_help"]
                )

                base_url = gemini_config.get("base_url", "https://generativelanguage.googleapis.com")

                # Gemini模型选择
                gemini_models = [
                    "gemini-1.5-flash",
                    "gemini-1.5-pro",
                    "gemini-1.0-pro"
                ]

                current_model = gemini_config.get("model_name", "gemini-1.5-flash")
                model_name = st.selectbox(
                    t["model_name"],
                    options=gemini_models,
                    index=gemini_models.index(current_model) if current_model in gemini_models else 0
                )

                temperature = gemini_config.get("temperature", 0.3)
                max_tokens = gemini_config.get("max_tokens", 2048)
                timeout = gemini_config.get("timeout", 60)
            else:
                # 使用基础分析时的默认值
                base_url = ""
                model_name = ""
                api_key = ""
                temperature = 0.3
                max_tokens = 2048
                timeout = 60

            # 按钮布局
            col1, col2, col3 = st.columns(3)

            with col1:
                test_clicked = st.form_submit_button(
                    t["test_connection"],
                    disabled=not use_llm,
                    use_container_width=True
                )

            with col2:
                save_clicked = st.form_submit_button(
                    t["save_config"],
                    type="primary",
                    use_container_width=True
                )

            with col3:
                close_clicked = st.form_submit_button(
                    t["close_config"],
                    use_container_width=True
                )
    else:
        # 没有表单时初始化按钮变量
        test_clicked = False
        save_clicked = False

        # 没有表单时，在页面底部显示关闭按钮
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            close_clicked = st.button(
                t["close_config"],
                type="secondary",
                use_container_width=True,
                key="close_config_no_form"
            )

        # 初始化配置变量
        base_url = ""
        model_name = ""
        api_key = ""
        temperature = 0.3
        max_tokens = 2048
        timeout = 60

    # 处理测试按钮
    if test_clicked and use_llm:
        with st.spinner(t["testing"]):
            test_config = _build_config(provider, use_llm, base_url, model_name, api_key, temperature, max_tokens, timeout, current_config)
            temp_service = LLMService(test_config)
            success, message, models = asyncio.run(temp_service.test_connection())
            
            if success:
                st.success(t["test_success"])
                if models:
                    st.info(f"{t['available_models']} {', '.join(models)}")
            else:
                st.error(f"{t['test_failed']} {message}")
    
    # 处理保存按钮
    if save_clicked:
        new_config = _build_config(provider, use_llm, base_url, model_name, api_key, temperature, max_tokens, timeout, current_config)
        success, message = llm_service.update_config(new_config)

        if success:
            if not use_llm:
                # 如果选择基础分析，清空 LLM 配置并重置设置状态
                st.session_state.llm_setup_completed = False
                st.session_state.user_chose_basic_analysis = True  # 标记用户主动选择基础分析
                st.session_state.llm_configured_via_sidebar = False
                st.success(t["llm_config_cleared"])
            else:
                st.success(t["config_saved"])
                # 标记用户通过侧边栏配置了 LLM
                st.session_state.llm_configured_via_sidebar = True
                st.session_state.llm_setup_completed = True
                st.session_state.user_chose_basic_analysis = False

                # 立即重新初始化 LLM 服务以更新状态
                with st.spinner("Applying new configuration..."):
                    init_success = asyncio.run(llm_service.initialize())
                    if init_success:
                        st.success("✅ LLM service successfully reinitialized")
                    else:
                        st.warning("⚠️ LLM service reinitialization failed, but configuration has been saved")

            config_changed = True
            st.session_state.show_main_llm_config = False
            time.sleep(1.5)
            st.rerun()
        else:
            st.error(f"{t['config_save_failed']} {message}")

    # 处理关闭按钮
    if close_clicked:
        st.session_state.show_main_llm_config = False
        st.rerun()

    return config_changed


def _build_config(provider, enabled, base_url, model_name, api_key, temperature, max_tokens, timeout, current_config):
    """构建配置字典"""
    if not enabled:
        # 如果禁用 LLM，返回一个清空的配置
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
    
    new_config = {
        "provider": provider,
        "enabled": enabled
    }
    
    if provider == "ollama":
        new_config["ollama"] = {
            "base_url": base_url,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout
        }
        new_config["deepseek"] = current_config.get("deepseek", {})
    elif provider == "deepseek":
        new_config["deepseek"] = {
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout
        }
        new_config["ollama"] = current_config.get("ollama", {})
        new_config["gemini"] = current_config.get("gemini", {})
    elif provider == "gemini":
        new_config["gemini"] = {
            "api_key": api_key,
            "base_url": base_url,
            "model_name": model_name,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout": timeout
        }
        new_config["ollama"] = current_config.get("ollama", {})
        new_config["deepseek"] = current_config.get("deepseek", {})

    return new_config
