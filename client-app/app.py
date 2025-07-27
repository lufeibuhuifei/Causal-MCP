# client-app/app.py
"""
Streamlit web application for the Causal-MCP framework.
"""

import streamlit as st
import asyncio
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import logging
import os
import base64
import zipfile
import io
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)

# Add current directory to Python path to ensure correct imports
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# JWT令牌集中管理
import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from jwt_manager import jwt_manager
from jwt_setup_ui import check_jwt_requirement, jwt_status_indicator, show_jwt_setup_page

def load_jwt_token():
    """从集中管理器加载JWT令牌"""
    jwt_token = jwt_manager.get_jwt_token()
    if jwt_token:
        # 检查令牌是否过期
        if jwt_manager.is_token_expired():
            logging.warning("JWT令牌已过期，需要重新配置")
            return None
        logging.info("从集中管理器加载JWT令牌")
        return jwt_token
    else:
        logging.warning("未找到有效的JWT令牌配置")
        return None

async def check_jwt_token_validity():
    """异步检查JWT令牌有效性"""
    if not jwt_manager.is_token_configured():
        return False

    # 检查令牌是否过期或无效
    is_valid = await jwt_manager.is_token_valid()
    if not is_valid:
        logging.warning("JWT令牌无效或已过期，需要重新配置")
        # 清除session state中的缓存
        if 'jwt_user_info' in st.session_state:
            del st.session_state.jwt_user_info
        if 'jwt_test_result' in st.session_state:
            del st.session_state.jwt_test_result

    return is_valid

# 检查JWT令牌可用性
JWT_TOKEN = load_jwt_token()
JWT_AVAILABLE = JWT_TOKEN is not None

def create_download_folder():
    """创建下载文件夹"""
    download_dir = Path("downloads")
    download_dir.mkdir(exist_ok=True)
    return download_dir

def save_plot_files(plot_data, plot_type, timestamp):
    """保存图表文件到下载文件夹（PNG和SVG格式）"""
    download_dir = create_download_folder()

    # 解码base64图像数据
    image_data = base64.b64decode(plot_data)

    # 生成文件名
    filename_base = f"{timestamp}_{plot_type}"

    # 保存PNG文件（高质量）
    png_path = download_dir / f"{filename_base}.png"
    with open(png_path, 'wb') as f:
        f.write(image_data)

    # 注意：当前MR服务器只生成PNG格式
    # SVG支持需要在MR服务器端实现

    return png_path

def regenerate_plots_with_formats(mr_data, timestamp):
    """重新生成包含多种格式的图表"""
    try:
        # 这里可以调用MR服务器重新生成SVG格式的图表
        # 目前先返回现有的PNG数据
        return mr_data.get("visualization", {})
    except Exception as e:
        st.error(f"Failed to regenerate chart: {e}")
        return {}

def create_download_zip(visualization_data):
    """创建包含所有图表的ZIP文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    download_dir = create_download_folder()

    # 保存所有图表文件
    saved_files = []

    plot_types = {
        'scatter_plot': 'scatter',
        'forest_plot': 'forest',
        'funnel_plot': 'funnel'
    }

    for plot_key, plot_name in plot_types.items():
        if plot_key in visualization_data and visualization_data[plot_key]:
            try:
                file_path = save_plot_files(visualization_data[plot_key], plot_name, timestamp)
                saved_files.append(file_path)
            except Exception as e:
                st.error(f"Failed to save {plot_name} plot: {e}")

    # 创建ZIP文件
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for file_path in saved_files:
            zip_file.write(file_path, file_path.name)

    zip_buffer.seek(0)
    return zip_buffer.getvalue(), f"mr_analysis_plots_{timestamp}.zip"

# 设置环境变量，确保MCP服务器能够访问
if JWT_AVAILABLE and JWT_TOKEN:
    os.environ['OPENGWAS_JWT'] = JWT_TOKEN
    logging.info(f"JWT令牌已设置到环境变量 (长度: {len(JWT_TOKEN)} 字符)")

    # 验证令牌设置是否成功
    if os.environ.get('OPENGWAS_JWT') == JWT_TOKEN:
        logging.info("✅ 环境变量设置验证成功")
    else:
        logging.error("❌ 环境变量设置验证失败")
else:
    logging.warning("⚠️ JWT令牌未配置，需要用户配置后才能访问GWAS数据")

# Import our modules - use absolute imports to avoid conflicts
try:
    from models import CausalAnalysisRequest, AnalysisType
except ImportError as e:
    st.error(f"❌ Failed to import models: {e}")
    st.error("Please ensure you are running the application in the correct directory")
    st.stop()

from causal_analyzer import CausalAnalyzer
from i18n import get_text, get_language_options
from input_validator import InputValidator
from input_validator import get_validator
from natural_language_parser import NaturalLanguageParser
from llm_workflow_coordinator import LLMWorkflowCoordinator, WorkflowState

# Page configuration
st.set_page_config(
    page_title="Causal-MCP: Interactive Causal Inference Framework",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'analyzer' not in st.session_state:
    st.session_state.analyzer = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'system_status' not in st.session_state:
    st.session_state.system_status = None
if 'language' not in st.session_state:
    st.session_state.language = "en"
if 'initialization_attempted' not in st.session_state:
    st.session_state.initialization_attempted = False
if 'initialization_in_progress' not in st.session_state:
    st.session_state.initialization_in_progress = False
if 'validator' not in st.session_state:
    st.session_state.validator = InputValidator()
if 'show_llm_setup' not in st.session_state:
    st.session_state.show_llm_setup = False
if 'workflow_coordinator' not in st.session_state:
    st.session_state.workflow_coordinator = None
if 'use_natural_language' not in st.session_state:
    st.session_state.use_natural_language = True
if 'workflow_state' not in st.session_state:
    st.session_state.workflow_state = None

async def initialize_analyzer():
    """Initialize the causal analyzer."""
    if st.session_state.analyzer is None:
        analyzer = CausalAnalyzer()
        success = await analyzer.initialize()
        if success:
            st.session_state.analyzer = analyzer
            # 初始化工作流协调器
            if st.session_state.workflow_coordinator is None:
                st.session_state.workflow_coordinator = LLMWorkflowCoordinator(
                    causal_analyzer=analyzer,
                    llm_service=analyzer.llm_service,
                    input_validator=st.session_state.validator
                )
            return True
        else:
            st.error(get_text("failed_to_initialize", "en"))  # 使用英文，因为这是系统级错误
            return False
    return True

def _is_first_time_llm_setup(llm_config: dict, llm_status: dict) -> bool:
    """
    检查是否是首次使用，需要进行 LLM 设置

    Args:
        llm_config: LLM 配置字典
        llm_status: LLM 状态字典

    Returns:
        bool: 是否需要首次设置
    """
    # 如果用户已经完成过设置，不再显示
    if st.session_state.get('llm_setup_completed', False):
        return False

    # 如果用户通过左侧 LLM 设置配置过，也不再显示首页设置
    if st.session_state.get('llm_configured_via_sidebar', False):
        st.session_state.llm_setup_completed = True
        return False

    # 🔑 关键改进：首先检查LLM服务是否实际可用
    # 如果LLM服务已经正常运行，不需要显示配置界面
    if llm_status.get('available', False) and llm_status.get('enabled', False):
        # LLM服务正常运行，标记为已完成设置
        st.session_state.llm_setup_completed = True
        # 确保隐藏配置界面
        st.session_state.show_llm_setup = False
        return False

    # 如果 LLM 已禁用，检查是否是用户主动禁用的
    if not llm_config.get('enabled', True):
        # 如果用户主动选择了基础分析模式，标记为已完成设置
        if st.session_state.get('user_chose_basic_analysis', False):
            st.session_state.llm_setup_completed = True
            return False
        # 否则仍然显示设置界面，让用户选择
        return True

    provider = llm_config.get('provider', 'ollama')

    if provider == 'ollama':
        # 对于 Ollama，检查是否有有效的模型配置
        ollama_config = llm_config.get('ollama', {})
        has_model_config = ollama_config.get('model_name') and ollama_config.get('model_name') != 'deepseek-r1:1.5b'

        if has_model_config:
            # 有模型配置，但需要检查服务是否真正可用
            # 如果配置存在但服务不可用，仍需要显示配置界面
            if not llm_status.get('available', False):
                return True

            # 配置存在且服务可用，标记为已完成设置
            st.session_state.llm_setup_completed = True
            return False

        return True

    elif provider == 'deepseek':
        # 对于 DeepSeek，检查是否有 API 密钥
        deepseek_config = llm_config.get('deepseek', {})
        has_api_key = bool(deepseek_config.get('api_key'))

        if has_api_key:
            # 有 API 密钥，但需要检查服务是否真正可用
            # 如果配置存在但服务不可用，仍需要显示配置界面
            if not llm_status.get('available', False):
                return True

            # 配置存在且服务可用，标记为已完成设置
            st.session_state.llm_setup_completed = True
            return False

        return True

    return True

def render_jwt_setup_page():
    """渲染JWT配置页面"""
    from jwt_config_ui import render_jwt_setup_guide, render_jwt_config_form

    # 获取语言设置
    lang = st.session_state.get('language', 'zh')

    # 页面标题
    st.title(get_text("jwt_token_config", lang))
    st.markdown("---")
    st.warning(get_text("jwt_token_required", lang))

    # 创建标签页
    tab1, tab2 = st.tabs([get_text("get_jwt_token", lang), get_text("configure_jwt_token", lang)])

    with tab1:
        render_jwt_setup_guide(lang)

    with tab2:
        configured_token = render_jwt_config_form(lang)

        if configured_token:
            # 配置成功，重新加载页面
            st.success(get_text("jwt_config_success", lang))

            # 更新全局变量
            global JWT_AVAILABLE, JWT_TOKEN
            JWT_AVAILABLE = True
            JWT_TOKEN = configured_token

            # 设置环境变量
            os.environ['OPENGWAS_JWT'] = JWT_TOKEN

            # 清除JWT过期状态和相关缓存
            st.session_state.jwt_expired = False
            st.session_state.jwt_validity_checked = False  # 重新检查有效性
            if 'jwt_user_info' in st.session_state:
                del st.session_state.jwt_user_info
            if 'jwt_test_result' in st.session_state:
                del st.session_state.jwt_test_result

            # 等待一下然后刷新
            import time
            time.sleep(2)
            st.rerun()

def main():
    """Main application function."""

    # Initialize validator
    if 'validator' not in st.session_state:
        st.session_state.validator = get_validator()

    # Get current language first
    lang = st.session_state.language

    # Check JWT configuration and validity
    if not JWT_AVAILABLE:
        show_jwt_setup_page()
        return

    # 异步检查JWT令牌有效性（仅在首次访问时检查）
    if 'jwt_validity_checked' not in st.session_state:
        st.session_state.jwt_validity_checked = True
        with st.spinner("验证JWT令牌有效性..."):
            jwt_valid = asyncio.run(check_jwt_token_validity())
            if not jwt_valid:
                st.error("🔴 JWT令牌已过期或无效，请重新配置")
                st.session_state.jwt_expired = True
                show_jwt_setup_page()
                return
            else:
                st.session_state.jwt_expired = False

                # 检查令牌是否即将过期
                if jwt_manager.is_token_expiring_soon():
                    days_remaining = jwt_manager.get_token_expiry_days()
                    if days_remaining == 0:
                        st.error("⚠️ JWT令牌今天过期！请尽快重新配置以避免服务中断")
                    elif days_remaining == 1:
                        st.warning("⚠️ JWT令牌明天过期，建议提前重新配置")
                    else:
                        st.warning(f"⚠️ JWT令牌将在{days_remaining}天后过期，建议提前重新配置")

    # 如果之前检测到JWT过期，继续显示配置页面
    if st.session_state.get('jwt_expired', False):
        show_jwt_setup_page()
        return

    # Auto-initialize system on first visit
    if not st.session_state.initialization_attempted and not st.session_state.initialization_in_progress:
        st.session_state.initialization_in_progress = True
        with st.spinner(get_text("auto_initializing_system", lang)):
            success = asyncio.run(initialize_analyzer())
            if success:
                # Get system status
                status = asyncio.run(st.session_state.analyzer.get_system_status())
                st.session_state.system_status = status
                st.success(get_text("system_init_success", lang))

                # Check if LLM needs first-time setup
                llm_service = st.session_state.analyzer.get_llm_service()

                # 🔑 关键改进：实际测试LLM连接状态，而不仅仅是读取配置
                try:
                    # 测试LLM连接以获取真实状态
                    connection_success, _, _ = asyncio.run(llm_service.test_connection())

                    # 获取更新后的状态
                    llm_status = llm_service.get_status()
                    llm_config = llm_service.get_config()

                    # 如果连接测试成功，更新状态
                    if connection_success:
                        llm_status['available'] = True

                except Exception as e:
                    # 如果测试失败，获取基本状态
                    llm_status = llm_service.get_status()
                    llm_config = llm_service.get_config()

                # Check if this is first-time use (no proper LLM configuration)
                # 但只有在用户没有明确完成设置时才检查
                if not st.session_state.get('llm_setup_completed', False):
                    is_first_time = _is_first_time_llm_setup(llm_config, llm_status)

                    if is_first_time:
                        st.session_state.show_llm_setup = True
                        st.session_state.show_llm_config = False  # 隐藏侧边栏配置
                    else:
                        # 如果不是首次使用，确保设置界面隐藏
                        st.session_state.show_llm_setup = False
            else:
                st.error(get_text("system_init_failed", lang))
        st.session_state.initialization_attempted = True
        st.session_state.initialization_in_progress = False
        st.rerun()  # Refresh the page to show the updated state

    # Language selector in the top right
    col1, col2 = st.columns([4, 1])
    with col2:
        language_options = get_language_options()
        current_lang_display = [k for k, v in language_options.items() if v == st.session_state.language][0]
        selected_lang_display = st.selectbox(
            get_text("language_selector", lang),
            options=list(language_options.keys()),
            index=list(language_options.keys()).index(current_lang_display),
            key="language_selector"
        )
        st.session_state.language = language_options[selected_lang_display]
        # Update lang if language changed
        lang = st.session_state.language

    # Header
    with col1:
        st.title(get_text("page_title", lang))

    st.markdown(get_text("page_description", lang))
    
    # Sidebar for system status and controls
    with st.sidebar:
        st.header(get_text("system_control", lang))

        # Show initialization status
        if st.session_state.analyzer is not None:
            st.success(get_text("system_ready", lang))
            # Re-initialize system button (optional)
            if st.button(get_text("reinitialize_system", lang), help=get_text("reinitialize_help", lang)):
                with st.spinner(get_text("reinitializing", lang)):
                    # Reset analyzer
                    st.session_state.analyzer = None
                    st.session_state.system_status = None
                    st.session_state.initialization_attempted = False
                    success = asyncio.run(initialize_analyzer())
                    if success:
                        st.success(get_text("reinit_success", lang))
                        # Get system status
                        status = asyncio.run(st.session_state.analyzer.get_system_status())
                        st.session_state.system_status = status
        else:
            if st.session_state.initialization_attempted:
                st.error(get_text("init_failed", lang))
                # Manual retry button
                if st.button(get_text("retry_initialization", lang), type="primary"):
                    st.session_state.initialization_attempted = False
                    st.rerun()
            else:
                st.info(get_text("initializing_auto", lang))

        # System status
        st.header(get_text("system_status", lang))
        if st.session_state.system_status:
            status = st.session_state.system_status
            
            # Overall health indicator
            health_color = {
                "healthy": "🟢",
                "degraded": "🟡",
                "down": "🔴"
            }
            health_text = {
                "healthy": get_text("status_online", lang),
                "degraded": "degraded",
                "down": get_text("status_offline", lang)
            }
            st.markdown(f"**{get_text('overall_health', lang)}:** {health_color.get(status.system_health, '⚪')} {health_text.get(status.system_health, status.system_health)}")

            # Server status details (排除 LLM Service，因为在下面单独显示)
            for server in status.servers:
                if server.server_name != "LLM Service":  # 跳过 LLM Service，避免重复显示
                    status_icon = "🟢" if server.status == "online" else "🔴"
                    status_text_display = get_text(f"status_{server.status}", lang) if server.status in ["online", "offline", "error"] else server.status
                    st.markdown(f"{status_icon} **{server.server_name}**: {status_text_display}")
        else:
            st.info(get_text("click_initialize", lang))

        # JWT配置界面
        st.markdown("**🔑 JWT Configuration**")
        from jwt_config_sidebar import render_jwt_config_section
        show_jwt_config = render_jwt_config_section(lang)

        # 将 show_jwt_config 状态传递给主页面
        if show_jwt_config:
            st.session_state.show_main_jwt_config = True

        st.markdown("---")

        # 简化的LLM配置界面
        if st.session_state.analyzer is not None:
            from llm_config_sidebar import render_llm_config_section

            llm_service = st.session_state.analyzer.get_llm_service()
            show_main_config = render_llm_config_section(llm_service, lang)

            # 将 show_main_config 状态传递给主页面
            if show_main_config:
                st.session_state.show_main_llm_config = True
        else:
            st.markdown("**🤖 LLM Configuration**")
            st.info(get_text("please_init_system_first", lang))

    # 🔄 动态检查LLM状态：如果用户在应用运行期间启动了LLM服务，自动隐藏配置界面
    if st.session_state.get('show_llm_setup', False) and hasattr(st.session_state, 'analyzer'):
        try:
            llm_service = st.session_state.analyzer.get_llm_service()
            current_status = llm_service.get_status()

            # 如果LLM现在可用了，隐藏配置界面
            if current_status.get('available', False) and current_status.get('enabled', False):
                st.session_state.show_llm_setup = False
                st.session_state.llm_setup_completed = True
                st.rerun()  # 刷新页面以隐藏配置界面
        except Exception:
            pass  # 如果检查失败，继续显示配置界面

    # Check if LLM first-time setup is needed
    if st.session_state.get('show_llm_setup', False) and not st.session_state.get('llm_setup_completed', False):
        from llm_first_setup import render_llm_first_time_setup

        # 显示首次设置界面
        with st.container():
            st.info("🎯 " + get_text("first_time_llm_setup", lang))
            setup_result = render_llm_first_time_setup(lang)

            if setup_result is not None:
                # Apply the setup configuration
                llm_service = st.session_state.analyzer.get_llm_service()
                success, message = llm_service.update_config(setup_result)

                if success:
                    # Mark setup as completed and hide all config interfaces
                    st.session_state.llm_setup_completed = True
                    st.session_state.show_llm_setup = False
                    st.session_state.show_llm_config = False  # 确保侧边栏配置也隐藏

                    # Show completion message
                    st.success(get_text("llm_config_completed", lang))
                    st.balloons()  # 庆祝动画
                    st.info("🎉 LLM configuration completed! Interface will refresh automatically...")

                    # Reinitialize LLM in background
                    with st.spinner(get_text("initializing_llm", lang)):
                        llm_success = asyncio.run(st.session_state.analyzer.reinitialize_llm())
                        if llm_success:
                            st.success(get_text("llm_service_started", lang))
                        else:
                            st.info(get_text("llm_service_failed", lang))

                    # Update system status
                    status = asyncio.run(st.session_state.analyzer.get_system_status())
                    st.session_state.system_status = status

                    # Add a brief delay to let user see the success message
                    import time
                    time.sleep(1.5)

                    # Clear the setup interface and refresh immediately
                    st.rerun()
                else:
                    st.error(f"{get_text('config_save_failed', lang)}: {message}")

        st.divider()

    # 检查是否需要显示主页面JWT配置
    if st.session_state.get('show_main_jwt_config', False):
        from jwt_config_sidebar import render_jwt_config_modal
        render_jwt_config_modal(lang)

    # 检查是否需要显示主页面LLM配置
    if st.session_state.get('show_main_llm_config', False):
        from llm_config_main import render_main_llm_config

        if st.session_state.analyzer:
            llm_service = st.session_state.analyzer.get_llm_service()
            config_changed = render_main_llm_config(llm_service, lang)

            # 如果配置有变更，重新初始化LLM
            if config_changed:
                with st.spinner(get_text("applying_new_config", lang)):
                    success = asyncio.run(st.session_state.analyzer.reinitialize_llm())
                    if success:
                        # 更新系统状态
                        status = asyncio.run(st.session_state.analyzer.get_system_status())
                        st.session_state.system_status = status

        # 如果显示了LLM配置，就不显示分析配置
        return

    # Main content area
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header(get_text("analysis_config", lang))

        # 界面模式切换
        interface_mode = st.radio(
            "选择输入方式" if lang == "zh" else "Choose Input Method",
            ["🤖 智能对话模式" if lang == "zh" else "🤖 Smart Chat Mode",
             "📝 传统表单模式" if lang == "zh" else "📝 Traditional Form Mode"],
            index=0 if st.session_state.use_natural_language else 1,
            help="智能对话模式：直接用自然语言描述您的分析需求\n传统表单模式：分别输入基因、疾病、组织" if lang == "zh" else "Smart Chat Mode: Describe your analysis needs in natural language\nTraditional Form Mode: Enter gene, disease, and tissue separately"
        )

        # 更新session state
        st.session_state.use_natural_language = "智能对话" in interface_mode or "Smart Chat" in interface_mode

        if st.session_state.use_natural_language:
            # 自然语言输入界面
            st.subheader("🤖 " + ("智能因果推断分析" if lang == "zh" else "Intelligent Causal Inference Analysis"))

            # 显示使用示例
            with st.expander("💡 Usage Examples"):
                st.markdown("""
                **Example inputs:**
                - "Analyze the causal relationship between PCSK9 gene and coronary heart disease"
                - "Study the effect of IL6R gene on coronary heart disease"
                - "Explore the association between SORT1 gene and type 2 diabetes"
                - "Does LPA gene cause coronary heart disease?"

                **The system will automatically identify:**
                - 🧬 Gene names (e.g., PCSK9, IL6R, SORT1, LPA, etc.)
                - 🏥 Disease names (e.g., coronary heart disease, type 2 diabetes, cardiovascular disease, etc.)
                - 🧪 Tissue types (e.g., whole blood, liver, brain, etc., optional)
                """)

            # 自然语言输入框
            user_input = st.text_area(
                "请描述您的分析需求：" if lang == "zh" else "Please describe your analysis needs:",
                placeholder="例如：分析PCSK9基因与冠心病的因果关系" if lang == "zh" else "e.g., Analyze the causal relationship between PCSK9 gene and coronary heart disease",
                height=100,
                help="用自然语言描述您想要分析的基因-疾病因果关系" if lang == "zh" else "Describe the gene-disease causal relationship you want to analyze in natural language"
            )

            # 智能分析按钮
            if st.button(
                "🚀 " + ("开始智能分析" if lang == "zh" else "Start Intelligent Analysis"),
                type="primary",
                disabled=not user_input.strip(),
                help="点击开始基于您的描述进行因果推断分析" if lang == "zh" else "Click to start causal inference analysis based on your description"
            ):
                # 使用asyncio.run来处理异步函数
                asyncio.run(handle_natural_language_analysis(user_input, lang))

        else:
            # 传统表单界面
            st.subheader(get_text("causal_analysis_params", lang))

        # 只有在传统模式下才显示表单输入
        if not st.session_state.use_natural_language:
            # Gene input with real-time validation (outside form for immediate feedback)
            exposure_gene = st.text_input(
                get_text("exposure_gene", lang),
                value="PCSK9",
                placeholder=get_text("gene_placeholder", lang),
                help=get_text("exposure_gene_help", lang),
                key="gene_input"
            ).upper().strip()

            # Validate gene input in real-time
            if exposure_gene:
                gene_valid, gene_error = st.session_state.validator.validate_gene(exposure_gene, lang)
                if gene_valid:
                    st.success(f"{get_text('gene_symbol_valid', lang)}: {exposure_gene}")
                else:
                    st.info(f"💡 {gene_error}")

            # Outcome input with real-time validation
            outcome_trait = st.text_input(
                get_text("outcome_trait", lang),
                value="Coronary Artery Disease",
                placeholder=get_text("outcome_placeholder", lang),
                help=get_text("outcome_trait_help", lang),
                key="trait_input"
            ).strip()

            # Validate outcome trait in real-time with enhanced feedback
            if outcome_trait:
                trait_valid, trait_message, trait_info = st.session_state.validator.validate_gwas_trait(outcome_trait, lang)

                if trait_info and trait_info.get("match_type") == "exact":
                    # 精确匹配
                    st.success(trait_message)

                elif trait_info and trait_info.get("warning"):
                    # 没有精确匹配但允许继续
                    st.warning(trait_message)

                    # 显示建议选项
                    suggestions = trait_info.get("suggestions", [])
                    if suggestions:
                        st.write("**💡 Recommended Options (Click to Quick Select):**")

                        # 创建建议按钮
                        cols = st.columns(min(len(suggestions), 3))
                        for i, suggestion in enumerate(suggestions[:3]):
                            with cols[i]:
                                if st.button(f"📋 {suggestion}", key=f"trait_suggestion_{i}"):
                                    st.session_state.trait_input = suggestion
                                    st.experimental_rerun()

                        # 如果有更多建议，显示在expander中
                        if len(suggestions) > 3:
                            with st.expander(f"View More Suggestions ({len(suggestions)-3} more)"):
                                for j, suggestion in enumerate(suggestions[3:8], 3):
                                    if st.button(f"📋 {suggestion}", key=f"trait_suggestion_more_{j}"):
                                        st.session_state.trait_input = suggestion
                                        st.experimental_rerun()
                else:
                    # 其他情况
                    st.info(f"💡 {trait_message}")

            # For consistency, set display name same as trait
            outcome_trait_display = outcome_trait

            # Tissue input with real-time validation
            tissue_context = st.text_input(
                get_text("tissue_context", lang),
                value="Whole_Blood",
                placeholder=get_text("tissue_placeholder", lang),
                help=get_text("tissue_context_help", lang),
                key="tissue_input"
            ).strip()

            # Validate tissue context in real-time
            if tissue_context:
                tissue_valid, tissue_error = st.session_state.validator.validate_tissue(tissue_context, lang)
                if tissue_valid:
                    st.success(f"{get_text('tissue_context_valid', lang)}: {tissue_context}")
                else:
                    st.info(f"💡 {tissue_error}")

            # Submit button (always enabled - allow users to run analysis even with validation warnings)
            submitted = st.button(
                get_text("run_analysis", lang),
                type="primary",
                key="submit_analysis"
            )

            if submitted:
                # 首先检查JWT令牌是否仍然有效
                with st.spinner("检查JWT令牌有效性..."):
                    jwt_valid = asyncio.run(check_jwt_token_validity())
                    if not jwt_valid:
                        st.error("🔴 JWT令牌已过期或无效，请重新配置后再进行分析")
                        st.session_state.jwt_expired = True
                        st.session_state.jwt_validity_checked = False
                        st.rerun()
                        return

                # Ensure system is initialized before running analysis
                if st.session_state.analyzer is None:
                    st.info(get_text("auto_initializing_for_analysis", lang))
                    with st.spinner(get_text("initializing", lang)):
                        success = asyncio.run(initialize_analyzer())
                        if success:
                            status = asyncio.run(st.session_state.analyzer.get_system_status())
                            st.session_state.system_status = status
                            st.success(get_text("init_success", lang))
                        else:
                            st.error(get_text("init_failed_analysis", lang))
                            return

                # Run analysis if system is ready
                if st.session_state.analyzer is not None:
                    # Create analysis request with default analysis options
                    request = CausalAnalysisRequest(
                        exposure_gene=exposure_gene,
                        outcome_trait=outcome_trait,
                        tissue_context=tissue_context,
                        analysis_type=AnalysisType.FULL_CAUSAL_ANALYSIS,
                        include_pathway_analysis=True,  # Default to True
                        include_drug_analysis=True,     # Default to True
                        language=lang,
                        show_thinking=st.session_state.get('show_thinking', False)
                    )

                    # Run analysis
                    with st.spinner(get_text("running_analysis", lang).format(exposure_gene, outcome_trait_display)):
                        try:
                            result = asyncio.run(
                                st.session_state.analyzer.perform_full_causal_analysis(request)
                            )
                            st.session_state.analysis_result = result

                            if result.success:
                                st.success(get_text("analysis_success", lang))
                            else:
                                st.error(get_text("analysis_failed", lang))
                                # 显示具体的错误信息
                                if result.error_message:
                                    with st.expander("🔍 Detailed Error Information", expanded=True):
                                        st.error(result.error_message)

                                        # 根据错误类型提供解决建议
                                        if "技术错误" in result.error_message or "语法错误" in result.error_message:
                                            st.info("💡 This is a system technical issue, please contact technical support.")
                                        elif "网络" in result.error_message or "超时" in result.error_message:
                                            st.info("💡 This is a network connection issue, please check your network or try again later.")
                                        elif "认证" in result.error_message or "令牌" in result.error_message:
                                            st.info("💡 This is an API authentication issue, please check your token configuration.")
                                        else:
                                            st.info("💡 Please check the error details or contact technical support for help.")

                        except Exception as e:
                            st.error(get_text("analysis_error", lang))
                            with st.expander("🔍 Detailed Error Information", expanded=True):
                                st.error(str(e))

                                # 分析异常类型并提供建议
                                error_str = str(e)
                                if "技术错误" in error_str or "语法错误" in error_str:
                                    st.info("💡 This is a system technical issue, please contact technical support.")
                                elif "网络" in error_str or "超时" in error_str:
                                    st.info("💡 This is a network connection issue, please check your network or try again later.")
                                elif "认证" in error_str or "令牌" in error_str:
                                    st.info("💡 This is an API authentication issue, please check your token configuration.")
                                elif "No harmonized data" in error_str:
                                    st.info("💡 Unable to obtain GWAS data for analysis, possibly due to network issues or temporary database unavailability.")
                                else:
                                    st.info("💡 Please check the error details or contact technical support for help.")
    
    with col2:
        st.header(get_text("parameter_help", lang))

        # 使用标签页组织帮助内容
        help_tab1, help_tab2, help_tab3 = st.tabs([
            get_text("fill_guide", lang),
            get_text("common_data", lang),
            get_text("tissue_info", lang)
        ])

        with help_tab1:
            st.markdown(get_text("gene_symbol_help", lang))
            st.markdown(get_text("outcome_trait_help", lang))
            st.markdown(get_text("tissue_context_help", lang))

            st.success(get_text("different_tissues_note", lang))

        with help_tab2:
            st.markdown(get_text("common_gene_examples", lang))
            st.markdown(get_text("common_gwas_examples", lang))

        with help_tab3:
            st.markdown(get_text("gtex_tissues", lang))
            st.markdown(get_text("data_source_note", lang))

            st.info(get_text("recommend_whole_blood", lang))

    # Quick Results Section
    st.header(get_text("quick_results", lang))

    if st.session_state.analysis_result:
        result = st.session_state.analysis_result

        # Summary metrics
        col2a, col2b, col2c = st.columns(3)

        with col2a:
            st.metric(
                get_text("instruments_found", lang),
                result.summary.get("n_instruments", 0)
            )

        with col2b:
            st.metric(
                get_text("snps_harmonized", lang),
                result.summary.get("n_harmonized_snps", 0)
            )

        with col2c:
            execution_time = result.total_execution_time
            st.metric(
                get_text("execution_time", lang),
                f"{execution_time:.1f}s"
            )

        # Causal conclusion
        if result.mr_results:
            conclusion = result.summary.get("causal_conclusion", "No conclusion available")
            conclusion_lower = conclusion.lower()

            # 检查是否有因果效应（支持中英文，先检查方向再检查显著性）
            has_negative_effect = any(keyword in conclusion_lower for keyword in [
                "negative", "负向", "inverse"
            ])
            has_positive_effect = any(keyword in conclusion_lower for keyword in [
                "positive", "正向"
            ])
            has_significant_effect = any(keyword in conclusion_lower for keyword in [
                "strong evidence", "强有力的证据", "支持"
            ])

            if has_negative_effect and has_significant_effect:
                st.info(get_text("causal_effect_found", lang).format(conclusion))
            elif has_positive_effect and has_significant_effect:
                st.success(get_text("causal_effect_found", lang).format(conclusion))
            else:
                st.warning(get_text("no_significant_effect", lang).format(conclusion))
    else:
        st.info(get_text("run_analysis_to_see", lang))

    # Results tabs
    if st.session_state.analysis_result:
        st.header(get_text("detailed_results", lang))

        result = st.session_state.analysis_result

        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            get_text("summary_tab", lang),
            get_text("mr_results_tab", lang),
            get_text("biological_context_tab", lang),
            get_text("drug_targets_tab", lang),
            get_text("analysis_steps_tab", lang)
        ])
        
        with tab1:
            st.subheader(get_text("analysis_summary", lang))

            # Key findings
            col1, col2 = st.columns([2, 1])

            with col1:
                st.markdown(get_text("key_findings", lang))
                st.markdown(result.interpretation)

                st.markdown(get_text("recommendations", lang))
                for i, rec in enumerate(result.recommendations, 1):
                    st.markdown(f"{i}. {rec}")

            with col2:
                st.markdown(get_text("analysis_overview", lang))
                summary_data = {
                    "Parameter": [
                        get_text("exposure_gene", lang),
                        get_text("outcome_trait", lang),
                        get_text("tissue_context", lang),
                        get_text("instruments_found", lang),
                        get_text("snps_harmonized", lang)
                    ],
                    "Value": [
                        result.summary.get("exposure_gene", "N/A"),
                        result.summary.get("outcome_trait", "N/A"),
                        result.summary.get("tissue_context", "N/A"),
                        result.summary.get("n_instruments", 0),
                        result.summary.get("n_harmonized_snps", 0)
                    ]
                }
                summary_df = pd.DataFrame(summary_data)
                # 确保数据类型一致，避免Arrow序列化错误
                for col in summary_df.columns:
                    summary_df[col] = summary_df[col].astype(str)
                st.dataframe(summary_df, hide_index=True)
        
        with tab2:
            st.subheader(get_text("mr_results", lang))

            if result.mr_results:
                mr_data = result.mr_results

                # 检查是否有错误
                if "error" in mr_data:
                    st.error(f"{get_text('mr_analysis_failed', lang)}: {mr_data['error']}")
                    st.info(get_text("check_input_retry", lang))
                else:
                    # 显示数据来源
                    data_source = mr_data.get("summary", {}).get("data_source", "Unknown")
                    if data_source == "Real_MR_Calculation":
                        st.success(get_text("real_mr_calculation", lang))

                    # MR method results
                    methods_df = None
                    if "results" in mr_data and mr_data["results"]:
                        st.markdown(get_text("mr_method_results", lang))

                        # 格式化P值为科学计数法
                        formatted_results = []
                        for mr_result_item in mr_data["results"]:
                            formatted_result = mr_result_item.copy()
                            # 格式化P值
                            p_val = mr_result_item.get("p_value", 0)
                            if isinstance(p_val, (int, float)):
                                if p_val == 0.0:
                                    formatted_result["p_value"] = "< 1.0e-100"
                                elif p_val < 1e-10:
                                    formatted_result["p_value"] = f"{p_val:.2e}"
                                elif p_val < 0.001:
                                    formatted_result["p_value"] = f"{p_val:.2e}"
                                else:
                                    formatted_result["p_value"] = f"{p_val:.4f}"
                            formatted_results.append(formatted_result)

                        methods_df = pd.DataFrame(formatted_results)
                        # 确保数据类型一致
                        for col in methods_df.columns:
                            if methods_df[col].dtype == 'object':
                                methods_df[col] = methods_df[col].astype(str)
                        st.dataframe(methods_df, hide_index=True)
                    else:
                        st.warning(get_text("no_mr_method_results", lang))

                    # 显示MR服务器生成的可视化图表
                    if "visualization" in mr_data and mr_data["visualization"]:
                        visualization = mr_data["visualization"]

                        # 创建标题和下载按钮
                        col_title, col_download = st.columns([3, 1])
                        with col_title:
                            st.markdown("### 📊 " + get_text("mr_visualizations", lang))
                        with col_download:
                            # 创建下载所有图表的按钮
                            if st.button("📥 Download All Charts", key="download_all_plots"):
                                try:
                                    zip_data, zip_filename = create_download_zip(visualization)
                                    st.download_button(
                                        label="💾 Click to Download ZIP File",
                                        data=zip_data,
                                        file_name=zip_filename,
                                        mime="application/zip",
                                        key="download_zip_button"
                                    )
                                    st.success(f"✅ Charts saved to png/ folder")
                                except Exception as e:
                                    st.error(f"❌ Download failed: {e}")

                        # 散点图
                        if visualization.get("scatter_plot"):
                            # 标题和下载按钮
                            col_title, col_btn = st.columns([4, 1])
                            with col_title:
                                st.markdown("#### " + get_text("scatter_plot_title", lang))
                            with col_btn:
                                if st.button("📥", key="download_scatter", help="Download Scatter Plot"):
                                    try:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        file_path = save_plot_files(visualization["scatter_plot"], "scatter", timestamp)
                                        st.success(f"✅ Scatter plot saved: {file_path.name}")
                                    except Exception as e:
                                        st.error(f"❌ Download failed: {e}")

                            try:
                                scatter_image = base64.b64decode(visualization["scatter_plot"])

                                # 使用列布局来控制图片大小
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.image(scatter_image,
                                            caption=get_text("scatter_plot_caption", lang),
                                            use_container_width=True)
                            except Exception as e:
                                st.error(f"Scatter plot display failed: {e}")

                        # 森林图单独一行 - 与散点图保持一致的布局
                        if visualization.get("forest_plot"):
                            # 标题和下载按钮
                            col_title, col_btn = st.columns([4, 1])
                            with col_title:
                                st.markdown("#### " + get_text("forest_plot_title", lang))
                            with col_btn:
                                if st.button("📥", key="download_forest", help="Download Forest Plot"):
                                    try:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        file_path = save_plot_files(visualization["forest_plot"], "forest", timestamp)
                                        st.success(f"✅ Forest plot saved: {file_path.name}")
                                    except Exception as e:
                                        st.error(f"❌ Download failed: {e}")

                            try:
                                forest_image = base64.b64decode(visualization["forest_plot"])

                                # 使用与散点图相同的列布局
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.image(forest_image,
                                            caption=get_text("forest_plot_caption", lang),
                                            use_container_width=True)
                            except Exception as e:
                                st.error(f"Forest plot display failed: {e}")

                        # 漏斗图单独一行 - 与散点图保持一致的布局
                        if visualization.get("funnel_plot"):
                            # 标题和下载按钮
                            col_title, col_btn = st.columns([4, 1])
                            with col_title:
                                st.markdown("#### " + get_text("funnel_plot_title", lang))
                            with col_btn:
                                if st.button("📥", key="download_funnel", help="Download Funnel Plot"):
                                    try:
                                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                        file_path = save_plot_files(visualization["funnel_plot"], "funnel", timestamp)
                                        st.success(f"✅ Funnel plot saved: {file_path.name}")
                                    except Exception as e:
                                        st.error(f"❌ Download failed: {e}")

                            try:
                                funnel_image = base64.b64decode(visualization["funnel_plot"])

                                # 使用与散点图相同的列布局
                                col1, col2, col3 = st.columns([1, 2, 1])
                                with col2:
                                    st.image(funnel_image,
                                            caption=get_text("funnel_plot_caption", lang),
                                            use_container_width=True)
                            except Exception as e:
                                st.error(f"Funnel plot display failed: {e}")

                    # 如果没有可视化数据，显示备用的Plotly森林图
                    elif methods_df is not None and len(methods_df) > 0:
                        st.markdown("#### " + get_text("forest_plot_title", lang) + " (Backup)")
                        fig = go.Figure()

                        for _, row in methods_df.iterrows():
                            fig.add_trace(go.Scatter(
                                x=[row['estimate']],
                                y=[row['method']],
                                error_x=dict(
                                    type='data',
                                    symmetric=False,
                                    array=[row['ci_upper'] - row['estimate']],
                                    arrayminus=[row['estimate'] - row['ci_lower']]
                                ),
                                mode='markers',
                                marker=dict(size=10),
                                name=row['method']
                            ))

                        fig.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig.update_layout(
                            title=get_text("forest_plot_title", lang),
                            xaxis_title=get_text("causal_effect_estimate", lang),
                            yaxis_title=get_text("mr_method", lang),
                            showlegend=False
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Sensitivity analysis
                if "sensitivity_analysis" in mr_data:
                    st.markdown(get_text("sensitivity_analysis", lang))
                    sens_data = mr_data["sensitivity_analysis"]

                    col1, col2 = st.columns(2)

                    with col1:
                        if "heterogeneity_test" in sens_data:
                            het = sens_data["heterogeneity_test"]
                            st.markdown(get_text("heterogeneity_test", lang))

                            # 格式化异质性检验P值
                            het_p = het.get('p_value', 'N/A')
                            if isinstance(het_p, (int, float)):
                                if het_p == 0.0:
                                    het_p_formatted = "< 1.0e-100"
                                elif het_p < 1e-10:
                                    het_p_formatted = f"{het_p:.2e}"
                                elif het_p < 0.001:
                                    het_p_formatted = f"{het_p:.2e}"
                                else:
                                    het_p_formatted = f"{het_p:.4f}"
                            else:
                                het_p_formatted = str(het_p)

                            st.markdown(f"- {get_text('p_value', lang)}: {het_p_formatted}")
                            st.markdown(f"- {get_text('interpretation', lang)}: {het.get('interpretation', 'N/A')}")

                    with col2:
                        if "pleiotropy_test" in sens_data:
                            pleio = sens_data["pleiotropy_test"]
                            st.markdown(get_text("pleiotropy_test", lang))

                            # 格式化多效性检验P值
                            pleio_p = pleio.get('p_value', 'N/A')
                            if isinstance(pleio_p, (int, float)):
                                if pleio_p == 0.0:
                                    pleio_p_formatted = "< 1.0e-100"
                                elif pleio_p < 1e-10:
                                    pleio_p_formatted = f"{pleio_p:.2e}"
                                elif pleio_p < 0.001:
                                    pleio_p_formatted = f"{pleio_p:.2e}"
                                else:
                                    pleio_p_formatted = f"{pleio_p:.4f}"
                            else:
                                pleio_p_formatted = str(pleio_p)

                            st.markdown(f"- {get_text('p_value', lang)}: {pleio_p_formatted}")
                            st.markdown(f"- {get_text('interpretation', lang)}: {pleio.get('interpretation', 'N/A')}")
            else:
                st.warning(get_text("no_mr_results", lang))
        
        with tab3:
            st.subheader(get_text("biological_context", lang))

            if result.gene_annotation:
                annotation = result.gene_annotation

                # 检查是否有错误
                if "error" in annotation:
                    error_type = annotation.get("data_source", "Unknown_Error")
                    error_message = annotation.get("message", annotation.get("error", "Unknown error"))

                    if error_type == "Real_API_No_Data":
                        st.warning(f"{get_text('real_data_unavailable', lang)}: {error_message}")
                        st.info(get_text("real_bio_data_limited", lang))
                    elif error_type == "Service_Error":
                        st.error(f"{get_text('service_error', lang)}: {error_message}")
                        st.info(get_text("check_network_retry", lang))
                    else:
                        st.error(f"{get_text('unknown_error', lang)}: {error_message}")
                else:
                    # 显示真实数据
                    data_source = annotation.get("data_source", "Unknown")
                    if data_source.startswith("Real_"):
                        st.success(get_text("real_bio_database", lang))

                    # Gene information
                    if "gene_info" in annotation:
                        st.markdown(get_text("gene_information", lang))
                        gene_info = annotation["gene_info"]
                        st.json(gene_info)

                    # Protein interactions
                    if "protein_interactions" in annotation:
                        interactions = annotation["protein_interactions"]
                        if interactions:
                            st.markdown(get_text("protein_interactions", lang))
                            interactions_df = pd.DataFrame(interactions)
                            # 确保数据类型一致
                            for col in interactions_df.columns:
                                if interactions_df[col].dtype == 'object':
                                    interactions_df[col] = interactions_df[col].astype(str)
                            st.dataframe(interactions_df, hide_index=True)

                    # Disease associations
                    if "disease_associations" in annotation:
                        diseases = annotation["disease_associations"]
                        if diseases:
                            st.markdown(get_text("disease_associations", lang))
                            diseases_df = pd.DataFrame(diseases)
                            # 确保数据类型一致
                            for col in diseases_df.columns:
                                if diseases_df[col].dtype == 'object':
                                    diseases_df[col] = diseases_df[col].astype(str)
                            st.dataframe(diseases_df, hide_index=True)
            else:
                st.warning(get_text("bio_context_not_performed", lang))
        
        with tab4:
            st.subheader(get_text("drug_target_analysis", lang))

            if result.drug_analysis:
                drug_data = result.drug_analysis

                # 检查是否有错误
                if "error" in drug_data:
                    error_type = drug_data.get("data_source", "Unknown_Error")
                    error_message = drug_data.get("message", drug_data.get("error", "Unknown error"))

                    if error_type == "Real_API_No_Data":
                        st.warning(f"{get_text('real_drug_data_unavailable', lang)}: {error_message}")
                        st.info(get_text("real_drug_database_no_info", lang))
                    elif error_type == "Service_Error":
                        st.error(f"{get_text('service_error', lang)}: {error_message}")
                        st.info(get_text("check_network_retry", lang))
                    else:
                        st.error(f"{get_text('unknown_error', lang)}: {error_message}")
                else:
                    # 显示真实数据
                    data_source = drug_data.get("data_source", "Unknown")
                    if data_source.startswith("Real_"):
                        st.success(get_text("real_drug_database", lang))

                    # Targeting drugs - 支持多种数据结构
                    drugs = None
                    if "targeting_drugs" in drug_data:
                        drugs = drug_data["targeting_drugs"]
                    elif "drug_targets" in drug_data:
                        drugs = drug_data["drug_targets"]

                    if drugs:
                        st.markdown(get_text("targeting_drugs", lang))
                        drugs_df = pd.DataFrame(drugs)
                        # 确保数据类型一致
                        for col in drugs_df.columns:
                            if drugs_df[col].dtype == 'object':
                                drugs_df[col] = drugs_df[col].astype(str)
                        st.dataframe(drugs_df, hide_index=True)
                    else:
                        st.info(get_text("no_drug_target_info", lang))

                    # Druggability assessment
                    if "druggability_assessment" in drug_data:
                        assessment = drug_data["druggability_assessment"]
                        st.markdown(get_text("druggability_assessment", lang))

                        col1, col2 = st.columns(2)
                        with col1:
                            score = assessment.get("druggability_score", 0)
                            st.metric(get_text("druggability_score", lang), f"{score:.2f}")

                        with col2:
                            level = assessment.get("assessment_level", "Unknown")
                            st.metric(get_text("assessment_level", lang), level)

                        opportunities = drug_data.get("therapeutic_opportunities", "")
                        if opportunities:
                            st.markdown(get_text("therapeutic_opportunities", lang))
                            st.markdown(opportunities)
            else:
                st.warning(get_text("drug_analysis_not_performed", lang))
        
        with tab5:
            st.subheader(get_text("analysis_workflow", lang))

            # Analysis steps timeline
            steps_data = []
            for step in result.analysis_steps:
                steps_data.append({
                    get_text("step", lang): step.step_name,
                    get_text("status", lang): step.status,
                    get_text("server", lang): step.server_used or "N/A",
                    get_text("execution_time", lang): f"{step.execution_time:.2f}s" if step.execution_time else "N/A",
                    get_text("error", lang): step.error_message or get_text("none", lang)
                })

            steps_df = pd.DataFrame(steps_data)
            # 确保数据类型一致
            for col in steps_df.columns:
                if steps_df[col].dtype == 'object':
                    steps_df[col] = steps_df[col].astype(str)
            st.dataframe(steps_df, hide_index=True)

            # Execution timeline
            if any(step.execution_time for step in result.analysis_steps):
                fig = go.Figure()

                step_names = [step.step_name for step in result.analysis_steps if step.execution_time]
                exec_times = [step.execution_time for step in result.analysis_steps if step.execution_time]

                fig.add_trace(go.Bar(
                    x=step_names,
                    y=exec_times,
                    name=get_text("execution_time", lang)
                ))

                fig.update_layout(
                    title=get_text("step_execution_times", lang),
                    xaxis_title=get_text("analysis_step", lang),
                    yaxis_title=get_text("time_seconds", lang)
                )
                st.plotly_chart(fig, use_container_width=True)

async def handle_natural_language_analysis(user_input: str, lang: str):
    """处理自然语言输入的分析请求"""

    # 确保系统已初始化
    if st.session_state.analyzer is None:
        st.info(get_text("auto_initializing_for_analysis", lang))
        with st.spinner(get_text("initializing", lang)):
            success = asyncio.run(initialize_analyzer())
            if not success:
                st.error(get_text("init_failed_analysis", lang))
                return

    # 确保工作流协调器已初始化
    if st.session_state.workflow_coordinator is None:
        st.session_state.workflow_coordinator = LLMWorkflowCoordinator(
            causal_analyzer=st.session_state.analyzer,
            llm_service=st.session_state.analyzer.llm_service,
            input_validator=st.session_state.validator
        )

    # 设置进度回调
    progress_container = st.empty()
    progress_bar = st.progress(0)
    status_container = st.empty()

    async def progress_callback(state: WorkflowState):
        """进度回调函数"""
        progress_bar.progress(state.progress)
        status_container.info(f"📊 {state.message}")

        # 如果需要澄清，显示澄清信息
        if state.stage.value == "parameter_validation" and state.data.get("clarification"):
            st.warning(state.data["clarification"])

    # 设置回调
    st.session_state.workflow_coordinator.set_progress_callback(progress_callback)

    # 执行分析
    try:
        with st.spinner("正在进行智能分析..." if lang == "zh" else "Performing intelligent analysis..."):
            result = await st.session_state.workflow_coordinator.execute_analysis(user_input, lang)

            # 清理进度显示
            progress_container.empty()
            progress_bar.empty()
            status_container.empty()

            # 保存结果
            st.session_state.analysis_result = result

            if result.success:
                st.success("✅ " + ("分析完成！" if lang == "zh" else "Analysis completed!"))
                st.rerun()
            else:
                # 检查是否需要澄清
                if result.warnings and any("澄清" in w or "clarification" in w.lower() for w in result.warnings):
                    st.warning("需要补充信息，请提供更详细的描述" if lang == "zh" else "Need more information, please provide a more detailed description")
                else:
                    st.error(f"❌ " + ("分析失败：" if lang == "zh" else "Analysis failed: ") + (result.error_message or "未知错误"))

    except Exception as e:
        progress_container.empty()
        progress_bar.empty()
        status_container.empty()
        st.error(f"❌ " + ("分析过程中发生错误：" if lang == "zh" else "Error during analysis: ") + str(e))

if __name__ == "__main__":
    main()
