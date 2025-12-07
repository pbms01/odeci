"""
Módulo web para interface Streamlit do ODECI Chat.

Fornece componentes e gerenciamento de estado para a interface web.
"""

from src.chat.web.components import (
    render_api_key_input,
    render_chat_input,
    render_error,
    render_follow_up,
    render_header,
    render_info,
    render_message,
    render_processing_indicator,
    render_sidebar_config,
    render_sources_expander,
    render_warning,
    render_welcome_message,
)
from src.chat.web.state import (
    WebState,
    add_assistant_message,
    add_user_message,
    clear_messages,
    get_state,
    get_style_options,
    init_state,
    style_from_string,
    update_last_assistant_message,
    update_state,
)

__all__ = [
    # State
    "WebState",
    "init_state",
    "get_state",
    "update_state",
    "add_user_message",
    "add_assistant_message",
    "update_last_assistant_message",
    "clear_messages",
    "get_style_options",
    "style_from_string",
    # Components
    "render_header",
    "render_sidebar_config",
    "render_message",
    "render_sources_expander",
    "render_follow_up",
    "render_chat_input",
    "render_error",
    "render_warning",
    "render_info",
    "render_processing_indicator",
    "render_welcome_message",
    "render_api_key_input",
]
