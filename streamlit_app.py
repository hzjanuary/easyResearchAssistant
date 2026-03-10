"""
easyResearchAssistant - Streamlit Chat Interface
=================================================
A modern chat UI with streaming responses and Research Mode support.

Features:
- Real-time streaming responses
- Research Mode toggle for academic/educational focus
- Chat history management
- Provider status monitoring
- Mobile-responsive design
"""
import os
import json
import requests
import streamlit as st
from typing import Generator

# =============================================================================
# Configuration
# =============================================================================

# API Configuration - defaults for local development
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DEFAULT_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")

# Page configuration
st.set_page_config(
    page_title="easyResearchAssistant",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UX
st.markdown("""
<style>
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Chat message styling */
    .stChatMessage {
        background-color: transparent;
    }
    
    /* Research mode indicator */
    .research-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    
    /* Status indicator */
    .status-online {
        color: #10b981;
        font-weight: 600;
    }
    .status-offline {
        color: #ef4444;
        font-weight: 600;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# Session State Initialization
# =============================================================================

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "access_token" not in st.session_state:
        st.session_state.access_token = DEFAULT_ACCESS_TOKEN
    
    if "research_mode" not in st.session_state:
        st.session_state.research_mode = False
    
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False
    
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    
    if "max_tokens" not in st.session_state:
        st.session_state.max_tokens = 2048


# =============================================================================
# API Functions
# =============================================================================

def check_api_health() -> dict:
    """Check API gateway health status"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Connection refused"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_gateway_status(token: str) -> dict:
    """Get detailed gateway status"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{API_BASE_URL}/v1/status", headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def stream_chat_response(
    prompt: str,
    token: str,
    research_mode: bool = False,
    temperature: float = 0.7,
    max_tokens: int = 2048
) -> Generator[str, None, None]:
    """
    Stream chat response from the API gateway.
    Yields text chunks as they arrive.
    """
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt,
        "stream": True,
        "research_mode": research_mode,
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    try:
        with requests.post(
            f"{API_BASE_URL}/v1/inference",
            headers=headers,
            json=payload,
            stream=True,
            timeout=120
        ) as response:
            if response.status_code != 200:
                yield f"Error: HTTP {response.status_code}"
                return
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8')
                    if line_str.startswith("data: "):
                        data = line_str[6:]
                        
                        if data == "[DONE]":
                            return
                        
                        try:
                            parsed = json.loads(data)
                            
                            # Handle info messages (provider switching)
                            if "info" in parsed:
                                yield f"\n*{parsed['info']}*\n"
                                continue
                            
                            # Handle errors
                            if "error" in parsed:
                                yield f"\n**Error:** {parsed['error']}"
                                return
                            
                            # Handle response content
                            if "response" in parsed:
                                yield parsed["response"]
                            
                        except json.JSONDecodeError:
                            # Raw text response
                            yield data
                            
    except requests.exceptions.Timeout:
        yield "Error: Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        yield "Error: Cannot connect to the API gateway. Is it running?"
    except Exception as e:
        yield f"Error: {str(e)}"


# =============================================================================
# UI Components
# =============================================================================

def render_sidebar():
    """Render the sidebar with settings and status"""
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Access Token
        st.subheader("🔐 Authentication")
        token = st.text_input(
            "Access Token",
            value=st.session_state.access_token,
            type="password",
            help="Your family access token for the API"
        )
        if token != st.session_state.access_token:
            st.session_state.access_token = token
        
        st.divider()
        
        # Research Mode Toggle
        st.subheader("📚 Research Mode")
        research_mode = st.toggle(
            "Enable Research Mode",
            value=st.session_state.research_mode,
            help="Optimized for academic summarization and educational explanations"
        )
        st.session_state.research_mode = research_mode
        
        if research_mode:
            st.info(
                "Research Mode active: Responses will focus on "
                "academic clarity, structured explanations, and educational value."
            )
        
        st.divider()
        
        # Generation Parameters
        st.subheader("🎛️ Parameters")
        
        st.session_state.temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=2.0,
            value=st.session_state.temperature,
            step=0.1,
            help="Higher = more creative, Lower = more focused"
        )
        
        st.session_state.max_tokens = st.slider(
            "Max Tokens",
            min_value=256,
            max_value=4096,
            value=st.session_state.max_tokens,
            step=256,
            help="Maximum response length"
        )
        
        st.divider()
        
        # Gateway Status
        st.subheader("📊 Gateway Status")
        
        health = check_api_health()
        if health.get("status") == "healthy":
            st.session_state.api_connected = True
            st.markdown('<span class="status-online">● Online</span>', unsafe_allow_html=True)
            
            providers = health.get("providers", {})
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Cloud Nodes", f"{providers.get('cloud_available', 0)}/{providers.get('cloud_total', 0)}")
            with col2:
                fallback = "✓" if providers.get("local_fallback") else "✗"
                st.metric("Local Fallback", fallback)
        else:
            st.session_state.api_connected = False
            st.markdown('<span class="status-offline">● Offline</span>', unsafe_allow_html=True)
            st.error(health.get("message", "Connection failed"))
        
        # Detailed status (if authenticated)
        if st.session_state.access_token and st.session_state.api_connected:
            with st.expander("Detailed Status"):
                status = get_gateway_status(st.session_state.access_token)
                if "error" not in status:
                    st.json(status)
                else:
                    st.warning(f"Could not fetch: {status['error']}")
        
        st.divider()
        
        # Actions
        st.subheader("🔧 Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.rerun()
        
        with col2:
            if st.button("🔄 Refresh", use_container_width=True):
                st.rerun()


def render_chat():
    """Render the main chat interface"""
    # Header
    col1, col2 = st.columns([3, 1])
    with col1:
        st.title("🔬 easyResearchAssistant")
        st.caption("High-Availability AI Inference Gateway")
    
    with col2:
        if st.session_state.research_mode:
            st.markdown(
                '<span class="research-badge">📚 Research Mode</span>',
                unsafe_allow_html=True
            )
    
    # Connection warning
    if not st.session_state.api_connected:
        st.warning(
            "⚠️ Not connected to the API gateway. "
            "Make sure the server is running: `python api_gateway.py`"
        )
    
    if not st.session_state.access_token:
        st.info(
            "🔐 Enter your access token in the sidebar to start chatting."
        )
    
    # Chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input(
        "Ask me anything...",
        disabled=not (st.session_state.api_connected and st.session_state.access_token)
    ):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            full_response = ""
            
            # Stream the response
            for chunk in stream_chat_response(
                prompt=prompt,
                token=st.session_state.access_token,
                research_mode=st.session_state.research_mode,
                temperature=st.session_state.temperature,
                max_tokens=st.session_state.max_tokens
            ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
        
        # Add assistant message to history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# =============================================================================
# Main Application
# =============================================================================

def main():
    """Main application entry point"""
    init_session_state()
    render_sidebar()
    render_chat()


if __name__ == "__main__":
    main()
