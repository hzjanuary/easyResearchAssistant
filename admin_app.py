"""Admin Dashboard for easyResearchAssistant - System Health Monitoring."""
import os
import hashlib
import requests
import streamlit as st
import pandas as pd
from typing import Dict, Any, List
from datetime import datetime

try:
    from streamlit_autorefresh import st_autorefresh
    AUTOREFRESH_AVAILABLE = True
except ImportError:
    AUTOREFRESH_AVAILABLE = False

# API Configuration
API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
DEFAULT_ACCESS_TOKEN = os.getenv("ACCESS_TOKEN", "")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")
SYSTEM_LOG_FILE = os.getenv("LOG_FILE", "system.log")

st.set_page_config(
    page_title="Admin Dashboard - easyResearchAssistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .status-online { color: #10b981; font-weight: 600; }
    .status-offline { color: #ef4444; font-weight: 600; }
    .status-active { 
        background-color: #10b981; color: white; 
        padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
    }
    .status-cooldown { 
        background-color: #f59e0b; color: white; 
        padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
    }
    .status-offline-badge { 
        background-color: #ef4444; color: white; 
        padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
    }
    .status-degraded { 
        background-color: #6b7280; color: white; 
        padding: 2px 8px; border-radius: 4px; font-size: 0.75rem;
    }
    .node-card {
        border: 1px solid #e5e7eb; border-radius: 8px; padding: 12px;
        margin-bottom: 8px; background: #fafafa;
    }
    .log-entry { 
        font-family: monospace; font-size: 0.8rem; 
        padding: 4px 8px; border-radius: 4px; margin-bottom: 4px;
    }
    .log-info { background-color: #dbeafe; }
    .log-warning { background-color: #fef3c7; }
    .log-error { background-color: #fee2e2; }
    .admin-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d5a87 100%);
        color: white; padding: 12px 20px; border-radius: 8px;
        margin-bottom: 20px;
    }
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


def init_session_state():
    if "admin_authenticated" not in st.session_state:
        st.session_state.admin_authenticated = False
    
    if "access_token" not in st.session_state:
        st.session_state.access_token = DEFAULT_ACCESS_TOKEN
    
    if "auto_refresh" not in st.session_state:
        st.session_state.auto_refresh = True
    
    if "api_connected" not in st.session_state:
        st.session_state.api_connected = False


def verify_admin_password(password: str) -> bool:
    """Verify admin password"""
    if not ADMIN_PASSWORD:
        return True
    return password == ADMIN_PASSWORD


def check_api_health() -> dict:
    try:
        response = requests.get(f"{API_BASE_URL}/v1/monitoring/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"status": "error", "message": f"HTTP {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"status": "error", "message": "Connection refused"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


def get_monitoring_stats(token: str) -> Dict[str, Any]:
    """Fetch comprehensive monitoring statistics from the backend"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(f"{API_BASE_URL}/v1/monitoring/stats", headers=headers, timeout=5)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 401:
            return {"error": "Invalid access token"}
        elif response.status_code == 403:
            return {"error": "Access denied"}
        return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}


def reset_all_providers(token: str) -> bool:
    """Reset all providers to healthy status"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(f"{API_BASE_URL}/v1/providers/reset", headers=headers, timeout=5)
        return response.status_code == 200
    except Exception:
        return False


def change_strategy(token: str, strategy: str) -> bool:
    """Change the load balancing strategy"""
    try:
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.post(
            f"{API_BASE_URL}/v1/providers/strategy/{strategy}", 
            headers=headers, 
            timeout=5
        )
        return response.status_code == 200
    except Exception:
        return False


def get_status_badge(status: str) -> str:
    """Get HTML badge for node status"""
    status_lower = status.lower()
    if "active" in status_lower:
        return '<span class="status-active">Active</span>'
    elif "cooldown" in status_lower:
        return f'<span class="status-cooldown">{status}</span>'
    elif "offline" in status_lower:
        return '<span class="status-offline-badge">Offline</span>'
    else:
        return f'<span class="status-degraded">{status}</span>'


def read_system_logs(num_lines: int = 100) -> List[str]:
    """
    Read the last N lines from the system.log file.
    Returns a list of log lines, newest first.
    """
    try:
        if not os.path.exists(SYSTEM_LOG_FILE):
            return []
        
        with open(SYSTEM_LOG_FILE, 'r', encoding='utf-8') as f:
            # Read all lines and get the last num_lines
            lines = f.readlines()
            # Return the last num_lines, reversed to show newest first
            return [line.strip() for line in lines[-num_lines:] if line.strip()]
    except Exception as e:
        return [f"Error reading log file: {str(e)}"]


def render_login():
    """Render admin login form"""
    st.markdown("""
    <div class="admin-header">
        <h2 style="margin: 0;">Admin Dashboard</h2>
        <p style="margin: 0; opacity: 0.8;">easyResearchAssistant System Monitoring</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("Authentication Required")
    
    col1, col2 = st.columns(2)
    
    with col1:
        access_token = st.text_input(
            "Access Token",
            value=st.session_state.access_token,
            type="password",
            help="API Access Token (same as chat app)"
        )
    
    with col2:
        admin_password = st.text_input(
            "Admin Password",
            type="password",
            help="Additional admin password for dashboard access"
        )
    
    if st.button("Login", type="primary", use_container_width=True):
        if not access_token:
            st.error("Access token is required")
            return
        
        if not verify_admin_password(admin_password):
            st.error("Invalid admin password")
            return
        
        # Verify access token with API
        test_stats = get_monitoring_stats(access_token)
        if "error" in test_stats:
            st.error(f"Authentication failed: {test_stats['error']}")
            return
        
        st.session_state.access_token = access_token
        st.session_state.admin_authenticated = True
        st.rerun()
    
    if not ADMIN_PASSWORD:
        st.info("No ADMIN_PASSWORD configured. Set it in .env for additional security.")


def render_node_card(node: Dict[str, Any]):
    """Render a single node status card"""
    status_html = get_status_badge(node.get("display_status", "Unknown"))
    
    avg_time = node.get("average_response_time", 0)
    time_str = f"{avg_time:.2f}s" if avg_time > 0 else "N/A"
    
    st.markdown(f"""
    <div class="node-card">
        <div style="display: flex; justify-content: space-between; align-items: center;">
            <strong>{node.get('name', 'Unknown')}</strong>
            {status_html}
        </div>
        <div style="font-size: 0.85rem; color: #6b7280; margin-top: 4px;">
            Type: {node.get('type', 'unknown').upper()} | 
            Requests: {node.get('request_count', 0)} | 
            Success: {node.get('success_rate', 100)}% |
            Avg: {time_str}
        </div>
    </div>
    """, unsafe_allow_html=True)


def render_dashboard():
    """Render the main monitoring dashboard"""
    
    # Auto-refresh (only when viewing dashboard, doesn't affect chat)
    if AUTOREFRESH_AVAILABLE and st.session_state.auto_refresh:
        st_autorefresh(interval=10000, limit=None, key="admin_dashboard_refresh")
    
    # Header
    st.markdown("""
    <div class="admin-header">
        <h2 style="margin: 0;">System Health Dashboard</h2>
        <p style="margin: 0; opacity: 0.8;">Real-time monitoring of inference providers</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API connection
    health = check_api_health()
    if health.get("status") == "error":
        st.error(f"Cannot connect to API: {health.get('message')}")
        st.session_state.api_connected = False
        return
    
    st.session_state.api_connected = True
    
    # Fetch monitoring stats
    stats = get_monitoring_stats(st.session_state.access_token)
    
    if "error" in stats:
        st.error(f"Failed to fetch monitoring data: {stats['error']}")
        if stats["error"] in ["Invalid access token", "Access denied"]:
            st.session_state.admin_authenticated = False
            st.rerun()
        return
    
    # Dashboard controls
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        if AUTOREFRESH_AVAILABLE:
            st.session_state.auto_refresh = st.toggle(
                "Auto-refresh (10s)", 
                value=st.session_state.auto_refresh,
                help="Automatically refresh dashboard every 10 seconds"
            )
        else:
            st.caption("Install streamlit-autorefresh for auto-refresh")
    
    with col2:
        if st.button("Refresh Now", use_container_width=True):
            st.rerun()
    
    with col3:
        if st.button("Reset All Nodes", use_container_width=True, type="secondary"):
            if reset_all_providers(st.session_state.access_token):
                st.success("All nodes reset!")
                st.rerun()
            else:
                st.error("Failed to reset nodes")
    
    with col4:
        if st.button("Logout", use_container_width=True):
            st.session_state.admin_authenticated = False
            st.rerun()
    
    st.divider()
    
    # Summary metrics
    summary = stats.get("summary", {})
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Requests",
            summary.get("total_requests", 0),
            help="Total requests across all nodes"
        )
    
    with col2:
        available = summary.get("available_nodes", 0)
        total = summary.get("total_nodes", 0)
        delta_color = "normal"
        if total > 0 and available < total:
            delta_color = "inverse"
        st.metric(
            "Available Nodes",
            f"{available}/{total}",
            delta=f"{int((available/total)*100)}%" if total > 0 else None,
            delta_color=delta_color,
            help="Nodes currently accepting requests"
        )
    
    with col3:
        success_rate = summary.get('overall_success_rate', 100)
        st.metric(
            "Success Rate",
            f"{success_rate}%",
            delta="Good" if success_rate >= 95 else "Low",
            delta_color="normal" if success_rate >= 95 else "inverse",
            help="Overall success rate"
        )
    
    with col4:
        rate_limits = summary.get("total_rate_limits", 0)
        st.metric(
            "Rate Limits (429)",
            rate_limits,
            help="Total rate limit errors received"
        )
    
    with col5:
        fallback = "Ready" if summary.get("has_fallback") else "N/A"
        st.metric("Local Fallback", fallback)
    
    st.divider()
    
    # Node status and request distribution
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Node Status")
        
        nodes_by_status = stats.get("nodes_by_status", {})
        
        # Active nodes
        active_nodes = nodes_by_status.get("active", [])
        if active_nodes:
            st.markdown("**Active**")
            for node in active_nodes:
                render_node_card(node)
        
        # Cooldown nodes
        cooldown_nodes = nodes_by_status.get("cooldown", [])
        if cooldown_nodes:
            st.markdown("**In Cooldown**")
            for node in cooldown_nodes:
                render_node_card(node)
        
        # Degraded nodes
        degraded_nodes = nodes_by_status.get("degraded", [])
        if degraded_nodes:
            st.markdown("**Degraded**")
            for node in degraded_nodes:
                render_node_card(node)
        
        # Offline nodes
        offline_nodes = nodes_by_status.get("offline", [])
        if offline_nodes:
            st.markdown("**Offline**")
            for node in offline_nodes:
                render_node_card(node)
        
        # Local fallback
        fallback_node = stats.get("fallback")
        if fallback_node:
            st.markdown("**Local Fallback (Ollama)**")
            render_node_card(fallback_node)
    
    with col2:
        st.subheader("Request Distribution")
        
        # Bar chart data
        all_nodes = stats.get("all_nodes", [])
        if all_nodes:
            chart_data = []
            for node in all_nodes:
                chart_data.append({
                    "Provider": node.get("name", "Unknown"),
                    "Requests": node.get("request_count", 0),
                    "Errors": node.get("error_count", 0),
                    "Success Rate": node.get("success_rate", 100)
                })
            
            # Add fallback if it has requests
            if fallback_node and fallback_node.get("request_count", 0) > 0:
                chart_data.append({
                    "Provider": fallback_node.get("name", "Local"),
                    "Requests": fallback_node.get("request_count", 0),
                    "Errors": fallback_node.get("error_count", 0),
                    "Success Rate": fallback_node.get("success_rate", 100)
                })
            
            if chart_data:
                df = pd.DataFrame(chart_data)
                
                # Bar chart
                st.bar_chart(df.set_index("Provider")["Requests"])
                
                # Detailed table
                with st.expander("Detailed Statistics"):
                    st.dataframe(df, use_container_width=True, hide_index=True)
        else:
            st.info("No request data yet")
        
        # Strategy selector
        st.subheader("Load Balancing Strategy")
        current_strategy = stats.get("strategy", "round_robin")
        
        strategy_options = ["round_robin", "random", "least_used"]
        strategy_labels = {
            "round_robin": "Round Robin",
            "random": "Random",
            "least_used": "Least Used"
        }
        
        new_strategy = st.selectbox(
            "Strategy",
            options=strategy_options,
            index=strategy_options.index(current_strategy) if current_strategy in strategy_options else 0,
            format_func=lambda x: strategy_labels.get(x) or str(x)
        )
        
        if new_strategy != current_strategy:
            if change_strategy(st.session_state.access_token, str(new_strategy)):
                st.success(f"Strategy changed to {strategy_labels.get(str(new_strategy), new_strategy)}")
                st.rerun()
            else:
                st.error("Failed to change strategy")
    
    st.divider()
    
    # Live logs
    st.subheader("Live Logs")
    
    logs = stats.get("recent_logs", [])
    if logs:
        log_container = st.container()
        with log_container:
            for log in reversed(logs[-15:]):
                level = log.get("level", "INFO").upper()
                timestamp = log.get("timestamp", "")
                message = log.get("message", "")
                node = log.get("node", "")
                
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp)
                        time_str = dt.strftime("%H:%M:%S")
                    except:
                        time_str = timestamp[:8]
                else:
                    time_str = ""
                
                node_str = f"[{node}]" if node else ""
                
                css_class = f"log-{level.lower()}"
                st.markdown(
                    f'<div class="log-entry {css_class}">'
                    f'<strong>{time_str}</strong> {level} {node_str} {message}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    else:
        st.info("No logs yet. Logs will appear as requests are processed.")
    
    # Live System Logs from file (persistent logs)
    st.divider()
    with st.expander("📋 Live System Logs (File)", expanded=False):
        st.caption(f"Reading from: {SYSTEM_LOG_FILE}")
        
        col1, col2 = st.columns([1, 4])
        with col1:
            if st.button("🔄 Refresh Logs", key="refresh_file_logs"):
                st.rerun()
        with col2:
            num_lines = st.selectbox(
                "Lines to show",
                options=[50, 100, 200],
                index=1,
                key="log_lines_select"
            )
        
        file_logs = read_system_logs(num_lines)
        
        if file_logs:
            # Show logs in a code block with monospace font
            log_text = "\n".join(reversed(file_logs))  # Newest first
            st.code(log_text, language="log")
            
            st.caption(f"Showing last {len(file_logs)} lines")
        else:
            st.info(f"No logs found in {SYSTEM_LOG_FILE}. Logs will appear as the system processes requests.")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption(f"Strategy: {stats.get('strategy', 'round_robin').replace('_', ' ').title()}")
    with col2:
        st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')}")
    with col3:
        st.caption(f"API: {API_BASE_URL}")


def main():
    init_session_state()
    
    if not st.session_state.admin_authenticated:
        render_login()
    else:
        render_dashboard()


if __name__ == "__main__":
    main()
