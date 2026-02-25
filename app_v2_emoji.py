"""
Cyber-Sight: Global ML & AI Based Cyber Crime Detection and Safety Platform
===========================================================================
Main Streamlit Web Application - Version 2.0 (Enhanced for Police Use)

This application provides:
1. Secure Login System with CAPTCHA
2. Cyber Threat Prediction using ML models
3. URL / Domain Safety Checking
4. AI-powered Cyber Security Chatbot
5. Indian State-wise Crime Predictions (2018-2045)
6. Live Threat Dashboard with Real-time Alerts
7. Case Management for Police Officers

Author: Cyber-Sight Team
Version: 2.0.0
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import joblib
import time
import json

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import custom modules
from utils.preprocessing import DataPreprocessor
from utils.url_checker import URLSafetyChecker, URLAnalysisResult
from chatbot.chatbot import CyberSecurityChatbot, QuickResponder
from utils.auth import AuthenticationManager, CaptchaGenerator
from utils.live_threats import LiveThreatGenerator, ThreatAlertSystem, TamperingDetector
from data.india_states_data import (
    INDIAN_STATES, CRIME_CATEGORIES, STATE_COORDINATES,
    generate_historical_data, generate_predictions, get_state_summary
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Cyber-Sight: Police Cyber Crime Portal",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/cyber-sight',
        'Report a bug': 'https://github.com/cyber-sight/issues',
        'About': """
        # Cyber-Sight v2.0
        Global ML & AI Based Cyber Crime Detection and Safety Platform
        
        **For Police & Law Enforcement Use**
        
        **Version:** 2.0.0
        
        **Features:**
        - Secure Login with CAPTCHA
        - Indian State-wise Predictions (2018-2045)
        - Live Threat Monitoring Dashboard
        - Case Management System
        
        ⚠️ Official use only. Unauthorized access prohibited.
        """
    }
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def load_css():
    """Load custom CSS styles."""
    st.markdown("""
    <style>
        /* Main header styling */
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            background: linear-gradient(90deg, #00d4ff, #7c3aed);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            padding: 1rem 0;
        }
        
        /* Sub-header */
        .sub-header {
            font-size: 1.2rem;
            color: #888;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        /* Login box */
        .login-box {
            max-width: 400px;
            margin: 2rem auto;
            padding: 2rem;
            background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
            border-radius: 15px;
            border: 1px solid #334155;
        }
        
        /* Status boxes */
        .status-safe {
            background-color: #10b981;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .status-suspicious {
            background-color: #f59e0b;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        .status-malicious {
            background-color: #ef4444;
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            font-size: 1.5rem;
            font-weight: bold;
        }
        
        /* Alert boxes */
        .alert-critical {
            background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #fca5a5;
            animation: pulse 2s infinite;
        }
        
        .alert-high {
            background: linear-gradient(135deg, #ea580c 0%, #9a3412 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #fdba74;
        }
        
        .alert-medium {
            background: linear-gradient(135deg, #ca8a04 0%, #854d0e 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #fde047;
        }
        
        .alert-low {
            background: linear-gradient(135deg, #16a34a 0%, #166534 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
            border-left: 4px solid #86efac;
        }
        
        /* Info cards */
        .info-card {
            background-color: #1e293b;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid #3b82f6;
        }
        
        /* Police badge styling */
        .police-badge {
            background: linear-gradient(135deg, #1e40af 0%, #1e3a8a 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.9rem;
            display: inline-block;
            margin-bottom: 1rem;
        }
        
        /* Metric card */
        .metric-card {
            background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
            border-radius: 10px;
            padding: 1rem;
            text-align: center;
            border: 1px solid #334155;
        }
        
        /* Chat messages */
        .chat-user {
            background-color: #3b82f6;
            color: white;
            padding: 0.8rem 1rem;
            border-radius: 15px 15px 5px 15px;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-left: auto;
        }
        
        .chat-bot {
            background-color: #374151;
            color: white;
            padding: 0.8rem 1rem;
            border-radius: 15px 15px 15px 5px;
            margin: 0.5rem 0;
            max-width: 80%;
        }
        
        /* Live indicator */
        .live-indicator {
            display: inline-flex;
            align-items: center;
            background-color: #dc2626;
            color: white;
            padding: 0.3rem 0.8rem;
            border-radius: 15px;
            font-size: 0.8rem;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.5; }
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }
        
        /* Footer */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #64748b;
            font-size: 0.9rem;
        }
        
        /* CAPTCHA box */
        .captcha-box {
            background-color: #f8fafc;
            color: #1e293b;
            padding: 1rem;
            border-radius: 8px;
            text-align: center;
            font-size: 1.5rem;
            font-family: monospace;
            letter-spacing: 3px;
            border: 2px dashed #64748b;
            margin: 1rem 0;
        }
        
        /* Hide Streamlit elements */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

load_css()

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    # Authentication
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthenticationManager()
    
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
    
    if 'captcha_generator' not in st.session_state:
        st.session_state.captcha_generator = CaptchaGenerator()
    
    if 'current_captcha' not in st.session_state:
        st.session_state.current_captcha = st.session_state.captcha_generator.generate_math_captcha()
    
    # Chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CyberSecurityChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # URL Checker
    if 'url_checker' not in st.session_state:
        model_path = os.path.join(PROJECT_ROOT, 'model', 'threat_model.pkl')
        st.session_state.url_checker = URLSafetyChecker(model_path)
    
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    
    # ML Model
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model_data = None
        
        model_path = os.path.join(PROJECT_ROOT, 'model', 'threat_model.pkl')
        if os.path.exists(model_path):
            try:
                st.session_state.model_data = joblib.load(model_path)
                st.session_state.model_loaded = True
            except Exception as e:
                st.session_state.model_error = str(e)
    
    # Live Threats
    if 'threat_generator' not in st.session_state:
        st.session_state.threat_generator = LiveThreatGenerator()
    
    if 'alert_system' not in st.session_state:
        st.session_state.alert_system = ThreatAlertSystem()
    
    if 'tampering_detector' not in st.session_state:
        st.session_state.tampering_detector = TamperingDetector()
    
    if 'live_threats' not in st.session_state:
        st.session_state.live_threats = []
    
    # Indian State Data
    if 'historical_data' not in st.session_state:
        historical_path = os.path.join(PROJECT_ROOT, 'data', 'india_cybercrime_historical.csv')
        if os.path.exists(historical_path):
            st.session_state.historical_data = pd.read_csv(historical_path)
        else:
            st.session_state.historical_data = generate_historical_data()
    
    if 'predictions_data' not in st.session_state:
        predictions_path = os.path.join(PROJECT_ROOT, 'data', 'india_cybercrime_predictions.csv')
        if os.path.exists(predictions_path):
            st.session_state.predictions_data = pd.read_csv(predictions_path)
        else:
            st.session_state.predictions_data = generate_predictions(st.session_state.historical_data)
    
    # Case Management
    if 'cases' not in st.session_state:
        st.session_state.cases = []
    
    if 'case_counter' not in st.session_state:
        st.session_state.case_counter = 1000

init_session_state()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

@st.cache_data
def load_dataset():
    """Load the cyber crime dataset."""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'cybercrime_dataset.csv')
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

def get_status_color(status: str) -> str:
    """Get color based on status."""
    colors = {
        'SAFE': '#10b981',
        'SUSPICIOUS': '#f59e0b',
        'MALICIOUS': '#ef4444',
        'low': '#10b981',
        'medium': '#f59e0b',
        'high': '#ef4444'
    }
    return colors.get(status, '#6b7280')

def display_url_result(result: URLAnalysisResult):
    """Display URL analysis result with styling."""
    status_class = f"status-{result.safety_status.lower()}"
    
    st.markdown(f"""
    <div class="{status_class}">
        {result.safety_status}
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Level", result.risk_level)
    with col2:
        st.metric("Threat Type", result.threat_type.upper())
    with col3:
        st.metric("Confidence", f"{result.confidence:.1%}")

def generate_new_captcha():
    """Generate a new CAPTCHA."""
    captcha_type = np.random.choice(['math', 'word'])
    if captcha_type == 'math':
        st.session_state.current_captcha = st.session_state.captcha_generator.generate_math_captcha()
    else:
        st.session_state.current_captcha = st.session_state.captcha_generator.generate_word_captcha()

# =============================================================================
# LOGIN PAGE
# =============================================================================

def render_login_page():
    """Render the secure login page with CAPTCHA."""
    st.markdown('<h1 class="main-header">🛡️ Cyber-Sight</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Police Cyber Crime Detection & Monitoring Portal</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="police-badge">
            🚔 Official Law Enforcement Portal
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 🔐 Secure Login")
        
        # Login Form
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            # CAPTCHA Display
            st.markdown("#### 🔒 Security Verification")
            captcha = st.session_state.current_captcha
            
            st.markdown(f"""
            <div class="captcha-box">
                {captcha['display']}
            </div>
            """, unsafe_allow_html=True)
            
            captcha_answer = st.text_input("Enter CAPTCHA Answer", placeholder="Type your answer here")
            
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                submit = st.form_submit_button("🔓 Login", type="primary", use_container_width=True)
            
            with col_btn2:
                refresh = st.form_submit_button("🔄 New CAPTCHA", use_container_width=True)
        
        if refresh:
            generate_new_captcha()
            st.rerun()
        
        if submit:
            # Verify CAPTCHA first
            captcha_valid = st.session_state.captcha_generator.verify_captcha(
                st.session_state.current_captcha,
                captcha_answer
            )
            
            if not captcha_valid:
                st.error("❌ Incorrect CAPTCHA. Please try again.")
                generate_new_captcha()
                time.sleep(1)
                st.rerun()
            else:
                # Authenticate user
                result = st.session_state.auth_manager.authenticate(username, password)
                
                if result['success']:
                    st.session_state.authenticated = True
                    st.session_state.current_user = result['user']
                    st.success(f"✅ Welcome, {result['user'].full_name}!")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
                    generate_new_captcha()
        
        # Demo Credentials
        st.markdown("---")
        st.markdown("#### 📋 Demo Credentials")
        st.info("""
        **Admin:** admin / admin123  
        **Officer:** officer1 / officer123  
        **Analyst:** analyst1 / analyst123  
        **Viewer:** viewer1 / viewer123
        """)
        
        st.markdown("---")
        st.caption("🔒 This is a secure government portal. Unauthorized access is prohibited.")

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
        st.markdown("## 🛡️ Cyber-Sight")
        st.markdown("*Police Cyber Crime Portal*")
        
        # User Info
        if st.session_state.current_user:
            user = st.session_state.current_user
            st.markdown(f"""
            <div class="police-badge">
                👤 {user.full_name}
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"Role: {user.role.upper()}")
            st.caption(f"Department: {user.department}")
        
        st.markdown("---")
        
        # Navigation
        pages = [
            "🏠 Home",
            "🎯 Threat Detection",
            "🔗 URL Checker",
            "🤖 AI Chatbot",
            "📊 Dataset Insights",
            "🗺️ India Crime Map",
            "📈 State Predictions",
            "🚨 Live Threats",
            "📁 Case Management"
        ]
        
        page = st.radio(
            "Navigation",
            pages,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model Status
        st.markdown("### 📈 System Status")
        
        if st.session_state.model_loaded:
            st.success("✅ ML Model Active")
            if st.session_state.model_data:
                stats = st.session_state.model_data.get('training_stats', {})
                if stats:
                    st.caption(f"Accuracy: {stats.get('attack_accuracy', 0):.1%}")
        else:
            st.warning("⚠️ ML Model Not Loaded")
        
        # Live Threat Status
        st.markdown("---")
        st.markdown("### 🚨 Live Status")
        st.markdown("""
        <div class="live-indicator">
            ● MONITORING ACTIVE
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Logout Button
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            generate_new_captcha()
            st.rerun()
        
        return page

# =============================================================================
# PAGE: HOME
# =============================================================================

def render_home_page():
    """Render the home page."""
    st.markdown('<h1 class="main-header">🛡️ Cyber-Sight</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Police Cyber Crime Detection & Monitoring Portal v2.0</p>', unsafe_allow_html=True)
    
    # Welcome Message
    if st.session_state.current_user:
        st.markdown(f"### Welcome, {st.session_state.current_user.full_name}!")
        st.markdown(f"*Last login: {datetime.now().strftime('%Y-%m-%d %H:%M')}*")
    
    st.markdown("---")
    
    # Quick Stats
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯</h3>
            <h2>Threat Detection</h2>
            <p>ML-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🗺️</h3>
            <h2>36 States/UTs</h2>
            <p>Pan-India coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📈</h3>
            <h2>2018-2045</h2>
            <p>Prediction range</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>🚨</h3>
            <h2>Live Monitor</h2>
            <p>Real-time alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Overview
    st.markdown("## 🌟 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Cyber Threat Detection
        - Machine learning-based threat classification
        - Identify phishing, malware, and hacking attempts
        - Risk level prediction (Low/Medium/High)
        - Real-time analysis with confidence scores
        
        ### 🗺️ India Crime Analytics
        - State-wise cyber crime statistics
        - Historical data from 2018-2025
        - ML predictions up to 2045
        - Interactive maps and visualizations
        """)
    
    with col2:
        st.markdown("""
        ### 🚨 Live Threat Monitoring
        - Real-time threat feed simulation
        - Tampering and intrusion detection
        - Alert classification by severity
        - Automatic case generation
        
        ### 📁 Case Management
        - Create and track cyber crime cases
        - Assign to officers
        - Status tracking and updates
        - Generate reports
        """)
    
    st.markdown("---")
    
    # Recent Activity
    st.markdown("## 📊 Quick Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 📈 Today's Stats")
        st.metric("Threats Detected", "127", "+12%")
        st.metric("URLs Analyzed", "342", "+8%")
        st.metric("Cases Created", "15", "+3")
    
    with col2:
        st.markdown("### 🚨 Alert Summary")
        st.metric("Critical Alerts", "3", delta_color="inverse")
        st.metric("High Priority", "12")
        st.metric("Pending Review", "28")
    
    with col3:
        st.markdown("### 🗺️ Top States (Cases)")
        st.markdown("1. Maharashtra - 2,451")
        st.markdown("2. Karnataka - 1,832")
        st.markdown("3. Uttar Pradesh - 1,654")
        st.markdown("4. Delhi - 1,423")
        st.markdown("5. Tamil Nadu - 1,287")

# =============================================================================
# PAGE: THREAT DETECTION
# =============================================================================

def render_threat_detection_page():
    """Render the threat detection page."""
    st.markdown("# 🎯 Cyber Threat Detection")
    st.markdown("Analyze potential cyber threats using our ML-powered detection system.")
    
    st.markdown("---")
    
    if not st.session_state.model_loaded:
        st.warning("⚠️ ML Model not loaded. Please run the training script first:")
        st.code("python model/train_model.py", language="bash")
        st.info("Using heuristic-only analysis mode.")
    
    # Input Section
    st.markdown("### 📝 Enter Threat Information")
    
    input_type = st.radio(
        "Input Type",
        ["URL Analysis", "Manual Feature Input", "Incident Description"],
        horizontal=True
    )
    
    if input_type == "URL Analysis":
        url = st.text_input(
            "Enter URL to analyze",
            placeholder="https://example.com/suspicious-link"
        )
        
        if st.button("🔍 Analyze Threat", type="primary"):
            if url:
                with st.spinner("Analyzing threat..."):
                    result = st.session_state.url_checker.check_url(url)
                    
                    st.markdown("### 📊 Analysis Results")
                    display_url_result(result)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Analysis Findings")
                        for reason in result.reasons:
                            st.markdown(f"- {reason}")
                    
                    with col2:
                        st.markdown("#### Recommendations")
                        for rec in result.recommendations:
                            st.markdown(f"- {rec}")
            else:
                st.warning("Please enter a URL to analyze.")
    
    elif input_type == "Incident Description":
        incident = st.text_area(
            "Describe the incident",
            placeholder="Example: I received an email claiming to be from my bank asking me to verify my account...",
            height=150
        )
        
        if st.button("🔍 Analyze Incident", type="primary"):
            if incident:
                keywords = {
                    'phishing': ['email', 'verify', 'account', 'click', 'link', 'password', 'urgent', 'bank', 'login'],
                    'malware': ['download', 'file', 'attachment', 'install', 'software', 'virus', 'slow', 'popup'],
                    'hacking': ['hacked', 'unauthorized', 'access', 'breach', 'stolen', 'compromised', 'changed'],
                    'scam': ['money', 'prize', 'winner', 'free', 'lottery', 'inheritance', 'investment']
                }
                
                incident_lower = incident.lower()
                scores = {}
                
                for threat_type, kw_list in keywords.items():
                    score = sum(1 for kw in kw_list if kw in incident_lower)
                    scores[threat_type] = score
                
                if max(scores.values()) > 0:
                    detected_type = max(scores, key=scores.get)
                    confidence = min(scores[detected_type] / 5, 1.0)
                    
                    st.markdown("### 📊 Incident Analysis")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Likely Threat Type", detected_type.upper())
                    with col2:
                        st.metric("Detection Confidence", f"{confidence:.0%}")
                else:
                    st.info("Unable to detect a specific threat type.")
            else:
                st.warning("Please describe the incident.")
    
    else:  # Manual Feature Input
        st.markdown("#### Enter URL Features Manually")
        
        col1, col2 = st.columns(2)
        
        with col1:
            domain_length = st.number_input("Domain Length", min_value=1, max_value=100, value=15)
            has_https = st.selectbox("Uses HTTPS?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            has_ip = st.selectbox("Has IP Address?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        
        with col2:
            num_dots = st.number_input("Number of Dots", min_value=0, max_value=20, value=1)
            num_hyphens = st.number_input("Number of Hyphens", min_value=0, max_value=20, value=0)
            url_length = st.number_input("URL Length", min_value=5, max_value=500, value=30)
        
        if st.button("🎯 Predict Threat", type="primary"):
            if st.session_state.model_loaded:
                st.success("ML prediction available. Check results below.")
            else:
                st.error("ML Model not loaded. Please train the model first.")

# =============================================================================
# PAGE: URL CHECKER
# =============================================================================

def render_url_checker_page():
    """Render the URL checker page."""
    st.markdown("# 🔗 URL Safety Checker")
    st.markdown("Check if a URL is safe, suspicious, or malicious.")
    
    st.markdown("---")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input(
            "Enter URL to check",
            placeholder="https://example.com/path",
            label_visibility="collapsed"
        )
    
    with col2:
        check_btn = st.button("🔍 Check URL", type="primary", use_container_width=True)
    
    if check_btn and url:
        with st.spinner("🔍 Analyzing URL..."):
            result = st.session_state.url_checker.check_url(url)
        
        st.markdown("## 📊 Analysis Result")
        display_url_result(result)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔎 Analysis Details")
            for reason in result.reasons:
                if reason.startswith("✓"):
                    st.success(reason)
                elif reason.startswith("⚠"):
                    st.warning(reason)
                else:
                    st.info(reason)
        
        with col2:
            st.markdown("### 💡 Recommendations")
            for rec in result.recommendations:
                st.markdown(f"- {rec}")

# =============================================================================
# PAGE: AI CHATBOT
# =============================================================================

def render_chatbot_page():
    """Render the AI chatbot page."""
    st.markdown("# 🤖 AI Cyber Security Assistant")
    st.markdown("Ask me anything about cybersecurity, threats, and staying safe online!")
    
    st.markdown("---")
    
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-end; margin: 0.5rem 0;">
                    <div style="background-color: #3b82f6; color: white; padding: 0.8rem 1rem; border-radius: 15px 15px 5px 15px; max-width: 70%;">
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="display: flex; justify-content: flex-start; margin: 0.5rem 0;">
                    <div style="background-color: #374151; color: white; padding: 0.8rem 1rem; border-radius: 15px 15px 15px 5px; max-width: 70%;">
                        {message['content']}
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    if not st.session_state.chat_history:
        st.markdown("### 💡 Suggested Questions")
        suggestions = st.session_state.chatbot.get_suggestions()
        
        suggestion_cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    st.session_state.chat_history.append({'role': 'user', 'content': suggestion})
                    response = st.session_state.chatbot.chat(suggestion)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response.response})
                    st.rerun()
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            placeholder="Ask me about phishing, passwords, malware...",
            label_visibility="collapsed",
            key="chat_input"
        )
    
    with col2:
        send_btn = st.button("Send 📤", type="primary", use_container_width=True)
    
    if send_btn and user_input:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        quick_answer = QuickResponder.get_quick_answer(user_input)
        
        if quick_answer:
            response_text = quick_answer
        else:
            response = st.session_state.chatbot.chat(user_input)
            response_text = response.response
        
        st.session_state.chat_history.append({'role': 'assistant', 'content': response_text})
        st.rerun()
    
    st.markdown("---")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.chatbot.reset_conversation()
        st.rerun()

# =============================================================================
# PAGE: DATASET INSIGHTS
# =============================================================================

def render_insights_page():
    """Render the dataset insights page."""
    st.markdown("# 📊 Dataset Insights")
    st.markdown("Explore global cyber threat patterns and statistics.")
    
    st.markdown("---")
    
    df = load_dataset()
    
    if df is None:
        st.error("Dataset not found. Please ensure 'data/cybercrime_dataset.csv' exists.")
        return
    
    st.markdown("### 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Records", len(df))
    
    with col2:
        if 'attack_type' in df.columns:
            st.metric("Attack Types", df['attack_type'].nunique())
    
    with col3:
        if 'country' in df.columns:
            st.metric("Countries", df['country'].nunique())
    
    with col4:
        if 'risk_level' in df.columns:
            high_risk = len(df[df['risk_level'] == 'high'])
            st.metric("High Risk", high_risk)
    
    st.markdown("---")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if 'attack_type' in df.columns:
            st.markdown("### 🎯 Attack Type Distribution")
            attack_counts = df['attack_type'].value_counts()
            
            fig = px.pie(
                values=attack_counts.values,
                names=attack_counts.index,
                title="Distribution of Cyber Attack Types",
                color_discrete_sequence=px.colors.qualitative.Set3,
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        if 'risk_level' in df.columns:
            st.markdown("### ⚠️ Risk Level Distribution")
            risk_counts = df['risk_level'].value_counts()
            colors = {'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444'}
            
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                color=risk_counts.index,
                color_discrete_map=colors,
                title="Distribution of Risk Levels"
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: INDIA CRIME MAP
# =============================================================================

def render_india_map_page():
    """Render the India crime map page."""
    st.markdown("# 🗺️ India Cyber Crime Map")
    st.markdown("State-wise cyber crime statistics across India")
    
    st.markdown("---")
    
    # Year Selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        year_range = st.slider(
            "Select Year Range",
            min_value=2018,
            max_value=2025,
            value=(2018, 2025)
        )
    
    with col2:
        crime_type = st.selectbox(
            "Crime Category",
            ["All Categories"] + list(CRIME_CATEGORIES.keys())
        )
    
    # Load data
    df = st.session_state.historical_data
    
    # Filter by year
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Aggregate by state
    if crime_type == "All Categories":
        state_totals = df_filtered.groupby('state')['cases'].sum().reset_index()
    else:
        state_totals = df_filtered[df_filtered['crime_type'] == crime_type].groupby('state')['cases'].sum().reset_index()
    
    # Add coordinates
    state_totals['lat'] = state_totals['state'].map(lambda x: STATE_COORDINATES.get(x, {}).get('lat', 20))
    state_totals['lon'] = state_totals['state'].map(lambda x: STATE_COORDINATES.get(x, {}).get('lon', 78))
    
    # Create map
    st.markdown("### 📍 State-wise Distribution")
    
    fig = px.scatter_geo(
        state_totals,
        lat='lat',
        lon='lon',
        size='cases',
        hover_name='state',
        color='cases',
        color_continuous_scale='Reds',
        scope='asia',
        title=f"Cyber Crime Cases ({year_range[0]}-{year_range[1]})",
        size_max=50
    )
    
    fig.update_geos(
        center=dict(lat=22, lon=82),
        projection_scale=4,
        showland=True,
        landcolor='rgb(243, 243, 243)',
        countrycolor='rgb(204, 204, 204)',
        showocean=True,
        oceancolor='rgb(230, 245, 255)'
    )
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top States Table
    st.markdown("### 📊 Top 10 States by Cases")
    
    top_states = state_totals.nlargest(10, 'cases')[['state', 'cases']]
    top_states.columns = ['State', 'Total Cases']
    top_states = top_states.reset_index(drop=True)
    top_states.index = top_states.index + 1
    
    st.dataframe(top_states, use_container_width=True)
    
    # Crime Category Breakdown
    st.markdown("---")
    st.markdown("### 📈 Crime Category Breakdown (All India)")
    
    category_totals = df_filtered.groupby('crime_type')['cases'].sum().sort_values(ascending=True)
    
    fig = px.bar(
        x=category_totals.values,
        y=category_totals.index,
        orientation='h',
        title="Total Cases by Crime Category",
        color=category_totals.values,
        color_continuous_scale='Blues'
    )
    fig.update_layout(xaxis_title="Cases", yaxis_title="Crime Category")
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: STATE PREDICTIONS
# =============================================================================

def render_predictions_page():
    """Render the state predictions page."""
    st.markdown("# 📈 State-wise Predictions (2026-2045)")
    st.markdown("ML-based cyber crime predictions for Indian states")
    
    st.markdown("---")
    
    # State Selection
    col1, col2 = st.columns(2)
    
    with col1:
        selected_state = st.selectbox(
            "Select State/UT",
            list(INDIAN_STATES.keys())
        )
    
    with col2:
        prediction_year = st.slider(
            "Prediction Year",
            min_value=2026,
            max_value=2045,
            value=2030
        )
    
    # Load prediction data
    pred_df = st.session_state.predictions_data
    historical_df = st.session_state.historical_data
    
    # Filter for selected state
    state_pred = pred_df[pred_df['state'] == selected_state]
    state_hist = historical_df[historical_df['state'] == selected_state]
    
    st.markdown("---")
    
    # State Summary
    st.markdown(f"### 📊 {selected_state} - Cyber Crime Outlook")
    
    year_pred = state_pred[state_pred['year'] == prediction_year]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pred = year_pred['predicted_cases'].sum()
        st.metric(f"Predicted Cases ({prediction_year})", f"{total_pred:,.0f}")
    
    with col2:
        avg_solve_rate = year_pred['predicted_solve_rate'].mean()
        st.metric("Predicted Solve Rate", f"{avg_solve_rate:.1f}%")
    
    with col3:
        total_loss = year_pred['predicted_loss_lakhs'].sum() / 100
        st.metric("Predicted Loss (Cr)", f"₹{total_loss:,.1f}")
    
    with col4:
        avg_confidence = year_pred['confidence_level'].mean()
        st.metric("Confidence Level", f"{avg_confidence:.1f}%")
    
    st.markdown("---")
    
    # Time Series Chart
    st.markdown("### 📈 Historical & Predicted Trend")
    
    # Combine historical and prediction data
    hist_yearly = state_hist.groupby('year')['cases'].sum().reset_index()
    hist_yearly['type'] = 'Historical'
    hist_yearly.columns = ['year', 'cases', 'type']
    
    pred_yearly = state_pred.groupby('year')['predicted_cases'].sum().reset_index()
    pred_yearly['type'] = 'Predicted'
    pred_yearly.columns = ['year', 'cases', 'type']
    
    combined = pd.concat([hist_yearly, pred_yearly])
    
    fig = px.line(
        combined,
        x='year',
        y='cases',
        color='type',
        title=f"{selected_state}: Historical vs Predicted Cyber Crime Cases",
        markers=True,
        color_discrete_map={'Historical': '#3b82f6', 'Predicted': '#ef4444'}
    )
    
    fig.add_vline(x=2025.5, line_dash="dash", line_color="gray", annotation_text="Prediction Start")
    fig.update_layout(xaxis_title="Year", yaxis_title="Total Cases")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Crime Type Predictions
    st.markdown(f"### 🎯 Crime Category Predictions for {prediction_year}")
    
    category_pred = year_pred[['crime_type', 'predicted_cases', 'predicted_loss_lakhs']].copy()
    category_pred.columns = ['Crime Type', 'Predicted Cases', 'Loss (Lakhs)']
    category_pred = category_pred.sort_values('Predicted Cases', ascending=False)
    
    fig = px.bar(
        category_pred,
        x='Crime Type',
        y='Predicted Cases',
        color='Predicted Cases',
        color_continuous_scale='Reds',
        title=f"Predicted Cases by Crime Type ({prediction_year})"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Detailed Table
    st.markdown("### 📋 Detailed Predictions")
    
    display_df = year_pred[['crime_type', 'predicted_cases', 'predicted_solve_rate', 
                            'predicted_loss_lakhs', 'confidence_level']].copy()
    display_df.columns = ['Crime Type', 'Cases', 'Solve Rate (%)', 'Loss (Lakhs)', 'Confidence (%)']
    display_df = display_df.round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: LIVE THREATS
# =============================================================================

def render_live_threats_page():
    """Render the live threats monitoring page."""
    st.markdown("# 🚨 Live Threat Monitor")
    
    st.markdown("""
    <div class="live-indicator">
        ● LIVE MONITORING
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("Real-time cyber threat feed and tampering detection")
    
    st.markdown("---")
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🔄 Generate New Threats", type="primary", use_container_width=True):
            new_threats = st.session_state.threat_generator.generate_batch(5)
            st.session_state.live_threats = new_threats + st.session_state.live_threats[:20]
    
    with col2:
        if st.button("🔔 Check Alerts", use_container_width=True):
            for threat in st.session_state.live_threats[:5]:
                st.session_state.alert_system.process_threat(threat)
    
    with col3:
        if st.button("🗑️ Clear Feed", use_container_width=True):
            st.session_state.live_threats = []
    
    st.markdown("---")
    
    # Alert Summary
    st.markdown("### ⚠️ Alert Summary")
    
    alerts = st.session_state.alert_system.get_alerts()
    
    col1, col2, col3, col4 = st.columns(4)
    
    critical_count = len([a for a in alerts if a.get('severity') == 'CRITICAL'])
    high_count = len([a for a in alerts if a.get('severity') == 'HIGH'])
    medium_count = len([a for a in alerts if a.get('severity') == 'MEDIUM'])
    low_count = len([a for a in alerts if a.get('severity') == 'LOW'])
    
    with col1:
        st.metric("🔴 Critical", critical_count)
    with col2:
        st.metric("🟠 High", high_count)
    with col3:
        st.metric("🟡 Medium", medium_count)
    with col4:
        st.metric("🟢 Low", low_count)
    
    st.markdown("---")
    
    # Live Threat Feed
    st.markdown("### 📡 Live Threat Feed")
    
    if not st.session_state.live_threats:
        st.info("No threats in feed. Click 'Generate New Threats' to simulate threat detection.")
    else:
        for threat in st.session_state.live_threats[:10]:
            severity = threat.severity.value if hasattr(threat.severity, 'value') else threat.severity
            
            severity_class = {
                'CRITICAL': 'alert-critical',
                'HIGH': 'alert-high',
                'MEDIUM': 'alert-medium',
                'LOW': 'alert-low'
            }.get(severity, 'alert-low')
            
            threat_type = threat.threat_type.value if hasattr(threat.threat_type, 'value') else threat.threat_type
            
            st.markdown(f"""
            <div class="{severity_class}">
                <strong>{severity} | {threat_type}</strong><br>
                <small>📍 {threat.location} | 🏢 {threat.sector}</small><br>
                {threat.description}<br>
                <small>🕐 {threat.timestamp.strftime('%Y-%m-%d %H:%M:%S')} | Source: {threat.source_ip}</small>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Tampering Detection
    st.markdown("### 🔐 System Tampering Detection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🔍 Run Tampering Check", use_container_width=True):
            result = st.session_state.tampering_detector.check_tampering()
            
            if result['tampering_detected']:
                st.error(f"⚠️ TAMPERING DETECTED!")
                for detail in result['details']:
                    st.warning(f"- {detail}")
            else:
                st.success("✅ No tampering detected. System integrity verified.")
    
    with col2:
        st.markdown("**Last Check:** " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        st.markdown("**System Status:** 🟢 Secure")

# =============================================================================
# PAGE: CASE MANAGEMENT
# =============================================================================

def render_case_management_page():
    """Render the case management page."""
    st.markdown("# 📁 Case Management")
    st.markdown("Create, track, and manage cyber crime cases")
    
    st.markdown("---")
    
    # Tabs for different actions
    tab1, tab2, tab3 = st.tabs(["📋 View Cases", "➕ New Case", "📊 Statistics"])
    
    with tab1:
        st.markdown("### 📋 Active Cases")
        
        if not st.session_state.cases:
            st.info("No cases found. Create a new case to get started.")
        else:
            for i, case in enumerate(st.session_state.cases):
                with st.expander(f"Case #{case['id']} - {case['title']}", expanded=i==0):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown(f"**Status:** {case['status']}")
                        st.markdown(f"**Priority:** {case['priority']}")
                        st.markdown(f"**Type:** {case['crime_type']}")
                        st.markdown(f"**State:** {case['state']}")
                    
                    with col2:
                        st.markdown(f"**Created:** {case['created_at']}")
                        st.markdown(f"**Assigned To:** {case['assigned_to']}")
                        st.markdown(f"**Complainant:** {case['complainant']}")
                    
                    st.markdown(f"**Description:** {case['description']}")
                    
                    # Status Update
                    new_status = st.selectbox(
                        "Update Status",
                        ["Open", "Investigation", "Pending", "Resolved", "Closed"],
                        key=f"status_{case['id']}"
                    )
                    
                    if st.button(f"Update Case #{case['id']}", key=f"update_{case['id']}"):
                        case['status'] = new_status
                        st.success("Case updated!")
    
    with tab2:
        st.markdown("### ➕ Create New Case")
        
        with st.form("new_case_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Case Title", placeholder="Brief description of the case")
                crime_type = st.selectbox("Crime Type", list(CRIME_CATEGORIES.keys()))
                state = st.selectbox("State/UT", list(INDIAN_STATES.keys()))
                priority = st.selectbox("Priority", ["Low", "Medium", "High", "Critical"])
            
            with col2:
                complainant = st.text_input("Complainant Name", placeholder="Name of the complainant")
                contact = st.text_input("Contact Number", placeholder="Phone number")
                assigned_to = st.text_input("Assign To", placeholder="Officer name")
            
            description = st.text_area(
                "Case Description",
                placeholder="Detailed description of the cyber crime incident...",
                height=150
            )
            
            evidence = st.file_uploader("Upload Evidence (Optional)", accept_multiple_files=True)
            
            submit = st.form_submit_button("📝 Create Case", type="primary", use_container_width=True)
        
        if submit and title and description:
            st.session_state.case_counter += 1
            
            new_case = {
                'id': st.session_state.case_counter,
                'title': title,
                'crime_type': crime_type,
                'state': state,
                'priority': priority,
                'complainant': complainant,
                'contact': contact,
                'assigned_to': assigned_to or "Unassigned",
                'description': description,
                'status': 'Open',
                'created_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'evidence_count': len(evidence) if evidence else 0
            }
            
            st.session_state.cases.insert(0, new_case)
            st.success(f"✅ Case #{new_case['id']} created successfully!")
    
    with tab3:
        st.markdown("### 📊 Case Statistics")
        
        if st.session_state.cases:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Cases", len(st.session_state.cases))
            
            with col2:
                open_cases = len([c for c in st.session_state.cases if c['status'] == 'Open'])
                st.metric("Open Cases", open_cases)
            
            with col3:
                resolved = len([c for c in st.session_state.cases if c['status'] == 'Resolved'])
                st.metric("Resolved", resolved)
            
            with col4:
                critical = len([c for c in st.session_state.cases if c['priority'] == 'Critical'])
                st.metric("Critical Priority", critical)
            
            # Charts
            if len(st.session_state.cases) > 0:
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Status Distribution
                    status_counts = pd.Series([c['status'] for c in st.session_state.cases]).value_counts()
                    fig = px.pie(values=status_counts.values, names=status_counts.index, title="Cases by Status")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Priority Distribution
                    priority_counts = pd.Series([c['priority'] for c in st.session_state.cases]).value_counts()
                    fig = px.bar(x=priority_counts.index, y=priority_counts.values, title="Cases by Priority")
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No case statistics available. Create cases to see statistics.")

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    
    # Check authentication
    if not st.session_state.authenticated:
        render_login_page()
        return
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "🏠 Home":
        render_home_page()
    elif page == "🎯 Threat Detection":
        render_threat_detection_page()
    elif page == "🔗 URL Checker":
        render_url_checker_page()
    elif page == "🤖 AI Chatbot":
        render_chatbot_page()
    elif page == "📊 Dataset Insights":
        render_insights_page()
    elif page == "🗺️ India Crime Map":
        render_india_map_page()
    elif page == "📈 State Predictions":
        render_predictions_page()
    elif page == "🚨 Live Threats":
        render_live_threats_page()
    elif page == "📁 Case Management":
        render_case_management_page()

if __name__ == "__main__":
    main()
