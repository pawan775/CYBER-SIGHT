"""
Cyber-Sight: Global ML & AI Based Cyber Crime Detection and Safety Platform
===========================================================================
Main Streamlit Web Application

This application provides:
1. Cyber Threat Prediction using ML models
2. URL / Domain Safety Checking
3. AI-powered Cyber Security Chatbot
4. Dataset Insights and Visualization

Author: Cyber-Sight Team
Version: 1.0.0
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import joblib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

# Import custom modules
from utils.preprocessing import DataPreprocessor
from utils.url_checker import URLSafetyChecker, URLAnalysisResult
from chatbot.chatbot import CyberSecurityChatbot, QuickResponder

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Cyber-Sight: Global Cyber Crime Detection",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/cyber-sight',
        'Report a bug': 'https://github.com/cyber-sight/issues',
        'About': """
        # Cyber-Sight
        Global ML & AI Based Cyber Crime Detection and Safety Platform
        
        **Version:** 1.0.0
        
        **Purpose:** Educational cyber security awareness and threat detection.
        
        ⚠️ This tool is for educational purposes only.
        """
    }
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

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
    
    /* Subheader */
    .sub-header {
        font-size: 1.2rem;
        color: #888;
        text-align: center;
        margin-bottom: 2rem;
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
    
    /* Info cards */
    .info-card {
        background-color: #1e293b;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #3b82f6;
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
    
    /* Metric styling */
    .metric-card {
        background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #334155;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #64748b;
        font-size: 0.9rem;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE SESSION STATE
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CyberSecurityChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'url_checker' not in st.session_state:
        model_path = os.path.join(PROJECT_ROOT, 'model', 'threat_model.pkl')
        st.session_state.url_checker = URLSafetyChecker(model_path)
    
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = DataPreprocessor()
    
    if 'model_loaded' not in st.session_state:
        st.session_state.model_loaded = False
        st.session_state.model_data = None
        
        # Try to load model
        model_path = os.path.join(PROJECT_ROOT, 'model', 'threat_model.pkl')
        if os.path.exists(model_path):
            try:
                st.session_state.model_data = joblib.load(model_path)
                st.session_state.model_loaded = True
            except Exception as e:
                st.session_state.model_error = str(e)

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

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation."""
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/security-checked.png", width=80)
        st.markdown("## 🔒 Cyber-Sight")
        st.markdown("*Global Cyber Crime Detection*")
        
        st.markdown("---")
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🏠 Home", "🎯 Threat Detection", "🔗 URL Checker", "🤖 AI Chatbot", "📊 Dataset Insights"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Model Status
        st.markdown("### 📈 System Status")
        
        if st.session_state.model_loaded:
            st.success("✅ ML Model Loaded")
            if st.session_state.model_data:
                stats = st.session_state.model_data.get('training_stats', {})
                if stats:
                    st.caption(f"Accuracy: {stats.get('attack_accuracy', 0):.1%}")
        else:
            st.warning("⚠️ ML Model Not Loaded")
            st.caption("Run train_model.py first")
        
        st.markdown("---")
        
        # Quick Stats
        st.markdown("### 🌍 Global Coverage")
        st.caption("This system is designed for worldwide cyber threat detection and awareness.")
        
        st.markdown("---")
        
        # Disclaimer
        st.markdown("### ⚠️ Disclaimer")
        st.caption("For educational purposes only. Does not perform actual hacking.")
        
        return page

# =============================================================================
# PAGE: HOME
# =============================================================================

def render_home_page():
    """Render the home page."""
    st.markdown('<h1 class="main-header">🔒 Cyber-Sight</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Global ML & AI Based Cyber Crime Detection and Safety Platform</p>', unsafe_allow_html=True)
    
    # Hero Section
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯</h3>
            <h4>Threat Detection</h4>
            <p>ML-powered classification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔗</h3>
            <h4>URL Checker</h4>
            <p>Phishing & malware detection</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖</h3>
            <h4>AI Chatbot</h4>
            <p>Cybersecurity Q&A</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>📊</h3>
            <h4>Insights</h4>
            <p>Global threat analytics</p>
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
        
        ### 🔗 URL Safety Analysis
        - Comprehensive URL scanning
        - Heuristic + ML hybrid approach
        - Suspicious pattern detection
        - Brand impersonation alerts
        """)
    
    with col2:
        st.markdown("""
        ### 🤖 AI Security Assistant
        - Natural language Q&A chatbot
        - Cybersecurity education
        - Best practices guidance
        - Incident response help
        
        ### 📊 Global Insights
        - Worldwide threat visualization
        - Attack type distribution
        - Trend analysis
        - Risk statistics
        """)
    
    st.markdown("---")
    
    # Quick Actions
    st.markdown("## 🚀 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info("**Check a URL**\n\nGo to URL Checker to analyze any suspicious link.")
    
    with col2:
        st.info("**Ask a Question**\n\nUse the AI Chatbot for cybersecurity guidance.")
    
    with col3:
        st.info("**Explore Data**\n\nView Dataset Insights for threat analytics.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🔒 Cyber-Sight | Global Cyber Crime Detection Platform</p>
        <p>For educational and awareness purposes only</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE: THREAT DETECTION
# =============================================================================

def render_threat_detection_page():
    """Render the threat detection page."""
    st.markdown("# 🎯 Cyber Threat Detection")
    st.markdown("Analyze potential cyber threats using our ML-powered detection system.")
    
    st.markdown("---")
    
    # Check if model is loaded
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
                    
                    # Detailed Analysis
                    st.markdown("### 🔎 Detailed Analysis")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### Analysis Findings")
                        for reason in result.reasons:
                            st.markdown(f"- {reason}")
                    
                    with col2:
                        st.markdown("#### Recommendations")
                        for rec in result.recommendations:
                            st.markdown(f"- {rec}")
                    
                    # ML Prediction Details (if available)
                    if st.session_state.model_loaded and st.session_state.model_data:
                        st.markdown("### 🤖 ML Model Prediction")
                        
                        try:
                            features = st.session_state.preprocessor.extract_url_features(url)
                            feature_cols = st.session_state.model_data.get('feature_columns', [])
                            scaler = st.session_state.model_data.get('scaler')
                            attack_model = st.session_state.model_data.get('attack_model')
                            label_encoders = st.session_state.model_data.get('label_encoders', {})
                            
                            feature_values = [features.get(col, 0) for col in feature_cols]
                            scaled_features = scaler.transform([feature_values])
                            
                            prediction = attack_model.predict(scaled_features)[0]
                            probabilities = attack_model.predict_proba(scaled_features)[0]
                            
                            attack_type = label_encoders['attack_type'].inverse_transform([prediction])[0]
                            
                            # Show probabilities
                            prob_df = pd.DataFrame({
                                'Attack Type': label_encoders['attack_type'].classes_,
                                'Probability': probabilities
                            }).sort_values('Probability', ascending=False)
                            
                            fig = px.bar(
                                prob_df,
                                x='Attack Type',
                                y='Probability',
                                color='Probability',
                                color_continuous_scale='RdYlGn_r',
                                title='Threat Type Probability Distribution'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"ML prediction error: {e}")
            else:
                st.warning("Please enter a URL to analyze.")
    
    elif input_type == "Manual Feature Input":
        st.markdown("#### Enter URL Features Manually")
        
        col1, col2 = st.columns(2)
        
        with col1:
            domain_length = st.number_input("Domain Length", min_value=1, max_value=100, value=15)
            has_https = st.selectbox("Uses HTTPS?", [1, 0], format_func=lambda x: "Yes" if x else "No")
            has_ip = st.selectbox("Has IP Address?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            num_dots = st.number_input("Number of Dots", min_value=0, max_value=20, value=1)
            num_hyphens = st.number_input("Number of Hyphens", min_value=0, max_value=20, value=0)
        
        with col2:
            num_slashes = st.number_input("Number of Slashes", min_value=0, max_value=20, value=1)
            num_digits = st.number_input("Number of Digits", min_value=0, max_value=50, value=0)
            url_length = st.number_input("URL Length", min_value=5, max_value=500, value=30)
            has_suspicious = st.selectbox("Has Suspicious Keywords?", [0, 1], format_func=lambda x: "Yes" if x else "No")
            num_underscores = st.number_input("Number of Underscores", min_value=0, max_value=20, value=0)
        
        if st.button("🎯 Predict Threat", type="primary"):
            if st.session_state.model_loaded:
                try:
                    feature_cols = st.session_state.model_data.get('feature_columns', [])
                    features = {
                        'domain_length': domain_length,
                        'has_https': has_https,
                        'has_ip': has_ip,
                        'num_dots': num_dots,
                        'num_hyphens': num_hyphens,
                        'num_slashes': num_slashes,
                        'num_digits': num_digits,
                        'url_length': url_length,
                        'has_suspicious_keywords': has_suspicious,
                        'num_underscores': num_underscores
                    }
                    
                    feature_values = [features.get(col, 0) for col in feature_cols]
                    scaler = st.session_state.model_data.get('scaler')
                    scaled = scaler.transform([feature_values])
                    
                    attack_model = st.session_state.model_data.get('attack_model')
                    risk_model = st.session_state.model_data.get('risk_model')
                    label_encoders = st.session_state.model_data.get('label_encoders', {})
                    
                    attack_pred = attack_model.predict(scaled)[0]
                    attack_proba = attack_model.predict_proba(scaled)[0]
                    risk_pred = risk_model.predict(scaled)[0]
                    
                    attack_type = label_encoders['attack_type'].inverse_transform([attack_pred])[0]
                    risk_level = label_encoders['risk_level'].inverse_transform([risk_pred])[0]
                    
                    st.markdown("### 📊 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Threat Type", attack_type.upper())
                    with col2:
                        st.metric("Risk Level", risk_level.upper())
                    with col3:
                        st.metric("Confidence", f"{max(attack_proba):.1%}")
                    
                except Exception as e:
                    st.error(f"Prediction error: {e}")
            else:
                st.error("ML Model not loaded. Please train the model first.")
    
    else:  # Incident Description
        st.markdown("#### Describe the Incident")
        
        incident = st.text_area(
            "Describe what happened",
            placeholder="Example: I received an email claiming to be from my bank asking me to verify my account by clicking a link...",
            height=150
        )
        
        if st.button("🔍 Analyze Incident", type="primary"):
            if incident:
                # Simple keyword-based analysis
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
                    
                    st.markdown("### 💡 Recommendations")
                    
                    recommendations = {
                        'phishing': [
                            "Do not click any links in suspicious emails",
                            "Verify the sender's email address carefully",
                            "Contact the organization directly using official channels",
                            "Report the email as phishing"
                        ],
                        'malware': [
                            "Do not download or open suspicious files",
                            "Run a full antivirus scan",
                            "Keep your software updated",
                            "Consider professional IT support if infected"
                        ],
                        'hacking': [
                            "Change all passwords immediately",
                            "Enable two-factor authentication",
                            "Check for unauthorized access in account logs",
                            "Report to relevant authorities"
                        ],
                        'scam': [
                            "Do not send money or personal information",
                            "If it sounds too good to be true, it probably is",
                            "Block and report the sender",
                            "Warn others about similar scams"
                        ]
                    }
                    
                    for rec in recommendations.get(detected_type, []):
                        st.markdown(f"- {rec}")
                else:
                    st.info("Unable to detect a specific threat type. Please provide more details or consult with a cybersecurity professional.")
            else:
                st.warning("Please describe the incident.")

# =============================================================================
# PAGE: URL CHECKER
# =============================================================================

def render_url_checker_page():
    """Render the URL checker page."""
    st.markdown("# 🔗 URL Safety Checker")
    st.markdown("Check if a URL is safe, suspicious, or malicious using our comprehensive analysis system.")
    
    st.markdown("---")
    
    # URL Input
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url = st.text_input(
            "Enter URL to check",
            placeholder="https://example.com/path",
            label_visibility="collapsed"
        )
    
    with col2:
        check_btn = st.button("🔍 Check URL", type="primary", use_container_width=True)
    
    # Quick check suggestions
    st.markdown("**Try these examples:**")
    example_cols = st.columns(4)
    
    example_urls = [
        "https://google.com",
        "http://paypal-secure-login.tk/verify",
        "http://192.168.1.1/admin",
        "https://microsoft.com/security"
    ]
    
    for i, example in enumerate(example_urls):
        with example_cols[i]:
            if st.button(example[:25] + "...", key=f"ex_{i}", use_container_width=True):
                url = example
                check_btn = True
    
    st.markdown("---")
    
    # Analysis
    if check_btn and url:
        with st.spinner("🔍 Analyzing URL..."):
            result = st.session_state.url_checker.check_url(url)
        
        # Main Result
        st.markdown("## 📊 Analysis Result")
        
        display_url_result(result)
        
        st.markdown("---")
        
        # Detailed Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🔎 Analysis Details")
            for reason in result.reasons:
                if reason.startswith("✓"):
                    st.success(reason)
                elif reason.startswith("⚠"):
                    st.warning(reason)
                elif reason.startswith("🚨") or reason.startswith("🤖"):
                    st.error(reason)
                else:
                    st.info(reason)
        
        with col2:
            st.markdown("### 💡 Recommendations")
            for rec in result.recommendations:
                st.markdown(f"- {rec}")
        
        st.markdown("---")
        
        # URL Breakdown
        st.markdown("### 🔬 URL Breakdown")
        
        from urllib.parse import urlparse
        try:
            parsed = urlparse(url if '://' in url else f'http://{url}')
            
            breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
            
            with breakdown_col1:
                st.markdown("**Protocol**")
                protocol_status = "✅ Secure" if parsed.scheme == "https" else "⚠️ Not Secure"
                st.code(f"{parsed.scheme}:// ({protocol_status})")
            
            with breakdown_col2:
                st.markdown("**Domain**")
                st.code(parsed.netloc)
            
            with breakdown_col3:
                st.markdown("**Path**")
                st.code(parsed.path or "/")
        except:
            st.warning("Unable to parse URL structure")
        
        # Feature Analysis
        st.markdown("### 📈 Feature Analysis")
        
        features = st.session_state.preprocessor.extract_url_features(url)
        
        feature_df = pd.DataFrame([
            {"Feature": "URL Length", "Value": features.get('url_length', 0), "Risk": "High" if features.get('url_length', 0) > 100 else "Low"},
            {"Feature": "Domain Length", "Value": features.get('domain_length', 0), "Risk": "High" if features.get('domain_length', 0) > 30 else "Low"},
            {"Feature": "Uses HTTPS", "Value": "Yes" if features.get('has_https', 0) else "No", "Risk": "Low" if features.get('has_https', 0) else "Medium"},
            {"Feature": "Has IP Address", "Value": "Yes" if features.get('has_ip', 0) else "No", "Risk": "High" if features.get('has_ip', 0) else "Low"},
            {"Feature": "Suspicious Keywords", "Value": features.get('suspicious_keyword_count', 0), "Risk": "High" if features.get('suspicious_keyword_count', 0) > 0 else "Low"},
            {"Feature": "Number of Dots", "Value": features.get('num_dots', 0), "Risk": "Medium" if features.get('num_dots', 0) > 3 else "Low"},
            {"Feature": "Number of Hyphens", "Value": features.get('num_hyphens', 0), "Risk": "Medium" if features.get('num_hyphens', 0) > 2 else "Low"},
        ])
        
        st.dataframe(feature_df, use_container_width=True, hide_index=True)
    
    elif check_btn:
        st.warning("Please enter a URL to check.")
    
    # Batch Check Section
    st.markdown("---")
    st.markdown("### 📋 Batch URL Check")
    
    urls_text = st.text_area(
        "Enter multiple URLs (one per line)",
        placeholder="https://example1.com\nhttps://example2.com\nhttp://suspicious-site.tk",
        height=100
    )
    
    if st.button("🔍 Check All URLs", key="batch_check"):
        if urls_text:
            urls = [u.strip() for u in urls_text.split('\n') if u.strip()]
            
            if urls:
                results_data = []
                
                progress = st.progress(0)
                for i, u in enumerate(urls):
                    result = st.session_state.url_checker.check_url(u)
                    results_data.append({
                        'URL': u[:50] + ('...' if len(u) > 50 else ''),
                        'Status': result.safety_status,
                        'Risk Level': result.risk_level,
                        'Threat Type': result.threat_type,
                        'Confidence': f"{result.confidence:.0%}"
                    })
                    progress.progress((i + 1) / len(urls))
                
                results_df = pd.DataFrame(results_data)
                
                # Color code the dataframe
                def color_status(val):
                    if val == 'SAFE':
                        return 'background-color: #10b981; color: white'
                    elif val == 'SUSPICIOUS':
                        return 'background-color: #f59e0b; color: white'
                    else:
                        return 'background-color: #ef4444; color: white'
                
                styled_df = results_df.style.applymap(color_status, subset=['Status'])
                st.dataframe(styled_df, use_container_width=True, hide_index=True)
                
                # Summary
                safe_count = len([r for r in results_data if r['Status'] == 'SAFE'])
                suspicious_count = len([r for r in results_data if r['Status'] == 'SUSPICIOUS'])
                malicious_count = len([r for r in results_data if r['Status'] == 'MALICIOUS'])
                
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("✅ Safe", safe_count)
                with summary_col2:
                    st.metric("⚠️ Suspicious", suspicious_count)
                with summary_col3:
                    st.metric("🚫 Malicious", malicious_count)
        else:
            st.warning("Please enter URLs to check.")

# =============================================================================
# PAGE: AI CHATBOT
# =============================================================================

def render_chatbot_page():
    """Render the AI chatbot page."""
    st.markdown("# 🤖 AI Cyber Security Assistant")
    st.markdown("Ask me anything about cybersecurity, threats, and staying safe online!")
    
    st.markdown("---")
    
    # Chat Container
    chat_container = st.container()
    
    # Display chat history
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
    
    # Suggested Questions
    if not st.session_state.chat_history:
        st.markdown("### 💡 Suggested Questions")
        suggestions = st.session_state.chatbot.get_suggestions()
        
        suggestion_cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with suggestion_cols[i]:
                if st.button(suggestion, key=f"sug_{i}", use_container_width=True):
                    # Process the suggestion
                    st.session_state.chat_history.append({'role': 'user', 'content': suggestion})
                    response = st.session_state.chatbot.chat(suggestion)
                    st.session_state.chat_history.append({'role': 'assistant', 'content': response.response})
                    st.rerun()
    
    # Input Section
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Type your message...",
            placeholder="Ask me about phishing, passwords, malware, or any cybersecurity topic...",
            label_visibility="collapsed",
            key="chat_input"
        )
    
    with col2:
        send_btn = st.button("Send 📤", type="primary", use_container_width=True)
    
    # Process input
    if send_btn and user_input:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
        # Check for quick answer first
        quick_answer = QuickResponder.get_quick_answer(user_input)
        
        if quick_answer:
            response_text = quick_answer
        else:
            # Get chatbot response
            response = st.session_state.chatbot.chat(user_input)
            response_text = response.response
        
        # Add bot response
        st.session_state.chat_history.append({'role': 'assistant', 'content': response_text})
        
        st.rerun()
    
    # Controls
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chatbot.reset_conversation()
            st.rerun()
    
    with col2:
        if st.session_state.chat_history:
            chat_text = "\n".join([f"{'User' if m['role']=='user' else 'Bot'}: {m['content']}" for m in st.session_state.chat_history])
            st.download_button(
                "📥 Download Chat",
                chat_text,
                file_name="cybersight_chat.txt",
                use_container_width=True
            )
    
    with col3:
        st.markdown("**Topics I can help with:**")
        st.caption("Phishing • Malware • Passwords • VPN • 2FA • Safe Browsing")

# =============================================================================
# PAGE: DATASET INSIGHTS
# =============================================================================

def render_insights_page():
    """Render the dataset insights page."""
    st.markdown("# 📊 Dataset Insights")
    st.markdown("Explore global cyber threat patterns and statistics.")
    
    st.markdown("---")
    
    # Load data
    df = load_dataset()
    
    if df is None:
        st.error("Dataset not found. Please ensure 'data/cybercrime_dataset.csv' exists.")
        return
    
    # Overview Metrics
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
            st.metric("High Risk Entries", high_risk)
    
    st.markdown("---")
    
    # Visualizations
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        # Attack Type Distribution
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
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        # Risk Level Distribution
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
            fig.update_layout(showlegend=False, xaxis_title="Risk Level", yaxis_title="Count")
            st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Geographic Distribution
    if 'country' in df.columns:
        st.markdown("### 🌍 Geographic Distribution")
        
        country_counts = df['country'].value_counts().head(15)
        
        fig = px.bar(
            x=country_counts.values,
            y=country_counts.index,
            orientation='h',
            title="Top 15 Countries by Cyber Threat Reports",
            color=country_counts.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, xaxis_title="Count", yaxis_title="Country")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Attack Type by Risk Level
    if 'attack_type' in df.columns and 'risk_level' in df.columns:
        st.markdown("### 📊 Attack Types by Risk Level")
        
        cross_tab = pd.crosstab(df['attack_type'], df['risk_level'])
        
        fig = px.bar(
            cross_tab,
            barmode='group',
            title="Attack Type Distribution by Risk Level",
            color_discrete_map={'low': '#10b981', 'medium': '#f59e0b', 'high': '#ef4444'}
        )
        fig.update_layout(xaxis_title="Attack Type", yaxis_title="Count", legend_title="Risk Level")
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # URL Feature Analysis
    st.markdown("### 🔗 URL Feature Analysis")
    
    feature_cols = ['domain_length', 'url_length', 'num_dots', 'num_hyphens', 'num_digits']
    available_features = [col for col in feature_cols if col in df.columns]
    
    if available_features and 'attack_type' in df.columns:
        feature_to_analyze = st.selectbox("Select Feature to Analyze", available_features)
        
        fig = px.box(
            df,
            x='attack_type',
            y=feature_to_analyze,
            color='attack_type',
            title=f"{feature_to_analyze.replace('_', ' ').title()} by Attack Type"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Time Analysis (if timestamp exists)
    if 'timestamp' in df.columns:
        st.markdown("### 📅 Temporal Analysis")
        
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['hour'] = df['timestamp'].dt.hour
            
            hourly_counts = df.groupby('hour').size()
            
            fig = px.line(
                x=hourly_counts.index,
                y=hourly_counts.values,
                title="Cyber Threats by Hour of Day",
                markers=True
            )
            fig.update_layout(xaxis_title="Hour (24h)", yaxis_title="Number of Threats")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.info("Unable to parse timestamp data for temporal analysis.")
    
    st.markdown("---")
    
    # Data Sample
    st.markdown("### 📋 Data Sample")
    
    sample_size = st.slider("Number of rows to display", 5, 50, 10)
    st.dataframe(df.head(sample_size), use_container_width=True)
    
    # Download Data
    st.markdown("---")
    
    csv = df.to_csv(index=False)
    st.download_button(
        "📥 Download Full Dataset",
        csv,
        "cybercrime_dataset.csv",
        "text/csv",
        key='download-csv'
    )

# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
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

if __name__ == "__main__":
    main()
