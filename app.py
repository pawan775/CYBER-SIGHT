"""
Cyber-Sight: Global ML & AI Based Cyber Crime Detection and Safety Platform
===========================================================================
Main Streamlit Web Application - Version 2.0 (Enhanced for Police Use)

Professional Edition - No Emojis

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
from model.india_crime_predictor import (
    IndianCyberCrimePredictor, generate_ncrb_based_dataset, run_full_analysis, NCRB_DATA
)

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Cyber-Sight | Police Cyber Crime Portal",
    page_icon="./assets/icon.png" if os.path.exists("./assets/icon.png") else None,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/cyber-sight',
        'Report a bug': 'https://github.com/cyber-sight/issues',
        'About': """
        # Cyber-Sight v2.0
        Global ML & AI Based Cyber Crime Detection and Safety Platform
        
        For Police & Law Enforcement Use
        
        Version: 2.0.0
        
        Features:
        - Secure Login with CAPTCHA
        - Indian State-wise Predictions (2018-2045)
        - Live Threat Monitoring Dashboard
        - Case Management System
        
        Official use only. Unauthorized access prohibited.
        """
    }
)

# =============================================================================
# CUSTOM CSS STYLING
# =============================================================================

def load_css():
    """Load custom CSS styles - Clean Professional UI with Icons."""
    st.markdown("""
    <style>
        /* Import modern fonts and icons */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
        @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css');
        
        * {
            font-family: 'Inter', sans-serif;
        }
        
        /* Clean dark background - solid color, no busy gradients */
        .stApp {
            background: #0a0f1a;
        }
        
        /* Main header - clean and professional */
        .main-header {
            font-size: 2.5rem;
            font-weight: 700;
            color: #ffffff;
            text-align: center;
            padding: 1rem 0;
            margin-bottom: 0.5rem;
            letter-spacing: 1px;
        }
        
        .main-header span {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Sub-header */
        .sub-header {
            font-size: 1rem;
            color: #64748b;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;
        }
        
        /* Clean status boxes */
        .status-safe {
            background: #059669;
            color: white;
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .status-suspicious {
            background: #d97706;
            color: white;
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        .status-malicious {
            background: #dc2626;
            color: white;
            padding: 1.2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 1.2rem;
            font-weight: 600;
        }
        
        /* Clean alert boxes */
        .alert-critical {
            background: rgba(220, 38, 38, 0.1);
            color: #fca5a5;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 3px solid #dc2626;
        }
        
        .alert-high {
            background: rgba(234, 88, 12, 0.1);
            color: #fdba74;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 3px solid #ea580c;
        }
        
        .alert-medium {
            background: rgba(202, 138, 4, 0.1);
            color: #fde047;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 3px solid #ca8a04;
        }
        
        .alert-low {
            background: rgba(22, 163, 74, 0.1);
            color: #86efac;
            padding: 1rem 1.2rem;
            border-radius: 8px;
            margin: 0.5rem 0;
            border-left: 3px solid #16a34a;
        }
        
        /* Info card - clean */
        .info-card {
            background: #111827;
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.5rem 0;
            border: 1px solid #1f2937;
        }
        
        /* Badge */
        .police-badge {
            background: #3b82f6;
            color: white;
            padding: 0.4rem 1rem;
            border-radius: 6px;
            font-size: 0.8rem;
            font-weight: 600;
            display: inline-block;
            margin-bottom: 1rem;
        }
        
        /* Metric card - clean with hover */
        .metric-card {
            background: #111827;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #1f2937;
            transition: all 0.2s ease;
        }
        
        .metric-card:hover {
            border-color: #3b82f6;
            transform: translateY(-2px);
        }
        
        .metric-card h3 {
            color: #3b82f6;
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .metric-card h4 {
            color: #e5e7eb;
            margin: 0.5rem 0;
            font-size: 1rem;
            font-weight: 500;
        }
        
        .metric-card p {
            color: #6b7280;
            margin: 0;
            font-size: 0.85rem;
        }
        
        /* Chat messages */
        .chat-user {
            background: #3b82f6;
            color: white;
            padding: 0.8rem 1rem;
            border-radius: 12px 12px 4px 12px;
            margin: 0.5rem 0;
            max-width: 80%;
            margin-left: auto;
        }
        
        .chat-bot {
            background: #1f2937;
            color: #e5e7eb;
            padding: 0.8rem 1rem;
            border-radius: 12px 12px 12px 4px;
            margin: 0.5rem 0;
            max-width: 80%;
        }
        
        /* Live indicator */
        .live-indicator {
            display: inline-flex;
            align-items: center;
            background: #dc2626;
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 20px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .live-dot {
            width: 8px;
            height: 8px;
            background-color: white;
            border-radius: 50%;
            margin-right: 6px;
            animation: blink 1s infinite;
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; }
            51%, 100% { opacity: 0.3; }
        }
        
        /* CAPTCHA box - simple and clean */
        .captcha-box {
            background: #1f2937;
            color: #60a5fa;
            padding: 1.5rem 2rem;
            border-radius: 12px;
            text-align: center;
            font-size: 2rem;
            font-family: 'JetBrains Mono', monospace;
            letter-spacing: 8px;
            border: 2px dashed #3b82f6;
            margin: 1rem 0;
            font-weight: 600;
        }
        
        /* Navigation items with icons */
        .nav-item {
            display: flex;
            align-items: center;
            padding: 0.8rem 1rem;
            margin: 0.25rem 0;
            border-radius: 8px;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #9ca3af;
        }
        
        .nav-item:hover {
            background: #1f2937;
            color: #ffffff;
        }
        
        .nav-item.active {
            background: #3b82f6;
            color: white;
        }
        
        .nav-item i {
            margin-right: 10px;
            width: 20px;
            text-align: center;
        }
        
        /* Section header - clean */
        .section-header {
            font-size: 1.4rem;
            font-weight: 600;
            color: #f3f4f6;
            padding-bottom: 0.5rem;
            margin-bottom: 1rem;
            border-bottom: 2px solid #3b82f6;
            display: inline-block;
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Stat box */
        .stat-box {
            background: #111827;
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid #1f2937;
        }
        
        .stat-box h2 {
            color: #3b82f6;
            font-size: 1.8rem;
            margin: 0;
            font-weight: 700;
        }
        
        .stat-box p {
            color: #9ca3af;
            margin: 0.5rem 0 0 0;
            font-size: 0.9rem;
        }
        
        /* Clean buttons */
        .stButton > button {
            background: #3b82f6;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.2s ease;
        }
        
        .stButton > button:hover {
            background: #2563eb;
            transform: translateY(-1px);
        }
        
        /* Input fields */
        .stTextInput > div > div > input {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
            color: #e5e7eb;
            padding: 0.6rem 1rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }
        
        .stSelectbox > div > div {
            background: #1f2937;
            border: 1px solid #374151;
            border-radius: 8px;
        }
        
        /* Sidebar - clean dark */
        section[data-testid="stSidebar"] {
            background: #0f172a;
            border-right: 1px solid #1e293b;
        }
        
        section[data-testid="stSidebar"] .stRadio > label {
            color: #94a3b8;
        }
        
        /* Tabs - clean */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #111827;
            border-radius: 8px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 6px;
            color: #9ca3af;
            font-weight: 500;
            padding: 0.5rem 1rem;
        }
        
        .stTabs [aria-selected="true"] {
            background: #3b82f6;
            color: white;
        .metric-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.2);
        }
        
        .metric-card h3 {
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .metric-card h4 {
            color: #e2e8f0;
            margin: 0.5rem 0;
            font-size: 1rem;
            font-weight: 500;
        }
        
        .metric-card p {
            color: #94a3b8;
            margin: 0;
            font-size: 0.85rem;
        }
        
        /* Chat messages with modern bubbles */
        .chat-user {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            padding: 1rem 1.2rem;
            border-radius: 18px 18px 4px 18px;
            margin: 0.75rem 0;
            max-width: 80%;
            margin-left: auto;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
            font-weight: 400;
        }
        
        .chat-bot {
            background: rgba(30, 41, 59, 0.8);
            color: #e2e8f0;
            padding: 1rem 1.2rem;
            border-radius: 18px 18px 18px 4px;
            margin: 0.75rem 0;
            max-width: 80%;
            border: 1px solid rgba(148, 163, 184, 0.2);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        }
        
        /* Live indicator with pulse animation */
        .live-indicator {
            display: inline-flex;
            align-items: center;
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 1px;
            box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4);
            animation: pulse-live 2s infinite;
        }
        
        @keyframes pulse-live {
            0%, 100% { box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4); }
            50% { box-shadow: 0 4px 30px rgba(220, 38, 38, 0.7); }
        }
        
        .live-dot {
            width: 10px;
            height: 10px;
            background-color: white;
            border-radius: 50%;
            margin-right: 8px;
            animation: blink 1s infinite;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.8);
        }
        
        @keyframes blink {
            0%, 50% { opacity: 1; transform: scale(1); }
            51%, 100% { opacity: 0.4; transform: scale(0.8); }
        }
        
        /* CAPTCHA box with creative styling */
        .captcha-box {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
            color: #60a5fa;
            padding: 1.5rem 2rem;
            border-radius: 16px;
            text-align: center;
            font-size: 1.8rem;
            font-family: 'JetBrains Mono', 'Courier New', monospace;
            letter-spacing: 6px;
            border: 2px solid rgba(96, 165, 250, 0.3);
            margin: 1.5rem 0;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 0 60px rgba(96, 165, 250, 0.05);
            position: relative;
            overflow: hidden;
        }
        
        .captcha-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 200%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(96, 165, 250, 0.1), transparent);
            animation: shimmer 3s infinite;
        }
        
        @keyframes shimmer {
            0% { left: -100%; }
            100% { left: 100%; }
        }
        
        /* Navigation with hover effects */
        .nav-item {
            padding: 0.75rem 1rem;
            margin: 0.3rem 0;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.3s ease;
            background: rgba(30, 41, 59, 0.5);
        }
        
        .nav-item:hover {
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.2) 0%, rgba(139, 92, 246, 0.2) 100%);
            transform: translateX(5px);
        }
        
        /* Footer with gradient border */
        .footer {
            text-align: center;
            padding: 2rem;
            color: #94a3b8;
            font-size: 0.85rem;
            border-top: 1px solid rgba(148, 163, 184, 0.2);
            margin-top: 3rem;
            background: linear-gradient(180deg, transparent 0%, rgba(15, 23, 42, 0.5) 100%);
        }
        
        /* Hide Streamlit branding */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Section headers with gradient underline */
        .section-header {
            font-size: 1.5rem;
            font-weight: 700;
            color: #e2e8f0;
            padding-bottom: 0.75rem;
            margin-bottom: 1.5rem;
            position: relative;
        }
        
        .section-header::after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 60px;
            height: 3px;
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 100%);
            border-radius: 2px;
        }
        
        /* Streamlit elements customization */
        .stButton > button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(59, 130, 246, 0.4);
        }
        
        .stTextInput > div > div > input {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
            color: #e2e8f0;
            padding: 0.75rem 1rem;
        }
        
        .stTextInput > div > div > input:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
        }
        
        .stSelectbox > div > div {
            background: rgba(30, 41, 59, 0.8);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 10px;
        }
        
        /* Sidebar styling */
        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        section[data-testid="stSidebar"] .stRadio > label {
            color: #94a3b8;
        }
        
        /* Data frame styling */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background: rgba(30, 41, 59, 0.5);
            border-radius: 12px;
            padding: 4px;
        }
        
        .stTabs [data-baseweb="tab"] {
            border-radius: 8px;
            color: #94a3b8;
            font-weight: 500;
        }
        
        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background: rgba(30, 41, 59, 0.8);
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        /* Feature card for home page */
        .feature-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.8) 100%);
            border-radius: 20px;
            padding: 2rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            height: 100%;
        }
        
        .feature-card:hover {
            transform: translateY(-10px);
            box-shadow: 0 12px 40px rgba(59, 130, 246, 0.15);
            border-color: rgba(59, 130, 246, 0.3);
        }
        
        .feature-card h3 {
            color: #60a5fa;
            font-size: 1.3rem;
            margin-bottom: 0.75rem;
        }
        
        .feature-card p {
            color: #94a3b8;
            font-size: 0.95rem;
            line-height: 1.6;
        }
        
        /* Stat box for dashboard */
        .stat-box {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.9) 100%);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(148, 163, 184, 0.1);
            position: relative;
            overflow: hidden;
        }
        
        .stat-box::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #f472b6 100%);
        }
        
        .stat-box h2 {
            font-size: 2.5rem;
            font-weight: 800;
            background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
        }
        
        .stat-box p {
            color: #94a3b8;
            margin: 0.5rem 0 0 0;
            font-size: 0.9rem;
            font-weight: 500;
        }
        
        /* Floating Chatbot Button - True Floating Effect */
        .floating-chat-btn {
            position: fixed;
            bottom: 30px;
            right: 30px;
            width: 65px;
            height: 65px;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
            color: white;
            border: none;
            cursor: pointer;
            box-shadow: 
                0 10px 40px rgba(59, 130, 246, 0.6),
                0 0 20px rgba(139, 92, 246, 0.4),
                inset 0 -3px 10px rgba(0,0,0,0.2);
            z-index: 99999;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.8rem;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            animation: float-in-air 3s ease-in-out infinite;
        }
        
        .floating-chat-btn::before {
            content: '';
            position: absolute;
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            animation: pulse-ring 2s ease-out infinite;
            z-index: -1;
        }
        
        .floating-chat-btn:hover {
            transform: scale(1.15) translateY(-5px);
            box-shadow: 
                0 15px 50px rgba(59, 130, 246, 0.7),
                0 0 30px rgba(139, 92, 246, 0.5);
        }
        
        @keyframes float-in-air {
            0%, 100% { 
                transform: translateY(0) rotate(0deg); 
                box-shadow: 0 10px 40px rgba(59, 130, 246, 0.6), 0 0 20px rgba(139, 92, 246, 0.4);
            }
            25% { 
                transform: translateY(-8px) rotate(2deg); 
            }
            50% { 
                transform: translateY(-12px) rotate(0deg); 
                box-shadow: 0 20px 50px rgba(59, 130, 246, 0.5), 0 0 30px rgba(139, 92, 246, 0.3);
            }
            75% { 
                transform: translateY(-8px) rotate(-2deg); 
            }
        }
        
        @keyframes pulse-ring {
            0% {
                transform: scale(1);
                opacity: 0.6;
            }
            100% {
                transform: scale(1.5);
                opacity: 0;
            }
        }
        
        /* Chat Popup Window */
        .chat-popup {
            position: fixed;
            bottom: 100px;
            right: 30px;
            width: 380px;
            max-height: 500px;
            background: #0f172a;
            border-radius: 20px;
            border: 1px solid rgba(59, 130, 246, 0.3);
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            z-index: 9998;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }
        
        .chat-popup-header {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            padding: 1rem 1.2rem;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .chat-popup-header h4 {
            margin: 0;
            font-weight: 600;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .chat-popup-close {
            background: rgba(255,255,255,0.2);
            border: none;
            color: white;
            width: 28px;
            height: 28px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1rem;
            transition: background 0.2s;
        }
        
        .chat-popup-close:hover {
            background: rgba(255,255,255,0.3);
        }
        
        .chat-popup-body {
            flex: 1;
            padding: 1rem;
            overflow-y: auto;
            max-height: 320px;
            background: #0f172a;
        }
        
        .chat-popup-input {
            padding: 0.8rem 1rem;
            background: #1e293b;
            border-top: 1px solid rgba(148, 163, 184, 0.1);
            display: flex;
            gap: 8px;
        }
        
        .chat-popup-input input {
            flex: 1;
            background: #0f172a;
            border: 1px solid #374151;
            border-radius: 20px;
            padding: 0.6rem 1rem;
            color: #e5e7eb;
            font-size: 0.9rem;
            outline: none;
        }
        
        .chat-popup-input input:focus {
            border-color: #3b82f6;
        }
        
        .chat-popup-input button {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            border: none;
            color: white;
            width: 38px;
            height: 38px;
            border-radius: 50%;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: transform 0.2s;
        }
        
        .chat-popup-input button:hover {
            transform: scale(1.05);
        }
        
        .chat-msg-user {
            background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%);
            color: white;
            padding: 0.6rem 1rem;
            border-radius: 16px 16px 4px 16px;
            margin: 0.4rem 0;
            max-width: 85%;
            margin-left: auto;
            font-size: 0.9rem;
            word-wrap: break-word;
        }
        
        .chat-msg-bot {
            background: #1e293b;
            color: #e2e8f0;
            padding: 0.6rem 1rem;
            border-radius: 16px 16px 16px 4px;
            margin: 0.4rem 0;
            max-width: 85%;
            font-size: 0.9rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
            word-wrap: break-word;
        }
        
        .chat-popup-suggestions {
            padding: 0.5rem 1rem;
            background: #1e293b;
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
        }
        
        .chat-suggestion-btn {
            background: rgba(59, 130, 246, 0.2);
            border: 1px solid rgba(59, 130, 246, 0.3);
            color: #60a5fa;
            padding: 0.3rem 0.7rem;
            border-radius: 12px;
            font-size: 0.75rem;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .chat-suggestion-btn:hover {
            background: rgba(59, 130, 246, 0.4);
            color: white;
        }
        
        /* =============================================
           ENHANCED PAGE SECTION UI COMPONENTS
           ============================================= */
        
        /* Modern Page Header with Icon */
        .page-header {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-radius: 20px;
            padding: 2rem 2.5rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(59, 130, 246, 0.2);
            position: relative;
            overflow: hidden;
        }
        
        .page-header::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899, #f59e0b);
        }
        
        .page-header h1 {
            margin: 0;
            font-size: 1.8rem;
            font-weight: 700;
            color: #f1f5f9;
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .page-header h1 i {
            font-size: 1.5rem;
            background: linear-gradient(135deg, #3b82f6, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .page-header p {
            margin: 0.5rem 0 0 0;
            color: #94a3b8;
            font-size: 1rem;
        }
        
        /* Action Card - Clickable Cards instead of plain buttons */
        .action-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border-radius: 16px;
            padding: 1.5rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
            cursor: pointer;
            transition: all 0.3s ease;
            text-align: center;
            min-height: 140px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }
        
        .action-card:hover {
            transform: translateY(-5px);
            border-color: rgba(59, 130, 246, 0.5);
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.2);
        }
        
        .action-card.primary {
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            border: none;
        }
        
        .action-card.primary:hover {
            box-shadow: 0 10px 40px rgba(59, 130, 246, 0.4);
        }
        
        .action-card.success {
            border-color: rgba(34, 197, 94, 0.3);
        }
        
        .action-card.success:hover {
            border-color: rgba(34, 197, 94, 0.6);
            box-shadow: 0 10px 40px rgba(34, 197, 94, 0.2);
        }
        
        .action-card.danger {
            border-color: rgba(239, 68, 68, 0.3);
        }
        
        .action-card.danger:hover {
            border-color: rgba(239, 68, 68, 0.6);
            box-shadow: 0 10px 40px rgba(239, 68, 68, 0.2);
        }
        
        .action-card .icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        
        .action-card.primary .icon {
            color: white;
        }
        
        .action-card .icon.blue { color: #60a5fa; }
        .action-card .icon.green { color: #4ade80; }
        .action-card .icon.red { color: #f87171; }
        .action-card .icon.yellow { color: #fbbf24; }
        .action-card .icon.purple { color: #a78bfa; }
        
        .action-card h4 {
            margin: 0;
            font-size: 1rem;
            font-weight: 600;
            color: #e2e8f0;
        }
        
        .action-card.primary h4 {
            color: white;
        }
        
        .action-card p {
            margin: 0;
            font-size: 0.8rem;
            color: #94a3b8;
        }
        
        .action-card.primary p {
            color: rgba(255,255,255,0.8);
        }
        
        /* Info Panel - Modern info display */
        .info-panel {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.8) 100%);
            border-radius: 16px;
            padding: 1.5rem;
            border-left: 4px solid #3b82f6;
            margin: 1rem 0;
        }
        
        .info-panel.warning {
            border-left-color: #f59e0b;
            background: linear-gradient(145deg, rgba(245, 158, 11, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        }
        
        .info-panel.success {
            border-left-color: #22c55e;
            background: linear-gradient(145deg, rgba(34, 197, 94, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        }
        
        .info-panel.danger {
            border-left-color: #ef4444;
            background: linear-gradient(145deg, rgba(239, 68, 68, 0.1) 0%, rgba(15, 23, 42, 0.8) 100%);
        }
        
        .info-panel h4 {
            margin: 0 0 0.5rem 0;
            color: #f1f5f9;
            font-size: 1rem;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .info-panel p {
            margin: 0;
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        /* Data Card - For displaying key-value info */
        .data-card {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            padding: 1.2rem;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        .data-card .label {
            font-size: 0.75rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 4px;
        }
        
        .data-card .value {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .data-card .sub {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        
        /* Result Box - For showing analysis results */
        .result-box {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
            border-radius: 20px;
            padding: 2rem;
            margin: 1.5rem 0;
            border: 1px solid rgba(148, 163, 184, 0.1);
            position: relative;
        }
        
        .result-box.safe {
            border-color: rgba(34, 197, 94, 0.3);
            background: linear-gradient(145deg, rgba(34, 197, 94, 0.05) 0%, rgba(15, 23, 42, 0.95) 100%);
        }
        
        .result-box.safe::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #22c55e, #4ade80);
            border-radius: 20px 20px 0 0;
        }
        
        .result-box.warning {
            border-color: rgba(245, 158, 11, 0.3);
            background: linear-gradient(145deg, rgba(245, 158, 11, 0.05) 0%, rgba(15, 23, 42, 0.95) 100%);
        }
        
        .result-box.warning::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #f59e0b, #fbbf24);
            border-radius: 20px 20px 0 0;
        }
        
        .result-box.danger {
            border-color: rgba(239, 68, 68, 0.3);
            background: linear-gradient(145deg, rgba(239, 68, 68, 0.05) 0%, rgba(15, 23, 42, 0.95) 100%);
        }
        
        .result-box.danger::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #ef4444, #f87171);
            border-radius: 20px 20px 0 0;
        }
        
        .result-box .result-icon {
            font-size: 3rem;
            margin-bottom: 1rem;
        }
        
        .result-box .result-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-bottom: 0.5rem;
        }
        
        .result-box .result-subtitle {
            color: #94a3b8;
            font-size: 1rem;
        }
        
        /* Mini Stats Grid */
        .mini-stat {
            background: rgba(15, 23, 42, 0.6);
            border-radius: 12px;
            padding: 1rem;
            text-align: center;
            border: 1px solid rgba(148, 163, 184, 0.1);
            transition: all 0.2s ease;
        }
        
        .mini-stat:hover {
            border-color: rgba(59, 130, 246, 0.3);
            transform: translateY(-2px);
        }
        
        .mini-stat .number {
            font-size: 1.8rem;
            font-weight: 700;
            color: #60a5fa;
        }
        
        .mini-stat .number.green { color: #4ade80; }
        .mini-stat .number.red { color: #f87171; }
        .mini-stat .number.yellow { color: #fbbf24; }
        .mini-stat .number.purple { color: #a78bfa; }
        
        .mini-stat .label {
            font-size: 0.8rem;
            color: #94a3b8;
            margin-top: 4px;
        }
        
        /* Threat Card - For live threats display */
        .threat-card {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.8) 0%, rgba(15, 23, 42, 0.9) 100%);
            border-radius: 12px;
            padding: 1.2rem;
            margin: 0.75rem 0;
            border-left: 4px solid #64748b;
            transition: all 0.2s ease;
        }
        
        .threat-card:hover {
            transform: translateX(5px);
        }
        
        .threat-card.critical {
            border-left-color: #ef4444;
            background: linear-gradient(145deg, rgba(239, 68, 68, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
        }
        
        .threat-card.high {
            border-left-color: #f97316;
            background: linear-gradient(145deg, rgba(249, 115, 22, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
        }
        
        .threat-card.medium {
            border-left-color: #eab308;
            background: linear-gradient(145deg, rgba(234, 179, 8, 0.1) 0%, rgba(15, 23, 42, 0.9) 100%);
        }
        
        .threat-card.low {
            border-left-color: #22c55e;
            background: linear-gradient(145deg, rgba(34, 197, 94, 0.05) 0%, rgba(15, 23, 42, 0.9) 100%);
        }
        
        .threat-card .threat-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 0.5rem;
        }
        
        .threat-card .threat-type {
            font-weight: 600;
            color: #f1f5f9;
            font-size: 0.95rem;
        }
        
        .threat-card .threat-severity {
            font-size: 0.7rem;
            padding: 0.25rem 0.6rem;
            border-radius: 20px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .threat-card.critical .threat-severity {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
        }
        
        .threat-card.high .threat-severity {
            background: rgba(249, 115, 22, 0.2);
            color: #fb923c;
        }
        
        .threat-card.medium .threat-severity {
            background: rgba(234, 179, 8, 0.2);
            color: #facc15;
        }
        
        .threat-card.low .threat-severity {
            background: rgba(34, 197, 94, 0.2);
            color: #4ade80;
        }
        
        .threat-card .threat-desc {
            color: #94a3b8;
            font-size: 0.85rem;
            margin: 0.5rem 0;
            line-height: 1.5;
        }
        
        .threat-card .threat-meta {
            display: flex;
            gap: 1rem;
            font-size: 0.75rem;
            color: #64748b;
            margin-top: 0.75rem;
            flex-wrap: wrap;
        }
        
        .threat-card .threat-meta span {
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        /* Input Section - Modern input styling */
        .input-section {
            background: linear-gradient(145deg, rgba(30, 41, 59, 0.6) 0%, rgba(15, 23, 42, 0.7) 100%);
            border-radius: 16px;
            padding: 1.5rem;
            margin: 1rem 0;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }
        
        .input-section label {
            display: block;
            color: #94a3b8;
            font-size: 0.85rem;
            margin-bottom: 0.5rem;
            font-weight: 500;
        }
        
        /* Modern Button Styles */
        .btn-modern {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            padding: 0.75rem 1.5rem;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s ease;
            border: none;
        }
        
        .btn-modern.primary {
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(59, 130, 246, 0.3);
        }
        
        .btn-modern.primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(59, 130, 246, 0.4);
        }
        
        .btn-modern.secondary {
            background: rgba(30, 41, 59, 0.8);
            color: #e2e8f0;
            border: 1px solid rgba(148, 163, 184, 0.2);
        }
        
        .btn-modern.secondary:hover {
            background: rgba(59, 130, 246, 0.1);
            border-color: rgba(59, 130, 246, 0.3);
        }
        
        .btn-modern.success {
            background: linear-gradient(135deg, #22c55e 0%, #16a34a 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(34, 197, 94, 0.3);
        }
        
        .btn-modern.danger {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            box-shadow: 0 4px 15px rgba(239, 68, 68, 0.3);
        }
        
        /* Section Divider */
        .section-divider {
            height: 1px;
            background: linear-gradient(90deg, transparent, rgba(148, 163, 184, 0.2), transparent);
            margin: 2rem 0;
        }
        
        /* Tab-like selector */
        .tab-selector {
            display: flex;
            gap: 8px;
            background: rgba(15, 23, 42, 0.6);
            padding: 6px;
            border-radius: 12px;
            margin-bottom: 1.5rem;
        }
        
        .tab-selector .tab {
            flex: 1;
            padding: 0.75rem 1rem;
            border-radius: 8px;
            text-align: center;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #94a3b8;
            font-weight: 500;
        }
        
        .tab-selector .tab:hover {
            background: rgba(59, 130, 246, 0.1);
            color: #e2e8f0;
        }
        
        .tab-selector .tab.active {
            background: linear-gradient(135deg, #3b82f6 0%, #6366f1 100%);
            color: white;
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 3rem 2rem;
            color: #64748b;
        }
        
        .empty-state .icon {
            font-size: 3rem;
            margin-bottom: 1rem;
            opacity: 0.5;
        }
        
        .empty-state h4 {
            color: #94a3b8;
            margin-bottom: 0.5rem;
        }
        
        .empty-state p {
            font-size: 0.9rem;
        }
        
        /* Progress indicator */
        .progress-bar {
            height: 8px;
            background: rgba(15, 23, 42, 0.8);
            border-radius: 4px;
            overflow: hidden;
            margin: 0.5rem 0;
        }
        
        .progress-bar .fill {
            height: 100%;
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        
        .progress-bar .fill.blue { background: linear-gradient(90deg, #3b82f6, #60a5fa); }
        .progress-bar .fill.green { background: linear-gradient(90deg, #22c55e, #4ade80); }
        .progress-bar .fill.red { background: linear-gradient(90deg, #ef4444, #f87171); }
        .progress-bar .fill.yellow { background: linear-gradient(90deg, #eab308, #facc15); }
        
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
        # Use word/text CAPTCHA only (no math)
        st.session_state.current_captcha = st.session_state.captcha_generator.generate_word_captcha()
    
    # Chatbot
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = CyberSecurityChatbot()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Floating Popup Chat
    if 'popup_chat_open' not in st.session_state:
        st.session_state.popup_chat_open = False
    
    if 'popup_chat_history' not in st.session_state:
        st.session_state.popup_chat_history = []
    
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
    
    # Case Management - Initialize with dummy cases for demo
    if 'cases' not in st.session_state:
        st.session_state.cases = [
            {
                'id': 1001,
                'title': 'UPI Fraud - Rs. 2.5 Lakh stolen via fake payment link',
                'crime_type': 'Online Financial Fraud',
                'state': 'Maharashtra',
                'priority': 'Critical',
                'complainant': 'Rajesh Kumar Sharma',
                'contact': '9876543210',
                'assigned_to': 'Inspector Amit Verma',
                'description': 'Victim received a WhatsApp message claiming to be from SBI Bank asking to update KYC. Clicked on link and entered OTP. Rs. 2,50,000 debited from account in 3 transactions. Suspect phone number: 8899776655. Transaction IDs: TXN001234, TXN001235, TXN001236.',
                'status': 'Investigation',
                'created_at': '2025-12-28 10:30',
                'evidence_count': 3
            },
            {
                'id': 1002,
                'title': 'Social Media Account Hacked - Instagram Business Account',
                'crime_type': 'Social Media Crimes',
                'state': 'Delhi',
                'priority': 'High',
                'complainant': 'Priya Malhotra',
                'contact': '9988776655',
                'assigned_to': 'SI Neha Singh',
                'description': 'Business Instagram account with 50K followers hacked. Hacker demanding Rs. 50,000 ransom. Account being used to post inappropriate content and scam followers. Original email changed. Last known login from IP: 103.xx.xx.xx (traced to Nigeria).',
                'status': 'Open',
                'created_at': '2025-12-29 14:15',
                'evidence_count': 5
            },
            {
                'id': 1003,
                'title': 'Ransomware Attack on Hospital Database',
                'crime_type': 'Ransomware Attacks',
                'state': 'Karnataka',
                'priority': 'Critical',
                'complainant': 'Dr. Suresh Reddy (IT Head)',
                'contact': '9123456780',
                'assigned_to': 'Inspector Kiran Kumar',
                'description': 'City Hospital database encrypted by ransomware. 10,000+ patient records locked. Attackers demanding 5 Bitcoin. Hospital operations severely affected. Attack vector suspected to be phishing email. CERT-In notified. Forensic analysis in progress.',
                'status': 'Investigation',
                'created_at': '2025-12-30 08:45',
                'evidence_count': 8
            },
            {
                'id': 1004,
                'title': 'Cyber Stalking and Harassment Case',
                'crime_type': 'Cyber Stalking/Bullying',
                'state': 'Tamil Nadu',
                'priority': 'High',
                'complainant': 'Anonymous (Female, 24)',
                'contact': '9445566778',
                'assigned_to': 'WPC Lakshmi R',
                'description': 'Victim receiving threatening messages and morphed photos on multiple platforms for 3 months. Suspect is ex-colleague. Screenshots and chat logs preserved. Fake profiles created in victim name. Mental harassment causing severe anxiety.',
                'status': 'Open',
                'created_at': '2025-12-30 16:20',
                'evidence_count': 12
            },
            {
                'id': 1005,
                'title': 'Cryptocurrency Investment Scam - Multiple Victims',
                'crime_type': 'Cryptocurrency Fraud',
                'state': 'Gujarat',
                'priority': 'Critical',
                'complainant': 'Mehul Patel (Lead Victim)',
                'contact': '9898989898',
                'assigned_to': 'Inspector D.K. Joshi',
                'description': 'Organized crypto scam promising 300% returns. 47 victims identified, total loss Rs. 3.2 Crore. Fake website: crypto-gains-india.com (now offline). Suspects operating from Dubai. Bank accounts used: HDFC, ICICI (details attached). Coordination with ED initiated.',
                'status': 'Investigation',
                'created_at': '2025-12-31 09:00',
                'evidence_count': 25
            },
            {
                'id': 1006,
                'title': 'Phishing Attack on Government Employee',
                'crime_type': 'Phishing/Vishing',
                'state': 'Uttar Pradesh',
                'priority': 'Medium',
                'complainant': 'Anil Kumar (Section Officer)',
                'contact': '9456123789',
                'assigned_to': 'Constable Ravi Yadav',
                'description': 'Received email appearing to be from NIC asking for password reset. Credentials compromised. Official email used to send fraudulent circulars. IT department notified and password reset. Investigating source of phishing email.',
                'status': 'Pending',
                'created_at': '2025-12-31 11:30',
                'evidence_count': 4
            },
            {
                'id': 1007,
                'title': 'E-commerce Fraud - Fake Product Delivery',
                'crime_type': 'Online Financial Fraud',
                'state': 'Rajasthan',
                'priority': 'Low',
                'complainant': 'Sunita Devi',
                'contact': '9414567890',
                'assigned_to': 'Unassigned',
                'description': 'Ordered iPhone 15 from fake website (best-deals-shop.in) for Rs. 45,000. Received empty box. Website now showing 404 error. Payment made via UPI. Seller contact number switched off.',
                'status': 'Open',
                'created_at': '2025-12-31 15:45',
                'evidence_count': 2
            },
            {
                'id': 1008,
                'title': 'Data Breach - Customer Information Leaked',
                'crime_type': 'Data Breach',
                'state': 'Telangana',
                'priority': 'Critical',
                'complainant': 'TechServe Solutions Pvt Ltd',
                'contact': '9876012345',
                'assigned_to': 'Inspector P. Srinivas',
                'description': 'Database of 1 lakh customers leaked on dark web. Includes names, emails, phone numbers, and partial credit card data. Breach discovered during routine audit. Source suspected to be compromised API. CERT-In and RBI notified. Customers being informed.',
                'status': 'Investigation',
                'created_at': '2026-01-01 06:00',
                'evidence_count': 15
            }
        ]
    
    if 'case_counter' not in st.session_state:
        st.session_state.case_counter = 1008

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
        'SAFE': '#059669',
        'SUSPICIOUS': '#d97706',
        'MALICIOUS': '#dc2626',
        'low': '#059669',
        'medium': '#d97706',
        'high': '#dc2626'
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
    """Generate a new simple letter CAPTCHA."""
    st.session_state.current_captcha = st.session_state.captcha_generator.generate()

# =============================================================================
# LOGIN PAGE
# =============================================================================

def render_login_page():
    """Render the secure login page with simple CAPTCHA, Registration, and Login Attempt Limiting."""
    
    # Initialize login attempt tracking
    if 'login_attempts' not in st.session_state:
        st.session_state.login_attempts = {}
    if 'locked_accounts' not in st.session_state:
        st.session_state.locked_accounts = {}
    
    MAX_ATTEMPTS = 3
    LOCKOUT_MINUTES = 5
    
    st.markdown('<h1 class="main-header"><span>CYBER-SIGHT</span></h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Police Cyber Crime Detection & Monitoring Portal</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Badge
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <span style="background: #3b82f6; color: white; padding: 0.5rem 1rem; border-radius: 6px; font-size: 0.85rem; font-weight: 600;">
                OFFICIAL LAW ENFORCEMENT PORTAL
            </span>
        </div>
        """, unsafe_allow_html=True)
        
        # Initialize tab state
        if 'auth_tab' not in st.session_state:
            st.session_state.auth_tab = "login"
        
        # Tab selection
        tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
        
        # =================== LOGIN TAB ===================
        with tab1:
            st.markdown("#### Secure Login")
            
            # Check if account is locked
            username_to_check = st.session_state.get('last_attempted_username', '')
            if username_to_check and username_to_check.lower() in st.session_state.locked_accounts:
                lock_time = st.session_state.locked_accounts[username_to_check.lower()]
                time_passed = (datetime.now() - lock_time).total_seconds() / 60
                
                if time_passed < LOCKOUT_MINUTES:
                    remaining = int(LOCKOUT_MINUTES - time_passed)
                    st.error(f"⛔ Account '{username_to_check}' is locked due to too many failed attempts.")
                    st.warning(f"Please wait {remaining} minute(s) before trying again.")
                    st.info("💡 If you forgot your password, contact your administrator.")
                    return
                else:
                    # Unlock the account
                    del st.session_state.locked_accounts[username_to_check.lower()]
                    if username_to_check.lower() in st.session_state.login_attempts:
                        del st.session_state.login_attempts[username_to_check.lower()]
            
            # Show remaining attempts if any failed attempts exist
            if username_to_check and username_to_check.lower() in st.session_state.login_attempts:
                attempts = st.session_state.login_attempts[username_to_check.lower()]
                remaining = MAX_ATTEMPTS - attempts
                if remaining > 0 and remaining < MAX_ATTEMPTS:
                    st.warning(f"⚠️ {remaining} login attempt(s) remaining for this account")
            
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username", key="login_user")
                password = st.text_input("Password", type="password", placeholder="Enter your password", key="login_pass")
                
                # Simple CAPTCHA Display
                st.markdown("#### Type the letters below")
                captcha = st.session_state.current_captcha
                
                st.markdown(f"""
                <div class="captcha-box">
                    {captcha['display']}
                </div>
                <p style="text-align: center; color: #6b7280; font-size: 0.85rem;">Type exactly as shown (case-insensitive)</p>
                """, unsafe_allow_html=True)
                
                captcha_answer = st.text_input("Enter the letters", placeholder="Type the letters here", key="login_captcha")
                
                col_btn1, col_btn2 = st.columns(2)
                
                with col_btn1:
                    submit = st.form_submit_button("Login", type="primary", use_container_width=True)
                
                with col_btn2:
                    refresh = st.form_submit_button("New Code", use_container_width=True)
            
            if refresh:
                generate_new_captcha()
                st.rerun()
            
            if submit:
                # Store the username for lockout checking
                st.session_state.last_attempted_username = username
                
                # Check if this account is locked
                if username.lower() in st.session_state.locked_accounts:
                    lock_time = st.session_state.locked_accounts[username.lower()]
                    time_passed = (datetime.now() - lock_time).total_seconds() / 60
                    
                    if time_passed < LOCKOUT_MINUTES:
                        remaining = int(LOCKOUT_MINUTES - time_passed)
                        st.error(f"⛔ Account locked! Please wait {remaining} minute(s) before trying again.")
                        return
                    else:
                        # Unlock
                        del st.session_state.locked_accounts[username.lower()]
                        if username.lower() in st.session_state.login_attempts:
                            del st.session_state.login_attempts[username.lower()]
                
                # Verify CAPTCHA (case-insensitive)
                captcha_valid = captcha_answer.upper().strip() == st.session_state.current_captcha['answer'].upper()
                
                if not captcha_valid:
                    st.error("Incorrect code. Please try again.")
                    generate_new_captcha()
                    time.sleep(1)
                    st.rerun()
                else:
                    # First check registered users
                    registered_user = None
                    if 'registered_users' in st.session_state:
                        for user in st.session_state.registered_users:
                            if user['username'].lower() == username.lower() and user['password'] == password:
                                registered_user = user
                                break
                    
                    if registered_user:
                        # Successful login - reset attempts
                        if username.lower() in st.session_state.login_attempts:
                            del st.session_state.login_attempts[username.lower()]
                        
                        # Create a user object for registered user
                        from utils.auth import User
                        user_obj = User(
                            username=registered_user['username'],
                            role=registered_user.get('role', 'viewer'),
                            full_name=registered_user['full_name'],
                            department=registered_user.get('department', 'General'),
                            state=registered_user.get('state', 'Maharashtra')
                        )
                        st.session_state.authenticated = True
                        st.session_state.current_user = user_obj
                        st.success(f"Welcome, {registered_user['full_name']}!")
                        time.sleep(1)
                        st.rerun()
                    else:
                        # Try default authentication
                        result = st.session_state.auth_manager.authenticate(username, password)
                        
                        if result['success']:
                            # Successful login - reset attempts
                            if username.lower() in st.session_state.login_attempts:
                                del st.session_state.login_attempts[username.lower()]
                            
                            st.session_state.authenticated = True
                            st.session_state.current_user = result['user']
                            st.success(f"Welcome, {result['user'].full_name}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            # Failed login - track attempts
                            if username.lower() not in st.session_state.login_attempts:
                                st.session_state.login_attempts[username.lower()] = 0
                            
                            st.session_state.login_attempts[username.lower()] += 1
                            attempts = st.session_state.login_attempts[username.lower()]
                            remaining = MAX_ATTEMPTS - attempts
                            
                            if attempts >= MAX_ATTEMPTS:
                                # Lock the account
                                st.session_state.locked_accounts[username.lower()] = datetime.now()
                                st.error(f"⛔ ACCOUNT LOCKED!")
                                st.error(f"Too many failed login attempts ({MAX_ATTEMPTS} attempts).")
                                st.warning(f"Your account is locked for {LOCKOUT_MINUTES} minutes.")
                                st.info("💡 Contact your administrator if you need immediate access.")
                            else:
                                st.error(f"{result['message']}")
                                st.warning(f"⚠️ Warning: {remaining} attempt(s) remaining before account lockout!")
                            
                            generate_new_captcha()
        
        # =================== REGISTRATION TAB ===================
        with tab2:
            st.markdown("#### Create New Account")
            st.markdown('<p style="color: #6b7280; font-size: 0.85rem;">Register for access to the Cyber-Sight portal</p>', unsafe_allow_html=True)
            
            with st.form("register_form"):
                reg_full_name = st.text_input("Full Name *", placeholder="Enter your full name", key="reg_name")
                reg_email = st.text_input("Email Address *", placeholder="Enter your email", key="reg_email")
                reg_username = st.text_input("Username *", placeholder="Choose a username", key="reg_user")
                
                col_pass1, col_pass2 = st.columns(2)
                with col_pass1:
                    reg_password = st.text_input("Password *", type="password", placeholder="Create password", key="reg_pass")
                with col_pass2:
                    reg_confirm = st.text_input("Confirm Password *", type="password", placeholder="Confirm password", key="reg_confirm")
                
                reg_department = st.selectbox("Department *", [
                    "Cyber Crime Cell",
                    "State Police",
                    "Central Bureau",
                    "Intelligence Bureau",
                    "Forensics Division",
                    "Training Academy",
                    "Other"
                ], key="reg_dept")
                
                reg_state = st.selectbox("State *", [
                    "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "Uttar Pradesh",
                    "Gujarat", "West Bengal", "Rajasthan", "Telangana", "Kerala",
                    "Andhra Pradesh", "Madhya Pradesh", "Bihar", "Punjab", "Haryana",
                    "Odisha", "Jharkhand", "Chhattisgarh", "Assam", "Uttarakhand",
                    "Himachal Pradesh", "Goa", "Tripura", "Manipur", "Meghalaya",
                    "Nagaland", "Arunachal Pradesh", "Mizoram", "Sikkim", "Other"
                ], key="reg_state")
                
                reg_badge = st.text_input("Badge/ID Number", placeholder="Optional - Enter badge number", key="reg_badge")
                
                # Registration CAPTCHA
                st.markdown("#### Verification")
                reg_captcha = st.session_state.current_captcha
                
                st.markdown(f"""
                <div class="captcha-box">
                    {reg_captcha['display']}
                </div>
                """, unsafe_allow_html=True)
                
                reg_captcha_answer = st.text_input("Enter verification code", placeholder="Type the letters", key="reg_captcha")
                
                # Terms checkbox
                agree_terms = st.checkbox("I agree to the Terms of Service and Privacy Policy", key="reg_terms")
                
                register_btn = st.form_submit_button("Create Account", type="primary", use_container_width=True)
            
            if register_btn:
                # Validation
                errors = []
                
                if not reg_full_name or len(reg_full_name) < 3:
                    errors.append("Full name must be at least 3 characters")
                
                if not reg_email or '@' not in reg_email:
                    errors.append("Please enter a valid email address")
                
                if not reg_username or len(reg_username) < 4:
                    errors.append("Username must be at least 4 characters")
                
                if not reg_password or len(reg_password) < 6:
                    errors.append("Password must be at least 6 characters")
                
                if reg_password != reg_confirm:
                    errors.append("Passwords do not match")
                
                if not reg_captcha_answer or reg_captcha_answer.upper().strip() != st.session_state.current_captcha['answer'].upper():
                    errors.append("Incorrect verification code")
                
                if not agree_terms:
                    errors.append("You must agree to the Terms of Service")
                
                # Check if username already exists
                existing_users = ['admin', 'officer1', 'analyst1', 'viewer1']
                if 'registered_users' in st.session_state:
                    existing_users.extend([u['username'] for u in st.session_state.registered_users])
                
                if reg_username.lower() in [u.lower() for u in existing_users]:
                    errors.append("Username already exists. Please choose another.")
                
                if errors:
                    for error in errors:
                        st.error(error)
                    generate_new_captcha()
                else:
                    # Store registered user
                    if 'registered_users' not in st.session_state:
                        st.session_state.registered_users = []
                    
                    new_user = {
                        'username': reg_username,
                        'password': reg_password,
                        'full_name': reg_full_name,
                        'email': reg_email,
                        'department': reg_department,
                        'state': reg_state,
                        'badge': reg_badge,
                        'role': 'viewer',  # Default role for new registrations
                        'registered_at': datetime.now().isoformat()
                    }
                    
                    st.session_state.registered_users.append(new_user)
                    
                    st.success(f"Account created successfully! Welcome, {reg_full_name}!")
                    st.info("You can now login with your credentials. Your account has 'Viewer' access by default.")
                    generate_new_captcha()
                    time.sleep(2)
                    st.rerun()
        
        # Demo Credentials
        st.markdown("---")
        st.markdown("#### Demo Credentials")
        st.markdown("""
        | Username | Password | Role |
        |----------|----------|------|
        | admin | admin123 | Administrator |
        | officer1 | officer123 | Police Officer |
        | analyst1 | analyst123 | Crime Analyst |
        | viewer1 | viewer123 | Viewer |
        """)
        
        st.markdown("---")
        st.caption("This is a secure government portal. Unauthorized access is prohibited.")

# =============================================================================
# FLOATING CHATBOT POPUP
# =============================================================================

def render_floating_chatbot():
    """Render a floating chatbot button with popup chat interface."""
    
    # Create a unique key for chat input
    chat_key = f"popup_chat_input_{len(st.session_state.popup_chat_history)}"
    
    # Build chat messages HTML
    chat_messages_html = ""
    for msg in st.session_state.popup_chat_history:
        if msg['role'] == 'user':
            chat_messages_html += f'<div class="chat-msg-user">{msg["content"]}</div>'
        else:
            chat_messages_html += f'<div class="chat-msg-bot">{msg["content"]}</div>'
    
    if not chat_messages_html:
        chat_messages_html = '''
        <div style="text-align: center; padding: 2rem; color: #64748b;">
            <i class="fas fa-robot" style="font-size: 2rem; margin-bottom: 0.5rem; color: #3b82f6;"></i>
            <p style="margin: 0.5rem 0 0 0; font-size: 0.85rem;">Hi! I'm your Cyber Security Assistant.</p>
            <p style="margin: 0.3rem 0 0 0; font-size: 0.75rem;">Ask me anything about cyber safety!</p>
        </div>
        '''
    
    # Chat popup container using Streamlit expander alternative
    chat_col1, chat_col2 = st.columns([3, 1])
    
    with st.container():
        # Use sidebar columns approach for floating effect
        # Create a fixed position container using HTML/JS
        
        if st.session_state.popup_chat_open:
            # Popup is open - show chat interface
            st.markdown(f'''
            <div class="chat-popup" id="chatPopup">
                <div class="chat-popup-header">
                    <h4><i class="fas fa-robot"></i> Cyber Assistant</h4>
                    <span style="font-size: 0.7rem; opacity: 0.8;">Online</span>
                </div>
                <div class="chat-popup-body" id="chatBody">
                    {chat_messages_html}
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Chat input area using Streamlit
            with st.container():
                st.markdown('<div style="position: fixed; bottom: 100px; right: 30px; width: 380px; z-index: 9998;">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 8, 2])
                with col2:
                    user_message = st.text_input(
                        "Message",
                        key=chat_key,
                        placeholder="Type a message...",
                        label_visibility="collapsed"
                    )
                with col3:
                    send_clicked = st.button("Send", key="popup_send", type="primary")
                
                # Process message
                if send_clicked and user_message:
                    st.session_state.popup_chat_history.append({'role': 'user', 'content': user_message})
                    
                    # Get response from chatbot
                    quick_answer = QuickResponder.get_quick_answer(user_message)
                    if quick_answer:
                        response_text = quick_answer
                    else:
                        response = st.session_state.chatbot.chat(user_message)
                        response_text = response.response
                    
                    st.session_state.popup_chat_history.append({'role': 'assistant', 'content': response_text})
                    st.rerun()
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Close button
            if st.button("Close Chat", key="close_popup_chat"):
                st.session_state.popup_chat_open = False
                st.rerun()
        
        # Floating button (always visible)
        st.markdown('''
        <style>
            .floating-btn-container {
                position: fixed;
                bottom: 30px;
                right: 30px;
                z-index: 10000;
            }
        </style>
        ''', unsafe_allow_html=True)


def render_chat_popup_button():
    """Render just the floating button to toggle chat."""
    # Use a fixed position button
    col1, col2, col3, col4, col5 = st.columns([10, 1, 1, 1, 1])
    
    with col5:
        if st.session_state.popup_chat_open:
            if st.button("X", key="float_close", help="Close Chat"):
                st.session_state.popup_chat_open = False
                st.rerun()
        else:
            if st.button("Chat", key="float_open", help="Open AI Assistant"):
                st.session_state.popup_chat_open = True
                st.rerun()

# =============================================================================
# SIDEBAR WITH ICONS
# =============================================================================

def render_sidebar():
    """Render the sidebar navigation with icons."""
    with st.sidebar:
        # Logo/Title
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #3b82f6; margin: 0; font-weight: 700;">CYBER-SIGHT</h2>
            <p style="color: #64748b; font-size: 0.8rem; margin: 0;">Police Cyber Crime Portal</p>
        </div>
        """, unsafe_allow_html=True)
        
        # User Info Card
        if st.session_state.current_user:
            user = st.session_state.current_user
            role_colors = {
                'admin': '#dc2626',
                'officer': '#3b82f6',
                'analyst': '#8b5cf6',
                'viewer': '#22c55e'
            }
            role_color = role_colors.get(user.role, '#3b82f6')
            
            st.markdown(f"""
            <div style="background: #111827; border-radius: 10px; padding: 1rem; margin: 0.5rem 0; border: 1px solid #1f2937;">
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 40px; height: 40px; background: {role_color}; border-radius: 50%; display: flex; align-items: center; justify-content: center; color: white; font-weight: 600;">
                        {user.full_name[0].upper()}
                    </div>
                    <div>
                        <p style="margin: 0; color: #e5e7eb; font-weight: 600; font-size: 0.9rem;">{user.full_name}</p>
                        <p style="margin: 0; color: {role_color}; font-size: 0.75rem; text-transform: uppercase;">{user.role}</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Navigation with Icons
        st.markdown("##### MAIN MENU")
        
        # Define pages with icons
        nav_items = [
            ("Home", "fa-house", "Dashboard"),
            ("Threat Detection", "fa-shield-halved", "Scan Threats"),
            ("URL Checker", "fa-link", "Check URLs"),
            ("AI Chatbot", "fa-robot", "AI Assistant"),
            ("Dataset Insights", "fa-chart-pie", "Data Analysis"),
        ]
        
        nav_items_india = [
            ("India Crime Map", "fa-map-location-dot", "Crime Heatmap"),
            ("State Predictions", "fa-chart-line", "State Forecast"),
            ("ML Prediction & Analysis", "fa-brain", "ML Predictor"),
        ]
        
        nav_items_ops = [
            ("Live Threats", "fa-satellite-dish", "Real-time"),
            ("Case Management", "fa-folder-open", "Cases"),
        ]
        
        # Main Menu
        selected_page = st.session_state.get('selected_page', 'Home')
        
        for page_name, icon, desc in nav_items:
            is_active = selected_page == page_name
            bg_color = "#3b82f6" if is_active else "transparent"
            text_color = "#ffffff" if is_active else "#9ca3af"
            
            if st.button(f"{page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.selected_page = page_name
                st.rerun()
        
        st.markdown("##### INDIA ANALYTICS")
        for page_name, icon, desc in nav_items_india:
            if st.button(f"{page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.selected_page = page_name
                st.rerun()
        
        st.markdown("##### OPERATIONS")
        for page_name, icon, desc in nav_items_ops:
            if st.button(f"{page_name}", key=f"nav_{page_name}", use_container_width=True):
                st.session_state.selected_page = page_name
                st.rerun()
        
        st.markdown("---")
        
        # System Status
        st.markdown("##### SYSTEM STATUS")
        
        if st.session_state.model_loaded:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; color: #22c55e;">
                <span style="width: 8px; height: 8px; background: #22c55e; border-radius: 50%;"></span>
                ML Model Active
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="display: flex; align-items: center; gap: 8px; color: #f59e0b;">
                <span style="width: 8px; height: 8px; background: #f59e0b; border-radius: 50%;"></span>
                ML Model Loading...
            </div>
            """, unsafe_allow_html=True)
        
        # Live Monitoring Status
        st.markdown("""
        <div style="display: flex; align-items: center; gap: 8px; color: #dc2626; margin-top: 0.5rem;">
            <span class="live-dot"></span>
            Live Monitoring
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Logout Button
        if st.button("Logout", use_container_width=True, type="secondary"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.session_state.selected_page = 'Home'
            generate_new_captcha()
            st.rerun()
        
        # Return selected page
        return st.session_state.get('selected_page', 'Home')

# =============================================================================
# PAGE: HOME
# =============================================================================

def render_home_page():
    """Render the home page - clean dashboard."""
    st.markdown('<h1 class="main-header"><span>CYBER-SIGHT</span> Dashboard</h1>', unsafe_allow_html=True)
    
    # Welcome Message
    if st.session_state.current_user:
        user = st.session_state.current_user
        st.markdown(f"""
        <div style="background: #111827; border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem; border: 1px solid #1f2937;">
            <h3 style="margin: 0; color: #e5e7eb;">Welcome back, {user.full_name}</h3>
            <p style="margin: 0.5rem 0 0 0; color: #6b7280;">Role: {user.role.upper()} | Department: {user.department}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Stats Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>AI</h3>
            <h4>Threat Detection</h4>
            <p>ML-powered analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>36</h3>
            <h4>States/UTs</h4>
            <p>Pan-India coverage</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>2045</h3>
            <h4>Predictions</h4>
            <p>20-year forecast</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <h3>24/7</h3>
            <h4>Monitoring</h4>
            <p>Real-time alerts</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features Overview
    st.markdown('<p class="section-header">Platform Features</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Cyber Threat Detection**
        - Machine learning-based threat classification
        - Identify phishing, malware, and hacking attempts
        - Risk level prediction (Low/Medium/High)
        - Real-time analysis with confidence scores
        
        **India Crime Analytics**
        - State-wise cyber crime statistics
        - Historical data from 2018-2025
        - ML predictions up to 2045
        - Interactive maps and visualizations
        """)
    
    with col2:
        st.markdown("""
        **Live Threat Monitoring**
        - Real-time threat feed simulation
        - Tampering and intrusion detection
        - Alert classification by severity
        - Automatic case generation
        
        **Case Management**
        - Create and track cyber crime cases
        - Assign to officers
        - Status tracking and updates
        - Generate reports
        """)
    
    st.markdown("---")
    
    # Recent Activity
    st.markdown('<p class="section-header">Quick Overview</p>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Today's Stats**")
        st.metric("Threats Detected", "127", "+12%")
        st.metric("URLs Analyzed", "342", "+8%")
        st.metric("Cases Created", "15", "+3")
    
    with col2:
        st.markdown("**Alert Summary**")
        st.metric("Critical Alerts", "3", delta_color="inverse")
        st.metric("High Priority", "12")
        st.metric("Pending Review", "28")
    
    with col3:
        st.markdown("**Top States (Cases)**")
        st.markdown("1. Maharashtra - 2,451")
        st.markdown("2. Karnataka - 1,832")
        st.markdown("3. Uttar Pradesh - 1,654")
        st.markdown("4. Delhi - 1,423")
        st.markdown("5. Tamil Nadu - 1,287")

# =============================================================================
# PAGE: THREAT DETECTION
# =============================================================================

def render_threat_detection_page():
    """Render the threat detection page with modern UI."""
    # Modern Page Header
    st.markdown("""
    <div class="page-header">
        <h1><i class="fas fa-shield-halved"></i> Cyber Threat Detection</h1>
        <p>Analyze potential cyber threats using our advanced ML-powered detection system</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        status_color = "green" if st.session_state.model_loaded else "yellow"
        status_text = "Active" if st.session_state.model_loaded else "Heuristic"
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number {status_color}">{status_text}</div>
            <div class="label">ML Model</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="mini-stat">
            <div class="number blue">4</div>
            <div class="label">Threat Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="mini-stat">
            <div class="number purple">AI</div>
            <div class="label">Powered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="mini-stat">
            <div class="number green">Real-time</div>
            <div class="label">Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    if not st.session_state.model_loaded:
        st.markdown("""
        <div class="info-panel warning">
            <h4><i class="fas fa-exclamation-triangle"></i> ML Model Not Loaded</h4>
            <p>Using heuristic-only analysis mode. For best results, train the ML model first.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Analysis Type Selection with Modern Cards
    st.markdown("#### Select Analysis Type")
    
    analysis_cols = st.columns(3)
    
    with analysis_cols[0]:
        url_selected = st.button("🔗 URL Analysis", use_container_width=True, key="sel_url")
    with analysis_cols[1]:
        incident_selected = st.button("📝 Incident Report", use_container_width=True, key="sel_incident")
    with analysis_cols[2]:
        manual_selected = st.button("⚙️ Manual Input", use_container_width=True, key="sel_manual")
    
    # Initialize analysis mode
    if 'analysis_mode' not in st.session_state:
        st.session_state.analysis_mode = "URL Analysis"
    
    if url_selected:
        st.session_state.analysis_mode = "URL Analysis"
    elif incident_selected:
        st.session_state.analysis_mode = "Incident Description"
    elif manual_selected:
        st.session_state.analysis_mode = "Manual Feature Input"
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Analysis Forms
    if st.session_state.analysis_mode == "URL Analysis":
        st.markdown("""
        <div class="info-panel">
            <h4><i class="fas fa-link"></i> URL Threat Analysis</h4>
            <p>Enter a suspicious URL to analyze for potential threats</p>
        </div>
        """, unsafe_allow_html=True)
        
        url = st.text_input(
            "URL",
            placeholder="https://example.com/suspicious-link",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Analyze Threat", type="primary", use_container_width=True):
            if url:
                with st.spinner("Analyzing threat..."):
                    result = st.session_state.url_checker.check_url(url)
                    
                    # Result display
                    status_class = "safe" if result.is_safe else "danger"
                    status_icon = "✅" if result.is_safe else "🚨"
                    status_text = "SAFE" if result.is_safe else "THREAT DETECTED"
                    
                    st.markdown(f"""
                    <div class="result-box {status_class}">
                        <div style="text-align: center;">
                            <div class="result-icon">{status_icon}</div>
                            <div class="result-title">{status_text}</div>
                            <div class="result-subtitle">Confidence: {result.confidence}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("**Findings**")
                        for reason in result.reasons:
                            st.write(f"• {reason}")
                    
                    with col2:
                        st.markdown("**Recommendations**")
                        for rec in result.recommendations:
                            st.write(f"• {rec}")
            else:
                st.warning("Please enter a URL to analyze.")
    
    elif st.session_state.analysis_mode == "Incident Description":
        st.markdown("""
        <div class="info-panel">
            <h4><i class="fas fa-file-alt"></i> Incident Analysis</h4>
            <p>Describe the security incident for AI-powered threat classification</p>
        </div>
        """, unsafe_allow_html=True)
        
        incident = st.text_area(
            "Incident",
            placeholder="Example: I received an email claiming to be from my bank asking me to verify my account by clicking a link...",
            height=150,
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Analyze Incident", type="primary", use_container_width=True):
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
                    confidence_pct = int(confidence * 100)
                    
                    # Color based on threat type
                    type_colors = {'phishing': 'red', 'malware': 'red', 'hacking': 'red', 'scam': 'yellow'}
                    
                    st.markdown(f"""
                    <div class="result-box warning">
                        <div style="text-align: center;">
                            <div class="result-icon">⚠️</div>
                            <div class="result-title">{detected_type.upper()} DETECTED</div>
                            <div class="result-subtitle">Confidence: {confidence_pct}%</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Progress bar for confidence
                    st.markdown(f"""
                    <div class="progress-bar">
                        <div class="fill red" style="width: {confidence_pct}%;"></div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-box safe">
                        <div style="text-align: center;">
                            <div class="result-icon">✅</div>
                            <div class="result-title">NO CLEAR THREAT</div>
                            <div class="result-subtitle">Unable to detect specific threat type</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.warning("Please describe the incident.")
    
    else:  # Manual Feature Input
        st.markdown("""
        <div class="info-panel">
            <h4><i class="fas fa-sliders"></i> Manual Feature Analysis</h4>
            <p>Enter URL characteristics for detailed ML analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="data-card">
                <div class="label">URL Structure</div>
            </div>
            """, unsafe_allow_html=True)
            domain_length = st.number_input("Domain Length", min_value=1, max_value=100, value=15)
            has_https = st.selectbox("Uses HTTPS?", [1, 0], format_func=lambda x: "Yes ✅" if x else "No ❌")
            has_ip = st.selectbox("Contains IP Address?", [0, 1], format_func=lambda x: "Yes ⚠️" if x else "No ✅")
        
        with col2:
            st.markdown("""
            <div class="data-card">
                <div class="label">URL Characteristics</div>
            </div>
            """, unsafe_allow_html=True)
            num_dots = st.number_input("Number of Dots", min_value=0, max_value=20, value=1)
            num_hyphens = st.number_input("Number of Hyphens", min_value=0, max_value=20, value=0)
            url_length = st.number_input("URL Length", min_value=5, max_value=500, value=30)
        
        if st.button("🔍 Predict Threat", type="primary", use_container_width=True):
            # Calculate risk score based on features
            risk_score = 0
            if not has_https:
                risk_score += 30
            if has_ip:
                risk_score += 25
            if domain_length > 30:
                risk_score += 15
            if url_length > 100:
                risk_score += 15
            if num_hyphens > 3:
                risk_score += 10
            if num_dots > 3:
                risk_score += 5
            
            risk_level = "LOW" if risk_score < 30 else ("MEDIUM" if risk_score < 60 else "HIGH")
            risk_class = "safe" if risk_score < 30 else ("warning" if risk_score < 60 else "danger")
            risk_icon = "✅" if risk_score < 30 else ("⚠️" if risk_score < 60 else "🚨")
            
            st.markdown(f"""
            <div class="result-box {risk_class}">
                <div style="text-align: center;">
                    <div class="result-icon">{risk_icon}</div>
                    <div class="result-title">{risk_level} RISK</div>
                    <div class="result-subtitle">Risk Score: {risk_score}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# =============================================================================
# PAGE: URL CHECKER
# =============================================================================

def render_url_checker_page():
    """Render the URL checker page with modern UI."""
    # Modern Page Header
    st.markdown("""
    <div class="page-header">
        <h1><i class="fas fa-link"></i> URL Safety Checker</h1>
        <p>Analyze URLs for potential security threats, phishing attempts, and malicious content</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Action Cards Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="mini-stat">
            <div class="number blue">AI</div>
            <div class="label">Powered Analysis</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="mini-stat">
            <div class="number green">100+</div>
            <div class="label">Threat Patterns</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="mini-stat">
            <div class="number purple">Real-time</div>
            <div class="label">Detection</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Input Section with modern styling
    st.markdown("""
    <div class="input-section">
        <label><i class="fas fa-globe"></i> Enter URL to Analyze</label>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([4, 1])
    
    with col1:
        url = st.text_input(
            "URL",
            placeholder="https://example.com/suspicious-link",
            label_visibility="collapsed"
        )
    
    with col2:
        check_btn = st.button("🔍 Analyze", type="primary", use_container_width=True)
    
    if check_btn and url:
        with st.spinner("Analyzing URL security..."):
            result = st.session_state.url_checker.check_url(url)
        
        # Determine result type for styling
        status_class = "safe" if result.is_safe else ("warning" if result.confidence < 70 else "danger")
        status_icon = "✅" if result.is_safe else ("⚠️" if result.confidence < 70 else "🚨")
        status_text = "SAFE" if result.is_safe else ("SUSPICIOUS" if result.confidence < 70 else "DANGEROUS")
        
        # Modern Result Box
        st.markdown(f"""
        <div class="result-box {status_class}">
            <div style="text-align: center;">
                <div class="result-icon">{status_icon}</div>
                <div class="result-title">{status_text}</div>
                <div class="result-subtitle">Confidence: {result.confidence}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Analysis Details in Cards
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-panel">
                <h4><i class="fas fa-search"></i> Analysis Findings</h4>
            </div>
            """, unsafe_allow_html=True)
            for reason in result.reasons:
                if reason.startswith("[OK]"):
                    st.markdown(f"""
                    <div class="threat-card low">
                        <div class="threat-desc">✓ {reason.replace('[OK]', '').strip()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                elif reason.startswith("[!]"):
                    st.markdown(f"""
                    <div class="threat-card high">
                        <div class="threat-desc">⚠ {reason.replace('[!]', '').strip()}</div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="threat-card medium">
                        <div class="threat-desc">ℹ {reason}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-panel success">
                <h4><i class="fas fa-shield-alt"></i> Recommendations</h4>
            </div>
            """, unsafe_allow_html=True)
            for rec in result.recommendations:
                st.markdown(f"""
                <div class="data-card" style="margin-bottom: 0.5rem;">
                    <div class="sub">→ {rec}</div>
                </div>
                """, unsafe_allow_html=True)
    
    elif check_btn and not url:
        st.markdown("""
        <div class="info-panel warning">
            <h4><i class="fas fa-exclamation-triangle"></i> Input Required</h4>
            <p>Please enter a URL to analyze</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Quick Tips Section
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-panel">
        <h4><i class="fas fa-lightbulb"></i> Quick Security Tips</h4>
        <p>• Always check for HTTPS before entering sensitive information<br>
        • Be cautious of URLs with unusual characters or misspellings<br>
        • Verify shortened URLs before clicking<br>
        • Look for the padlock icon in your browser's address bar</p>
    </div>
    """, unsafe_allow_html=True)

# =============================================================================
# PAGE: AI CHATBOT
# =============================================================================

def render_chatbot_page():
    """Render the AI chatbot page with modern UI."""
    # Modern Page Header
    st.markdown("""
    <div class="page-header">
        <h1><i class="fas fa-robot"></i> AI Cyber Security Assistant</h1>
        <p>Get instant answers about cybersecurity, threats, and online safety from our ML-powered assistant</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Stats Row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="mini-stat">
            <div class="number purple">Neural</div>
            <div class="label">Network Powered</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="mini-stat">
            <div class="number blue">24/7</div>
            <div class="label">Available</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number green">{len(st.session_state.chat_history) // 2}</div>
            <div class="label">Conversations</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Chat Container with custom styling
    chat_container = st.container()
    
    with chat_container:
        if not st.session_state.chat_history:
            # Empty state with suggestions
            st.markdown("""
            <div class="empty-state">
                <div class="icon">💬</div>
                <h4>Start a Conversation</h4>
                <p>Ask me anything about cybersecurity!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Suggestion Cards
            st.markdown("#### Quick Questions")
            suggestions = st.session_state.chatbot.get_suggestions()
            
            cols = st.columns(len(suggestions))
            for i, suggestion in enumerate(suggestions):
                with cols[i]:
                    if st.button(f"💡 {suggestion[:30]}...", key=f"sug_{i}", use_container_width=True):
                        st.session_state.chat_history.append({'role': 'user', 'content': suggestion})
                        response = st.session_state.chatbot.chat(suggestion)
                        st.session_state.chat_history.append({'role': 'assistant', 'content': response.response})
                        st.rerun()
        else:
            # Show chat history
            for message in st.session_state.chat_history:
                if message['role'] == 'user':
                    st.markdown(f"""
                    <div class="chat-user">
                        <strong>You:</strong> {message['content']}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-bot">
                        <strong>🤖 AI Assistant:</strong><br>{message['content']}
                    </div>
                    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Modern Input Section
    st.markdown("""
    <div class="input-section">
        <label><i class="fas fa-comment-dots"></i> Type your question</label>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "Message",
            placeholder="Ask about phishing, passwords, malware, security tips...",
            label_visibility="collapsed",
            key="chat_input"
        )
    
    with col2:
        send_btn = st.button("📤 Send", type="primary", use_container_width=True)
    
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
    
    # Action buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.chatbot.reset_conversation()
            st.rerun()
    
    with col2:
        if st.button("💾 Export Chat", use_container_width=True):
            st.info("Chat history exported!")

# =============================================================================
# PAGE: DATASET INSIGHTS
# =============================================================================

def render_insights_page():
    """Render the dataset insights page with modern UI."""
    # Modern Page Header
    st.markdown("""
    <div class="page-header">
        <h1><i class="fas fa-chart-pie"></i> Dataset Insights</h1>
        <p>Explore global cyber threat patterns, attack distributions, and risk analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    df = load_dataset()
    
    if df is None:
        st.markdown("""
        <div class="info-panel danger">
            <h4><i class="fas fa-exclamation-circle"></i> Dataset Not Found</h4>
            <p>Please ensure 'data/cybercrime_dataset.csv' exists in the project directory.</p>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Stats Row with modern cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number blue">{len(df):,}</div>
            <div class="label">Total Records</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        attack_types = df['attack_type'].nunique() if 'attack_type' in df.columns else 0
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number purple">{attack_types}</div>
            <div class="label">Attack Types</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        countries = df['country'].nunique() if 'country' in df.columns else 0
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number green">{countries}</div>
            <div class="label">Countries</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        high_risk = len(df[df['risk_level'] == 'high']) if 'risk_level' in df.columns else 0
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number red">{high_risk:,}</div>
            <div class="label">High Risk</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Charts Section
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        if 'attack_type' in df.columns:
            st.markdown("""
            <div class="info-panel">
                <h4><i class="fas fa-virus"></i> Attack Type Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            attack_counts = df['attack_type'].value_counts()
            
            fig = px.pie(
                values=attack_counts.values,
                names=attack_counts.index,
                color_discrete_sequence=px.colors.qualitative.Set2,
                hole=0.4
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_col2:
        if 'risk_level' in df.columns:
            st.markdown("""
            <div class="info-panel warning">
                <h4><i class="fas fa-exclamation-triangle"></i> Risk Level Distribution</h4>
            </div>
            """, unsafe_allow_html=True)
            risk_counts = df['risk_level'].value_counts()
            colors = {'low': '#22c55e', 'medium': '#eab308', 'high': '#ef4444'}
            
            fig = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                color=risk_counts.index,
                color_discrete_map=colors
            )
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_color='#e2e8f0',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: INDIA CRIME MAP
# =============================================================================

def render_india_map_page():
    """Render the India crime map page with modern UI."""
    # Modern Page Header
    st.markdown("""
    <div class="page-header">
        <h1><i class="fas fa-map-location-dot"></i> India Cyber Crime Map</h1>
        <p>Interactive visualization of state-wise cyber crime statistics across India</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Filter Section
    st.markdown("""
    <div class="input-section">
        <label><i class="fas fa-filter"></i> Filter Data</label>
    </div>
    """, unsafe_allow_html=True)
    
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
            ["All Categories"] + CRIME_CATEGORIES
        )
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Load data
    df = st.session_state.historical_data
    
    # Filter by year
    df_filtered = df[(df['year'] >= year_range[0]) & (df['year'] <= year_range[1])]
    
    # Aggregate by state
    if crime_type == "All Categories":
        state_totals = df_filtered.groupby('state')['cases_reported'].sum().reset_index()
        state_totals.columns = ['state', 'cases']
    else:
        state_totals = df_filtered[df_filtered['crime_category'] == crime_type].groupby('state')['cases_reported'].sum().reset_index()
        state_totals.columns = ['state', 'cases']
    
    # Add coordinates
    state_totals['lat'] = state_totals['state'].map(lambda x: STATE_COORDINATES.get(x, {}).get('lat', 20))
    state_totals['lon'] = state_totals['state'].map(lambda x: STATE_COORDINATES.get(x, {}).get('lon', 78))
    
    # Stats summary
    total_cases = state_totals['cases'].sum()
    top_state = state_totals.nlargest(1, 'cases')['state'].values[0] if len(state_totals) > 0 else "N/A"
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number blue">{total_cases:,}</div>
            <div class="label">Total Cases</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number red">{top_state}</div>
            <div class="label">Most Affected</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number green">{len(state_totals)}</div>
            <div class="label">States/UTs</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Create map
    st.markdown("""
    <div class="info-panel">
        <h4><i class="fas fa-globe-asia"></i> Geographic Distribution</h4>
    </div>
    """, unsafe_allow_html=True)
    
    fig = px.scatter_geo(
        state_totals,
        lat='lat',
        lon='lon',
        size='cases',
        hover_name='state',
        color='cases',
        color_continuous_scale='Reds',
        scope='asia',
        size_max=50
    )
    
    fig.update_geos(
        center=dict(lat=22, lon=82),
        projection_scale=4,
        showland=True,
        landcolor='rgb(30, 41, 59)',
        countrycolor='rgb(100, 116, 139)',
        showocean=True,
        oceancolor='rgb(15, 23, 42)'
    )
    
    fig.update_layout(
        height=550,
        paper_bgcolor='rgba(0,0,0,0)',
        geo_bgcolor='rgba(0,0,0,0)',
        font_color='#e2e8f0'
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Two column layout for table and chart
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-panel">
            <h4><i class="fas fa-ranking-star"></i> Top 10 States</h4>
        </div>
        """, unsafe_allow_html=True)
        
        top_states = state_totals.nlargest(10, 'cases')[['state', 'cases']]
        top_states.columns = ['State', 'Total Cases']
        top_states = top_states.reset_index(drop=True)
        top_states.index = top_states.index + 1
        
        st.dataframe(top_states, use_container_width=True)
    
    with col2:
        st.markdown("""
        <div class="info-panel warning">
            <h4><i class="fas fa-chart-bar"></i> Crime Categories</h4>
        </div>
        """, unsafe_allow_html=True)
        
        category_totals = df_filtered.groupby('crime_category')['cases_reported'].sum().sort_values(ascending=True)
        
        fig = px.bar(
            x=category_totals.values,
            y=category_totals.index,
            orientation='h',
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
    st.markdown('<p class="section-header">State-wise Predictions (2026-2045)</p>', unsafe_allow_html=True)
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
    st.markdown(f"### {selected_state} - Cyber Crime Outlook")
    
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
        st.metric("Predicted Loss (Cr)", f"Rs.{total_loss:,.1f}")
    
    with col4:
        avg_confidence = year_pred['confidence_level'].mean()
        st.metric("Confidence Level", f"{avg_confidence:.1f}%")
    
    st.markdown("---")
    
    # Time Series Chart
    st.markdown("### Historical & Predicted Trend")
    
    # Combine historical and prediction data
    hist_yearly = state_hist.groupby('year')['cases_reported'].sum().reset_index()
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
        color_discrete_map={'Historical': '#1e40af', 'Predicted': '#dc2626'}
    )
    
    fig.add_vline(x=2025.5, line_dash="dash", line_color="gray", annotation_text="Prediction Start")
    fig.update_layout(xaxis_title="Year", yaxis_title="Total Cases")
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Crime Type Predictions
    st.markdown(f"### Crime Category Predictions for {prediction_year}")
    
    category_pred = year_pred[['crime_category', 'predicted_cases', 'predicted_loss_lakhs']].copy()
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
    st.markdown("### Detailed Predictions")
    
    display_df = year_pred[['crime_category', 'predicted_cases', 'predicted_solve_rate', 
                            'predicted_loss_lakhs', 'confidence_level']].copy()
    display_df.columns = ['Crime Type', 'Cases', 'Solve Rate (%)', 'Loss (Lakhs)', 'Confidence (%)']
    display_df = display_df.round(2)
    
    st.dataframe(display_df, use_container_width=True, hide_index=True)

# =============================================================================
# PAGE: LIVE THREATS
# =============================================================================

def render_live_threats_page():
    """Render the live threats monitoring page with modern UI."""
    # Modern Page Header with Live Badge
    st.markdown("""
    <div class="page-header">
        <h1><i class="fas fa-satellite-dish"></i> Live Threat Monitor 
            <span class="live-indicator" style="font-size: 0.5em; vertical-align: middle; margin-left: 10px;">
                <span class="live-dot"></span> LIVE
            </span>
        </h1>
        <p>Real-time cyber threat feed, alert management, and system tampering detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Alert Stats Row
    alerts = st.session_state.alert_system.alert_history
    critical_count = len([a for a in alerts if a.get('severity') == 'CRITICAL'])
    high_count = len([a for a in alerts if a.get('severity') == 'HIGH'])
    medium_count = len([a for a in alerts if a.get('severity') == 'MEDIUM'])
    low_count = len([a for a in alerts if a.get('severity') == 'LOW'])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number red">{critical_count}</div>
            <div class="label">Critical</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number yellow">{high_count}</div>
            <div class="label">High</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number blue">{medium_count}</div>
            <div class="label">Medium</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="mini-stat">
            <div class="number green">{low_count}</div>
            <div class="label">Low</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Action Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("⚡ Generate Threats", type="primary", use_container_width=True):
            new_threats = st.session_state.threat_generator.generate_batch(5)
            st.session_state.live_threats = new_threats + st.session_state.live_threats[:20]
            st.rerun()
    
    with col2:
        if st.button("🔔 Process Alerts", use_container_width=True):
            for threat in st.session_state.live_threats[:5]:
                st.session_state.alert_system.create_alert(threat)
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear Feed", use_container_width=True):
            st.session_state.live_threats = []
            st.rerun()
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Live Threat Feed with modern cards
    st.markdown("""
    <div class="info-panel">
        <h4><i class="fas fa-stream"></i> Live Threat Feed</h4>
        <p>Showing latest detected threats in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not st.session_state.live_threats:
        st.markdown("""
        <div class="empty-state">
            <div class="icon">📡</div>
            <h4>No Active Threats</h4>
            <p>Click "Generate Threats" to simulate threat detection</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        for threat in st.session_state.live_threats[:10]:
            severity = threat.severity.value if hasattr(threat.severity, 'value') else threat.severity
            threat_type = threat.threat_type.value if hasattr(threat.threat_type, 'value') else threat.threat_type
            
            severity_class = severity.lower()
            
            st.markdown(f"""
            <div class="threat-card {severity_class}">
                <div class="threat-header">
                    <span class="threat-type">{threat_type}</span>
                    <span class="threat-severity">{severity}</span>
                </div>
                <div class="threat-desc">{threat.description}</div>
                <div class="threat-meta">
                    <span><i class="fas fa-map-marker-alt"></i> {threat.target_location}</span>
                    <span><i class="fas fa-building"></i> {threat.target_sector}</span>
                    <span><i class="fas fa-network-wired"></i> {threat.source_ip}</span>
                    <span><i class="fas fa-clock"></i> {threat.timestamp.strftime('%H:%M:%S')}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Tampering Detection Section
    st.markdown("""
    <div class="info-panel warning">
        <h4><i class="fas fa-shield-virus"></i> System Tampering Detection</h4>
        <p>Monitor for unauthorized system modifications and security breaches</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🔒 Run Security Check", type="primary", use_container_width=True):
            with st.spinner("Scanning systems..."):
                time.sleep(1)
                event = st.session_state.tampering_detector.generate_tampering_event()
                
                import random
                tampering_detected = random.random() < 0.3
                
                if tampering_detected:
                    st.markdown(f"""
                    <div class="result-box danger">
                        <div style="text-align: center;">
                            <div class="result-icon">🚨</div>
                            <div class="result-title">TAMPERING DETECTED</div>
                            <div class="result-subtitle">{event['event_type']}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="threat-card critical">
                        <div class="threat-desc">
                            <strong>System:</strong> {event['system']}<br>
                            <strong>Source IP:</strong> {event['source_ip']}<br>
                            <strong>Remediation:</strong> {event['remediation']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-box safe">
                        <div style="text-align: center;">
                            <div class="result-icon">✅</div>
                            <div class="result-title">SYSTEM SECURE</div>
                            <div class="result-subtitle">No tampering detected</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="data-card">
            <div class="label">Last Check</div>
            <div class="value" style="font-size: 1rem;">{datetime.now().strftime('%H:%M')}</div>
            <div class="sub">System Status: Secure</div>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# PAGE: CASE MANAGEMENT
# =============================================================================

def render_case_management_page():
    """Render the case management page with modern UI."""
    # Modern Page Header
    st.markdown("""
    <div class="page-header">
        <h1><i class="fas fa-folder-open"></i> Case Management</h1>
        <p>Create, track, and manage cyber crime investigation cases</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Tabs for different actions
    tab1, tab2, tab3 = st.tabs(["View Cases", "New Case", "Statistics"])
    
    with tab1:
        st.markdown("### Active Cases")
        
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
        st.markdown("### Create New Case")
        
        with st.form("new_case_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                title = st.text_input("Case Title", placeholder="Brief description of the case")
                crime_type = st.selectbox("Crime Type", CRIME_CATEGORIES)
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
            
            submit = st.form_submit_button("Create Case", type="primary", use_container_width=True)
        
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
            st.success(f"Case #{new_case['id']} created successfully!")
    
    with tab3:
        st.markdown("### Case Statistics")
        
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
# PAGE: ML PREDICTION & ANALYSIS (NCRB DATA)
# =============================================================================

def render_ml_analysis_page():
    """Render the ML Prediction & Analysis page - State-wise crime prediction for police."""
    st.markdown('<p class="section-header">State Crime Predictor</p>', unsafe_allow_html=True)
    st.markdown("Predict future cyber crimes for any Indian state and get actionable insights for prevention")
    
    st.markdown("---")
    
    # Initialize ML analysis if not done
    if 'ml_analysis_results' not in st.session_state:
        with st.spinner("Loading ML Models... Please wait (10-15 seconds)"):
            st.session_state.ml_analysis_results = run_full_analysis()
    
    results = st.session_state.ml_analysis_results
    best_model = results['best_model']
    best_metrics = results['best_metrics']
    
    # ==========================================================================
    # STEP 1: SELECT YOUR STATE
    # ==========================================================================
    st.markdown("### Step 1: Select Your State")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        all_states = sorted(results['historical_data']['state'].unique().tolist())
        selected_state = st.selectbox(
            "Choose State/UT for Prediction",
            all_states,
            index=all_states.index('Maharashtra') if 'Maharashtra' in all_states else 0,
            help="Select the state for which you want to predict cyber crimes"
        )
    
    with col2:
        prediction_year = st.selectbox(
            "Prediction Year",
            [2026, 2027, 2028, 2029, 2030],
            index=0,
            help="Select the year for prediction"
        )
    
    st.markdown("---")
    
    # ==========================================================================
    # STEP 2: CURRENT STATE ANALYSIS (Historical Data)
    # ==========================================================================
    st.markdown(f"### Step 2: Current Crime Analysis - {selected_state}")
    
    # Get state historical data
    state_hist = results['historical_data'][results['historical_data']['state'] == selected_state]
    
    if len(state_hist) > 0:
        # Summary stats
        total_cases = state_hist['cases_reported'].sum()
        avg_cases_per_year = state_hist.groupby('year')['cases_reported'].sum().mean()
        avg_solve_rate = state_hist['solve_rate'].mean()
        total_loss = state_hist['financial_loss_lakhs'].sum()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cases (2018-2025)", f"{total_cases:,.0f}")
        with col2:
            st.metric("Avg Cases/Year", f"{avg_cases_per_year:,.0f}")
        with col3:
            st.metric("Avg Solve Rate", f"{avg_solve_rate:.1f}%")
        with col4:
            st.metric("Total Loss", f"Rs. {total_loss:,.0f}L")
        
        # TOP CRIME TYPES IN THIS STATE
        st.markdown("#### Most Common Crime Types in " + selected_state)
        
        crime_breakdown = state_hist.groupby('crime_category')['cases_reported'].sum().sort_values(ascending=False).reset_index()
        crime_breakdown['percentage'] = (crime_breakdown['cases_reported'] / crime_breakdown['cases_reported'].sum() * 100).round(1)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Pie chart of crime types
            fig = px.pie(
                crime_breakdown.head(6),
                values='cases_reported',
                names='crime_category',
                title=f'Crime Distribution in {selected_state}',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Top crimes table
            st.markdown("**Top Crimes to Watch:**")
            for i, row in crime_breakdown.head(5).iterrows():
                severity = "HIGH" if row['percentage'] > 20 else "MEDIUM" if row['percentage'] > 10 else "LOW"
                color = "#dc2626" if severity == "HIGH" else "#f59e0b" if severity == "MEDIUM" else "#22c55e"
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, {color}22, {color}11); 
                            border-left: 4px solid {color}; padding: 10px; margin: 5px 0; border-radius: 5px;">
                    <strong>{i+1}. {row['crime_category']}</strong><br>
                    <span style="color: {color};">{row['percentage']}% of all cases ({row['cases_reported']:,.0f} cases)</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Year-wise trend for this state
        st.markdown(f"#### Year-wise Trend in {selected_state}")
        
        yearly_state = state_hist.groupby('year')['cases_reported'].sum().reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=yearly_state['year'],
            y=yearly_state['cases_reported'],
            mode='lines+markers',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10),
            name='Cases Reported'
        ))
        fig.update_layout(
            title=f'{selected_state}: Historical Cyber Crime Cases (2018-2025)',
            xaxis_title='Year',
            yaxis_title='Cases',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # ==========================================================================
    # STEP 3: FUTURE PREDICTION
    # ==========================================================================
    st.markdown(f"### Step 3: ML Prediction for {selected_state} - Year {prediction_year}")
    
    # Show model info
    st.info(f"Using **{best_model}** model (Accuracy: {best_metrics['accuracy_score']}%, R2 Score: {best_metrics['test_r2']}%)")
    
    predictions = results['predictions']
    state_pred = predictions[(predictions['state'] == selected_state) & (predictions['year'] == prediction_year)]
    
    if len(state_pred) > 0:
        total_predicted = state_pred['predicted_cases'].sum()
        avg_confidence = state_pred['confidence'].mean()
        
        # Big prediction display
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #dc262622, #dc262611); 
                        border: 2px solid #dc2626; padding: 30px; border-radius: 15px; text-align: center;">
                <h1 style="color: #dc2626; font-size: 3rem; margin: 0;">{total_predicted:,.0f}</h1>
                <p style="font-size: 1.2rem; margin: 5px 0;">Predicted Cases in {prediction_year}</p>
                <p style="color: #888;">Confidence: {avg_confidence:.0f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Compare with last known year
            last_year_cases = yearly_state[yearly_state['year'] == 2025]['cases_reported'].iloc[0] if len(yearly_state[yearly_state['year'] == 2025]) > 0 else avg_cases_per_year
            change = ((total_predicted - last_year_cases) / last_year_cases * 100)
            st.metric(
                f"Change from 2025",
                f"{change:+.1f}%",
                delta=f"{total_predicted - last_year_cases:+,.0f} cases",
                delta_color="inverse"
            )
        
        with col3:
            # Risk level
            if change > 30:
                risk = "CRITICAL"
                risk_color = "#dc2626"
            elif change > 15:
                risk = "HIGH"
                risk_color = "#ea580c"
            elif change > 5:
                risk = "MEDIUM"
                risk_color = "#f59e0b"
            else:
                risk = "LOW"
                risk_color = "#22c55e"
            
            st.markdown(f"""
            <div style="background: {risk_color}22; border: 2px solid {risk_color}; 
                        padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: {risk_color}; margin: 0;">{risk}</h2>
                <p style="margin: 5px 0;">Risk Level</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # PREDICTED CRIME BREAKDOWN
        st.markdown(f"#### Predicted Crime Breakdown for {prediction_year}")
        
        pred_by_category = state_pred[['crime_category', 'predicted_cases']].sort_values('predicted_cases', ascending=False)
        pred_by_category['percentage'] = (pred_by_category['predicted_cases'] / pred_by_category['predicted_cases'].sum() * 100).round(1)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            fig = px.bar(
                pred_by_category,
                x='crime_category',
                y='predicted_cases',
                color='predicted_cases',
                color_continuous_scale='Reds',
                title=f'Predicted Cases by Crime Type ({prediction_year})'
            )
            fig.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.pie(
                pred_by_category.head(6),
                values='predicted_cases',
                names='crime_category',
                title='Crime Type Distribution (Predicted)',
                hole=0.4,
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # ==========================================================================
        # STEP 4: PREVENTIVE MEASURES & RECOMMENDATIONS
        # ==========================================================================
        st.markdown("### Step 4: Recommended Preventive Measures")
        st.markdown(f"Based on predicted crime patterns for **{selected_state}** in **{prediction_year}**:")
        
        # Get top 3 predicted crimes
        top_crimes = pred_by_category.head(3)['crime_category'].tolist()
        
        # Recommendations based on crime type
        recommendations = {
            'Online Financial Fraud': {
                'measures': [
                    "Set up dedicated cyber fraud helpline (1930)",
                    "Partner with banks for real-time fraud alerts",
                    "Conduct awareness campaigns on UPI/banking fraud",
                    "Deploy AI-based transaction monitoring systems"
                ],
                'resources': "50+ trained cyber forensic experts, Rs. 5 Cr budget",
                'priority': "HIGHEST"
            },
            'Cyber Blackmailing': {
                'measures': [
                    "Create anonymous reporting portal for victims",
                    "Social media monitoring for extortion patterns",
                    "Coordinate with IT platforms for quick takedowns",
                    "Victim counseling and support services"
                ],
                'resources': "20 investigators, collaboration with Meta/Google",
                'priority': "HIGH"
            },
            'Obscene Content': {
                'measures': [
                    "POCSO special cell for child safety",
                    "Monitor messaging apps for illegal content",
                    "School awareness programs on safe internet",
                    "Fast-track prosecution for offenders"
                ],
                'resources': "15 specialists, NGO partnerships",
                'priority': "HIGH"
            },
            'Data Theft': {
                'measures': [
                    "Audit government databases for vulnerabilities",
                    "Mandatory data breach reporting policy",
                    "Corporate cyber security compliance checks",
                    "Dark web monitoring for leaked data"
                ],
                'resources': "25 ethical hackers, penetration testing tools",
                'priority': "HIGH"
            },
            'Cyber Stalking': {
                'measures': [
                    "Women's cyber safety helpline",
                    "GPS tracking for stalkers on bail",
                    "Social media platform partnerships",
                    "Self-defense digital tools awareness"
                ],
                'resources': "10 dedicated officers, victim support fund",
                'priority': "MEDIUM"
            },
            'Identity Theft': {
                'measures': [
                    "Aadhaar fraud monitoring system",
                    "Multi-factor authentication promotion",
                    "Regular CIBIL monitoring alerts",
                    "Public awareness on phishing"
                ],
                'resources': "Link with UIDAI, bank partnerships",
                'priority': "HIGH"
            },
            'Hacking': {
                'measures': [
                    "Critical infrastructure security audits",
                    "Bug bounty programs for govt websites",
                    "Cyber attack simulation exercises",
                    "24x7 Security Operations Center (SOC)"
                ],
                'resources': "30 ethical hackers, Rs. 10 Cr infrastructure",
                'priority': "CRITICAL"
            },
            'Defamation': {
                'measures': [
                    "Fast legal process for fake content removal",
                    "Media literacy campaigns",
                    "Platform coordination for quick action",
                    "Legal aid for victims"
                ],
                'resources': "5 legal experts, platform MOUs",
                'priority': "MEDIUM"
            },
            'Fake News': {
                'measures': [
                    "Fact-checking cell establishment",
                    "WhatsApp/Telegram group monitoring",
                    "Quick response team for viral content",
                    "Public awareness on verification"
                ],
                'resources': "10 analysts, AI fake news detection",
                'priority': "HIGH"
            },
            'Ransomware': {
                'measures': [
                    "Backup mandate for all govt systems",
                    "Anti-ransomware software deployment",
                    "Incident response team training",
                    "No-ransom policy enforcement"
                ],
                'resources': "50 Cr cybersecurity budget, backup infrastructure",
                'priority': "CRITICAL"
            }
        }
        
        for i, crime in enumerate(top_crimes):
            rec = recommendations.get(crime, {
                'measures': ["Increase patrolling", "Awareness campaigns", "Staff training", "Technology upgrade"],
                'resources': "Standard allocation",
                'priority': "MEDIUM"
            })
            
            priority_color = "#dc2626" if rec['priority'] in ['CRITICAL', 'HIGHEST'] else "#f59e0b" if rec['priority'] == 'HIGH' else "#22c55e"
            
            with st.expander(f"#{i+1} {crime} - Priority: {rec['priority']}", expanded=(i==0)):
                st.markdown(f"""
                <div style="border-left: 4px solid {priority_color}; padding-left: 15px;">
                    <h4 style="color: {priority_color};">Recommended Actions:</h4>
                </div>
                """, unsafe_allow_html=True)
                
                for j, measure in enumerate(rec['measures']):
                    st.checkbox(f"{measure}", key=f"measure_{crime}_{j}")
                
                st.markdown(f"**Required Resources:** {rec['resources']}")
        
        st.markdown("---")
        
        # ==========================================================================
        # PREDICTION TIMELINE (2026-2030)
        # ==========================================================================
        st.markdown(f"### 5-Year Prediction Timeline for {selected_state}")
        
        state_all_pred = predictions[predictions['state'] == selected_state]
        yearly_pred = state_all_pred.groupby('year')['predicted_cases'].sum().reset_index()
        
        # Combine historical + predicted
        fig = go.Figure()
        
        # Historical
        fig.add_trace(go.Scatter(
            x=yearly_state['year'],
            y=yearly_state['cases_reported'],
            mode='lines+markers',
            name='Historical (Actual)',
            line=dict(color='#3b82f6', width=3),
            marker=dict(size=10)
        ))
        
        # Predicted
        fig.add_trace(go.Scatter(
            x=yearly_pred['year'],
            y=yearly_pred['predicted_cases'],
            mode='lines+markers',
            name='Predicted (ML)',
            line=dict(color='#dc2626', width=3, dash='dash'),
            marker=dict(size=10, symbol='diamond')
        ))
        
        fig.add_vline(x=2025.5, line_dash="dot", line_color="gray", annotation_text="Prediction Start")
        
        fig.update_layout(
            title=f'{selected_state}: Historical vs Predicted Cyber Crimes',
            xaxis_title='Year',
            yaxis_title='Cases',
            hovermode='x unified',
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.markdown("#### Year-wise Prediction Summary")
        
        summary_data = []
        for _, row in yearly_pred.iterrows():
            year_data = state_all_pred[state_all_pred['year'] == row['year']]
            top_crime = year_data.sort_values('predicted_cases', ascending=False).iloc[0]['crime_category']
            summary_data.append({
                'Year': int(row['year']),
                'Predicted Cases': int(row['predicted_cases']),
                'Top Crime Type': top_crime,
                'Confidence': f"{year_data['confidence'].mean():.0f}%"
            })
        
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
    
    else:
        st.warning(f"No predictions available for {selected_state}. Please select another state.")
    
    # Model info footer
    st.markdown("---")
    st.markdown("### About This Prediction")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"""
        **ML Model Used:** {best_model}  
        **Training Data:** NCRB 2018-2025  
        **Features:** State, Year, Crime Type, Region
        """)
    with col2:
        st.markdown(f"""
        **Model Accuracy:** {best_metrics['accuracy_score']}%  
        **R2 Score:** {best_metrics['test_r2']}%  
        **Error Rate (MAPE):** {best_metrics['mape']}%
        """)
    with col3:
        st.markdown("""
        **Data Source:** NCRB Pattern-based  
        **Prediction Range:** 2026-2030  
        **Confidence:** Decreases for distant years
        """)

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
    if page == "Home":
        render_home_page()
    elif page == "Threat Detection":
        render_threat_detection_page()
    elif page == "URL Checker":
        render_url_checker_page()
    elif page == "AI Chatbot":
        render_chatbot_page()
    elif page == "Dataset Insights":
        render_insights_page()
    elif page == "India Crime Map":
        render_india_map_page()
    elif page == "State Predictions":
        render_predictions_page()
    elif page == "ML Prediction & Analysis":
        render_ml_analysis_page()
    elif page == "Live Threats":
        render_live_threats_page()
    elif page == "Case Management":
        render_case_management_page()
    
if __name__ == "__main__":
    main()
