"""
Cyber-Sight: Authentication Module
==================================
Login system with CAPTCHA for secure access.
Designed for cyber crime police departments.
"""

import hashlib
import random
import string
import json
import os
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import streamlit as st


@dataclass
class User:
    """User data class."""
    username: str
    role: str  # 'admin', 'officer', 'analyst', 'viewer'
    full_name: str
    department: str
    state: str
    last_login: Optional[str] = None


class CaptchaGenerator:
    """
    Simple CAPTCHA generator for login security.
    Just type the displayed letters - simple and effective.
    """
    
    @staticmethod
    def generate() -> Dict[str, str]:
        """
        Generate a simple letter CAPTCHA.
        User just needs to type the displayed letters.
        
        Returns:
            Dict with 'display' (letters to type) and 'answer' (correct answer)
        """
        # Use only uppercase letters that are easy to read
        # Avoid confusing letters like O/0, I/1/L
        chars = 'ABCDEFGHJKMNPQRSTUVWXYZ'
        length = random.randint(5, 6)
        captcha_text = ''.join(random.choices(chars, k=length))
        
        return {
            'display': captcha_text,
            'answer': captcha_text,
            'type': 'text'
        }
    
    @staticmethod
    def generate_math_captcha() -> Dict[str, any]:
        """Generate simple CAPTCHA - redirects to main generate."""
        return CaptchaGenerator.generate()
    
    @staticmethod
    def generate_text_captcha(length: int = 5) -> Dict[str, str]:
        """Generate simple CAPTCHA - redirects to main generate."""
        return CaptchaGenerator.generate()
    
    @staticmethod
    def generate_word_captcha() -> Dict[str, str]:
        """Generate simple CAPTCHA - redirects to main generate."""
        return CaptchaGenerator.generate()
    
    @staticmethod
    def generate_simple_captcha() -> Dict[str, str]:
        """Alias for generate() - simple letter CAPTCHA."""
        return CaptchaGenerator.generate()
    
    # Keep old method for compatibility but redirect
    @staticmethod
    def _old_word_captcha() -> Dict[str, str]:
        """Old word captcha - not used anymore."""
        questions = [
            ("What color is the sky on a clear day?", "blue"),
            ("How many days are in a week?", "7"),
            ("What is the capital of India?", "delhi"),
            ("What comes after 'one, two, ...'?", "three"),
            ("Which planet do we live on?", "earth"),
            ("How many months in a year?", "12"),
            ("What is the first letter of the alphabet?", "a"),
            ("What animal says 'meow'?", "cat"),
            ("What is the opposite of 'hot'?", "cold"),
            ("What color is grass?", "green"),
            ("What is the last letter of the alphabet?", "z"),
            ("How many legs does a dog have?", "4"),
            ("What comes after Monday?", "tuesday"),
            ("What do bees make?", "honey"),
            ("What is the opposite of 'up'?", "down"),
            ("What animal barks?", "dog"),
            ("What is frozen water called?", "ice"),
            ("How many hours in a day?", "24"),
            ("What season comes after winter?", "spring"),
            ("What color is a banana?", "yellow"),
            ("What is the largest ocean?", "pacific"),
            ("Capital of Maharashtra?", "mumbai"),
            ("How many states in India?", "28"),
            ("National bird of India?", "peacock"),
            ("What color is snow?", "white"),
        ]
        
        question, answer = random.choice(questions)
        return {'display': question, 'answer': answer.lower(), 'type': 'word'}
    
    @staticmethod
    def verify_captcha(captcha_data: Dict, user_answer: str) -> bool:
        """
        Verify a CAPTCHA answer.
        
        Args:
            captcha_data: Dict with 'answer' key containing correct answer
            user_answer: User's answer
            
        Returns:
            True if correct, False otherwise
        """
        if not captcha_data or not user_answer:
            return False
        
        correct = str(captcha_data.get('answer', '')).strip().lower()
        given = str(user_answer).strip().lower()
        
        return correct == given


class AuthenticationManager:
    """
    Manages user authentication with CAPTCHA verification.
    """
    
    # Default users for demo (in production, use proper database)
    DEFAULT_USERS = {
        'admin': {
            'password_hash': hashlib.sha256('admin123'.encode()).hexdigest(),
            'role': 'admin',
            'full_name': 'System Administrator',
            'department': 'Cyber Crime Cell',
            'state': 'Maharashtra'
        },
        'officer1': {
            'password_hash': hashlib.sha256('officer123'.encode()).hexdigest(),
            'role': 'officer',
            'full_name': 'Inspector Sharma',
            'department': 'Cyber Crime Division',
            'state': 'Delhi'
        },
        'analyst1': {
            'password_hash': hashlib.sha256('analyst123'.encode()).hexdigest(),
            'role': 'analyst',
            'full_name': 'Analyst Patel',
            'department': 'Analysis Wing',
            'state': 'Karnataka'
        },
        'viewer1': {
            'password_hash': hashlib.sha256('viewer123'.encode()).hexdigest(),
            'role': 'viewer',
            'full_name': 'Public User',
            'department': 'Public Access',
            'state': 'Tamil Nadu'
        }
    }
    
    ROLE_PERMISSIONS = {
        'admin': ['all'],
        'officer': ['threat_detection', 'url_checker', 'chatbot', 'insights', 'live_threats', 'state_prediction', 'case_management'],
        'analyst': ['threat_detection', 'url_checker', 'chatbot', 'insights', 'state_prediction'],
        'viewer': ['chatbot', 'insights', 'state_prediction']
    }
    
    def __init__(self):
        """Initialize authentication manager."""
        self.users = self.DEFAULT_USERS.copy()
        self.captcha_generator = CaptchaGenerator()
        self.login_attempts = {}
        self.max_attempts = 3
        self.lockout_duration = 300  # 5 minutes
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def verify_credentials(self, username: str, password: str) -> Tuple[bool, Optional[User]]:
        """
        Verify user credentials.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Tuple of (success, User object or None)
        """
        username = username.lower().strip()
        
        if username not in self.users:
            return False, None
        
        user_data = self.users[username]
        password_hash = self._hash_password(password)
        
        if password_hash == user_data['password_hash']:
            user = User(
                username=username,
                role=user_data['role'],
                full_name=user_data['full_name'],
                department=user_data['department'],
                state=user_data['state'],
                last_login=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            )
            return True, user
        
        return False, None
    
    def authenticate(self, username: str, password: str) -> Dict:
        """
        Authenticate a user - wrapper for verify_credentials with lockout check.
        
        Args:
            username: Username
            password: Password
            
        Returns:
            Dict with 'success', 'message', and 'user' keys
        """
        username = username.lower().strip()
        
        # Check lockout first
        is_locked, remaining = self.check_lockout(username)
        if is_locked:
            return {
                'success': False,
                'message': f'Account locked. Try again in {remaining} seconds.',
                'user': None
            }
        
        # Verify credentials
        success, user = self.verify_credentials(username, password)
        
        if success:
            self.reset_attempts(username)
            return {
                'success': True,
                'message': 'Login successful',
                'user': user
            }
        else:
            self.record_failed_attempt(username)
            attempts = self.login_attempts.get(username, {}).get('count', 0)
            remaining_attempts = self.max_attempts - attempts
            
            if remaining_attempts > 0:
                return {
                    'success': False,
                    'message': f'Invalid credentials. {remaining_attempts} attempts remaining.',
                    'user': None
                }
            else:
                return {
                    'success': False,
                    'message': f'Account locked for {self.lockout_duration} seconds due to multiple failed attempts.',
                    'user': None
                }
    
    def check_lockout(self, username: str) -> Tuple[bool, int]:
        """
        Check if user is locked out due to failed attempts.
        
        Returns:
            Tuple of (is_locked, remaining_seconds)
        """
        username = username.lower().strip()
        
        if username not in self.login_attempts:
            return False, 0
        
        attempts = self.login_attempts[username]
        
        if attempts['count'] >= self.max_attempts:
            elapsed = (datetime.now() - attempts['last_attempt']).total_seconds()
            remaining = self.lockout_duration - elapsed
            
            if remaining > 0:
                return True, int(remaining)
            else:
                # Reset after lockout period
                self.login_attempts[username] = {'count': 0, 'last_attempt': datetime.now()}
        
        return False, 0
    
    def record_failed_attempt(self, username: str):
        """Record a failed login attempt."""
        username = username.lower().strip()
        
        if username not in self.login_attempts:
            self.login_attempts[username] = {'count': 0, 'last_attempt': datetime.now()}
        
        self.login_attempts[username]['count'] += 1
        self.login_attempts[username]['last_attempt'] = datetime.now()
    
    def reset_attempts(self, username: str):
        """Reset login attempts after successful login."""
        username = username.lower().strip()
        if username in self.login_attempts:
            del self.login_attempts[username]
    
    def has_permission(self, user: User, feature: str) -> bool:
        """Check if user has permission for a feature."""
        permissions = self.ROLE_PERMISSIONS.get(user.role, [])
        return 'all' in permissions or feature in permissions
    
    def get_new_captcha(self, captcha_type: str = 'math') -> Dict[str, str]:
        """
        Generate a new CAPTCHA.
        
        Args:
            captcha_type: 'math', 'text', or 'word'
            
        Returns:
            Dict with 'display', 'answer', and 'type' keys
        """
        if captcha_type == 'math':
            return self.captcha_generator.generate_math_captcha()
        elif captcha_type == 'text':
            return self.captcha_generator.generate_text_captcha()
        else:
            return self.captcha_generator.generate_word_captcha()


def render_login_page():
    """Render the login page with CAPTCHA."""
    
    # Initialize auth manager in session state
    if 'auth_manager' not in st.session_state:
        st.session_state.auth_manager = AuthenticationManager()
    
    if 'captcha' not in st.session_state:
        st.session_state.captcha = st.session_state.auth_manager.get_new_captcha('math')
    
    if 'captcha_type' not in st.session_state:
        st.session_state.captcha_type = 'math'
    
    # Page styling
    st.markdown("""
    <style>
        .login-container {
            max-width: 400px;
            margin: 0 auto;
            padding: 2rem;
            background: linear-gradient(135deg, #1e3a5f 0%, #0f172a 100%);
            border-radius: 15px;
            border: 1px solid #334155;
        }
        .login-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .captcha-box {
            background: #1e293b;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin: 1rem 0;
            font-size: 1.2rem;
            font-weight: bold;
            letter-spacing: 3px;
            border: 2px dashed #3b82f6;
        }
        .credentials-info {
            background: #0f172a;
            padding: 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            margin-top: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="login-header">
            <h1>🔒 Cyber-Sight</h1>
            <p style="color: #94a3b8;">Cyber Crime Detection Platform</p>
            <p style="color: #64748b; font-size: 0.9rem;">For Law Enforcement Use</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Login form
        st.markdown("### 🔐 Secure Login")
        
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔑 Password", type="password", placeholder="Enter password")
        
        # Check lockout
        if username:
            is_locked, remaining = st.session_state.auth_manager.check_lockout(username)
            if is_locked:
                st.error(f"🚫 Account temporarily locked. Try again in {remaining} seconds.")
                return False
        
        # CAPTCHA Section
        st.markdown("### 🤖 Security Verification")
        
        # CAPTCHA type selector
        captcha_type = st.radio(
            "CAPTCHA Type",
            ['math', 'word'],
            horizontal=True,
            format_func=lambda x: "🔢 Math" if x == 'math' else "📝 Question"
        )
        
        if captcha_type != st.session_state.captcha_type:
            st.session_state.captcha_type = captcha_type
            st.session_state.captcha = st.session_state.auth_manager.get_new_captcha(captcha_type)
        
        # Display CAPTCHA
        question, correct_answer = st.session_state.captcha
        
        st.markdown(f"""
        <div class="captcha-box">
            {question}
        </div>
        """, unsafe_allow_html=True)
        
        col_captcha, col_refresh = st.columns([3, 1])
        with col_captcha:
            captcha_answer = st.text_input("Enter Answer", placeholder="Type your answer", label_visibility="collapsed")
        with col_refresh:
            if st.button("🔄", help="Get new CAPTCHA"):
                st.session_state.captcha = st.session_state.auth_manager.get_new_captcha(st.session_state.captcha_type)
                st.rerun()
        
        st.markdown("")
        
        # Login button
        if st.button("🔓 Login", type="primary", use_container_width=True):
            if not username or not password:
                st.error("Please enter both username and password.")
                return False
            
            if not captcha_answer:
                st.error("Please complete the CAPTCHA verification.")
                return False
            
            # Verify CAPTCHA
            if captcha_answer.lower().strip() != correct_answer.lower().strip():
                st.error("[X] Incorrect CAPTCHA. Please try again.")
                st.session_state.captcha = st.session_state.auth_manager.get_new_captcha(st.session_state.captcha_type)
                st.rerun()
                return False
            
            # Verify credentials
            success, user = st.session_state.auth_manager.verify_credentials(username, password)
            
            if success:
                st.session_state.authenticated = True
                st.session_state.user = user
                st.session_state.auth_manager.reset_attempts(username)
                st.success(f"[OK] Welcome, {user.full_name}!")
                st.rerun()
                return True
            else:
                st.session_state.auth_manager.record_failed_attempt(username)
                attempts_left = st.session_state.auth_manager.max_attempts - \
                               st.session_state.auth_manager.login_attempts.get(username, {}).get('count', 0)
                st.error(f"[X] Invalid credentials. {attempts_left} attempts remaining.")
                st.session_state.captcha = st.session_state.auth_manager.get_new_captcha(st.session_state.captcha_type)
                st.rerun()
                return False
        
        # Demo credentials info
        with st.expander("📋 Demo Credentials"):
            st.markdown("""
            | Username | Password | Role |
            |----------|----------|------|
            | `admin` | `admin@123` | Administrator |
            | `officer` | `officer@123` | Police Officer |
            | `analyst` | `analyst@123` | Crime Analyst |
            | `demo` | `demo@123` | Viewer |
            """)
        
        st.markdown("---")
        st.markdown("""
        <p style="text-align: center; color: #64748b; font-size: 0.8rem;">
            [IND] Designed for Indian Cyber Crime Police Departments<br>
            [!] Unauthorized access is prohibited
        </p>
        """, unsafe_allow_html=True)
    
    return False


def check_authentication() -> bool:
    """Check if user is authenticated."""
    return st.session_state.get('authenticated', False)


def get_current_user() -> Optional[User]:
    """Get current logged-in user."""
    return st.session_state.get('user', None)


def logout():
    """Logout the current user."""
    st.session_state.authenticated = False
    st.session_state.user = None
    if 'captcha' in st.session_state:
        del st.session_state.captcha


def render_user_info():
    """Render user info in sidebar."""
    user = get_current_user()
    if user:
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 👤 Logged In As")
        st.sidebar.markdown(f"**{user.full_name}**")
        st.sidebar.caption(f"Role: {user.role.upper()}")
        st.sidebar.caption(f"Dept: {user.department}")
        st.sidebar.caption(f"State: {user.state}")
        
        if st.sidebar.button("🚪 Logout", use_container_width=True):
            logout()
            st.rerun()


if __name__ == "__main__":
    # Test the authentication module
    print("Testing Authentication Module...")
    
    auth = AuthenticationManager()
    
    # Test credentials
    success, user = auth.verify_credentials('admin', 'admin@123')
    print(f"Admin login: {success}, User: {user}")
    
    # Test CAPTCHA
    print("\nMath CAPTCHA:", auth.get_new_captcha('math'))
    print("Word CAPTCHA:", auth.get_new_captcha('word'))
    print("Text CAPTCHA:", auth.get_new_captcha('text'))
