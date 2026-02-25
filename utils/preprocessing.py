"""
Cyber-Sight: Data Preprocessing Module
======================================
This module handles data preprocessing for cyber crime detection.
Includes feature extraction, encoding, and data transformation functions.
"""

import pandas as pd
import numpy as np
import re
from urllib.parse import urlparse
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import Tuple, Dict, Any, List
import os


class DataPreprocessor:
    """
    Preprocessor class for cyber crime dataset.
    Handles feature extraction, encoding, and data transformation.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_columns = []
        
        # Suspicious keywords commonly found in malicious URLs
        self.suspicious_keywords = [
            'login', 'verify', 'secure', 'account', 'update', 'confirm',
            'banking', 'paypal', 'ebay', 'amazon', 'apple', 'microsoft',
            'google', 'facebook', 'instagram', 'twitter', 'netflix',
            'password', 'credential', 'signin', 'signup', 'auth',
            'free', 'winner', 'prize', 'claim', 'reward', 'gift',
            'urgent', 'suspended', 'locked', 'expired', 'alert',
            'hack', 'crack', 'exploit', 'malware', 'virus', 'trojan',
            'download', 'install', 'setup', 'exe', 'zip', 'rar'
        ]
        
        # Known safe domains (whitelist)
        self.safe_domains = [
            'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
            'facebook.com', 'twitter.com', 'linkedin.com', 'github.com',
            'stackoverflow.com', 'wikipedia.org', 'reddit.com', 'youtube.com',
            'netflix.com', 'spotify.com', 'dropbox.com', 'zoom.us',
            'slack.com', 'notion.so', 'figma.com', 'canva.com'
        ]
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work',
            '.click', '.link', '.info', '.biz', '.online', '.site',
            '.club', '.win', '.download', '.stream', '.racing'
        ]
    
    def extract_url_features(self, url: str) -> Dict[str, Any]:
        """
        Extract features from a URL for ML classification.
        
        Args:
            url: The URL to analyze
            
        Returns:
            Dictionary containing extracted features
        """
        features = {}
        
        try:
            # Parse the URL
            parsed = urlparse(url if '://' in url else f'http://{url}')
            domain = parsed.netloc or parsed.path.split('/')[0]
            path = parsed.path
            
            # Basic features
            features['url_length'] = len(url)
            features['domain_length'] = len(domain)
            
            # HTTPS check
            features['has_https'] = 1 if parsed.scheme == 'https' else 0
            
            # IP address check (URLs with direct IP are suspicious)
            ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
            features['has_ip'] = 1 if re.match(ip_pattern, domain) else 0
            
            # Count special characters
            features['num_dots'] = url.count('.')
            features['num_hyphens'] = url.count('-')
            features['num_underscores'] = url.count('_')
            features['num_slashes'] = url.count('/')
            features['num_digits'] = sum(c.isdigit() for c in url)
            features['num_at_symbols'] = url.count('@')
            features['num_question_marks'] = url.count('?')
            features['num_ampersands'] = url.count('&')
            features['num_equals'] = url.count('=')
            
            # Suspicious keyword check
            url_lower = url.lower()
            suspicious_count = sum(1 for kw in self.suspicious_keywords if kw in url_lower)
            features['has_suspicious_keywords'] = 1 if suspicious_count > 0 else 0
            features['suspicious_keyword_count'] = suspicious_count
            
            # Check for suspicious TLD
            features['has_suspicious_tld'] = 1 if any(domain.endswith(tld) for tld in self.suspicious_tlds) else 0
            
            # Check if domain is in safe list
            features['is_safe_domain'] = 1 if any(safe in domain for safe in self.safe_domains) else 0
            
            # Path depth (number of directories)
            features['path_depth'] = len([p for p in path.split('/') if p])
            
            # Subdomain count
            features['subdomain_count'] = len(domain.split('.')) - 2 if len(domain.split('.')) > 2 else 0
            
            # Check for URL shorteners patterns
            shortener_patterns = ['bit.ly', 'tinyurl', 't.co', 'goo.gl', 'ow.ly', 'is.gd']
            features['is_shortened'] = 1 if any(s in domain for s in shortener_patterns) else 0
            
            # Entropy of URL (randomness indicator)
            features['url_entropy'] = self._calculate_entropy(url)
            
        except Exception as e:
            # Return default features if parsing fails
            features = {
                'url_length': len(url),
                'domain_length': 0,
                'has_https': 0,
                'has_ip': 0,
                'num_dots': 0,
                'num_hyphens': 0,
                'num_underscores': 0,
                'num_slashes': 0,
                'num_digits': 0,
                'num_at_symbols': 0,
                'num_question_marks': 0,
                'num_ampersands': 0,
                'num_equals': 0,
                'has_suspicious_keywords': 0,
                'suspicious_keyword_count': 0,
                'has_suspicious_tld': 0,
                'is_safe_domain': 0,
                'path_depth': 0,
                'subdomain_count': 0,
                'is_shortened': 0,
                'url_entropy': 0
            }
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """
        Calculate Shannon entropy of a string.
        Higher entropy indicates more randomness (potentially suspicious).
        """
        if not text:
            return 0.0
        
        prob = [float(text.count(c)) / len(text) for c in set(text)]
        entropy = -sum(p * np.log2(p) for p in prob if p > 0)
        return round(entropy, 4)
    
    def load_dataset(self, filepath: str) -> pd.DataFrame:
        """
        Load the cyber crime dataset from CSV.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Loaded DataFrame
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        df = pd.read_csv(filepath)
        print(f"[OK] Loaded dataset with {len(df)} records and {len(df.columns)} columns")
        return df
    
    def preprocess_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Preprocess the dataset for ML training.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Tuple of (features, attack_type_labels, risk_level_labels)
        """
        # Create a copy to avoid modifying original
        data = df.copy()
        
        # Extract additional features from URLs if 'url' column exists
        if 'url' in data.columns:
            print("[OK] Extracting URL features...")
            url_features = data['url'].apply(self.extract_url_features)
            url_features_df = pd.DataFrame(url_features.tolist())
            
            # Add extracted features to dataframe
            for col in url_features_df.columns:
                if col not in data.columns:
                    data[col] = url_features_df[col]
        
        # Define feature columns (numeric features for ML)
        self.feature_columns = [
            'domain_length', 'has_ip', 'has_https', 'num_dots', 
            'num_hyphens', 'num_underscores', 'num_slashes', 'num_digits',
            'has_suspicious_keywords', 'url_length'
        ]
        
        # Add additional extracted features if they exist
        additional_features = [
            'num_at_symbols', 'num_question_marks', 'num_ampersands',
            'num_equals', 'suspicious_keyword_count', 'has_suspicious_tld',
            'is_safe_domain', 'path_depth', 'subdomain_count', 
            'is_shortened', 'url_entropy'
        ]
        
        for feat in additional_features:
            if feat in data.columns:
                self.feature_columns.append(feat)
        
        # Ensure all feature columns exist
        available_features = [col for col in self.feature_columns if col in data.columns]
        self.feature_columns = available_features
        
        print(f"[OK] Using {len(self.feature_columns)} features: {self.feature_columns}")
        
        # Extract features
        X = data[self.feature_columns].values
        
        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)
        
        # Encode target variables
        if 'attack_type' in data.columns:
            self.label_encoders['attack_type'] = LabelEncoder()
            y_attack = self.label_encoders['attack_type'].fit_transform(data['attack_type'])
        else:
            y_attack = np.zeros(len(data))
        
        if 'risk_level' in data.columns:
            self.label_encoders['risk_level'] = LabelEncoder()
            y_risk = self.label_encoders['risk_level'].fit_transform(data['risk_level'])
        else:
            y_risk = np.zeros(len(data))
        
        print(f"[OK] Preprocessed {len(X)} samples")
        print(f"[OK] Attack types: {list(self.label_encoders.get('attack_type', LabelEncoder()).classes_) if 'attack_type' in self.label_encoders else 'N/A'}")
        print(f"[OK] Risk levels: {list(self.label_encoders.get('risk_level', LabelEncoder()).classes_) if 'risk_level' in self.label_encoders else 'N/A'}")
        
        return X, y_attack, y_risk
    
    def scale_features(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler.
        
        Args:
            X: Feature array
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Scaled feature array
        """
        if fit:
            return self.scaler.fit_transform(X)
        else:
            return self.scaler.transform(X)
    
    def decode_prediction(self, prediction: int, label_type: str) -> str:
        """
        Decode a numeric prediction back to its original label.
        
        Args:
            prediction: Numeric prediction
            label_type: 'attack_type' or 'risk_level'
            
        Returns:
            Original label string
        """
        if label_type in self.label_encoders:
            return self.label_encoders[label_type].inverse_transform([prediction])[0]
        return str(prediction)
    
    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        return self.feature_columns.copy()


def create_sample_dataset(output_path: str, num_samples: int = 100):
    """
    Create a sample cyber crime dataset for testing.
    
    Args:
        output_path: Path to save the CSV file
        num_samples: Number of samples to generate
    """
    import random
    
    # Sample data templates
    safe_domains = [
        'google.com', 'microsoft.com', 'apple.com', 'amazon.com',
        'facebook.com', 'github.com', 'linkedin.com', 'twitter.com'
    ]
    
    malicious_patterns = [
        'login-verify-{}.tk', 'secure-account-{}.xyz', 'free-{}-winner.net',
        'update-{}-now.ml', '{}-support-urgent.cf', 'claim-{}-prize.ga'
    ]
    
    attack_types = ['safe', 'phishing', 'malware', 'hacking']
    risk_levels = ['low', 'medium', 'high']
    countries = ['USA', 'Russia', 'China', 'India', 'Brazil', 'Unknown', 'Nigeria', 'Vietnam']
    
    data = []
    
    for i in range(num_samples):
        if random.random() < 0.4:  # 40% safe URLs
            domain = random.choice(safe_domains)
            url = f'https://{domain}/{random.choice(["home", "products", "about", "contact"])}'
            attack = 'safe'
            risk = 'low'
        else:  # 60% malicious URLs
            pattern = random.choice(malicious_patterns)
            brand = random.choice(['paypal', 'bank', 'apple', 'amazon', 'netflix'])
            url = f'http://{pattern.format(brand)}/login'
            attack = random.choice(['phishing', 'malware', 'hacking'])
            risk = random.choice(['medium', 'high'])
        
        parsed = urlparse(url if '://' in url else f'http://{url}')
        domain = parsed.netloc
        
        data.append({
            'url': url,
            'domain_length': len(domain),
            'has_ip': 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain) else 0,
            'has_https': 1 if parsed.scheme == 'https' else 0,
            'num_dots': url.count('.'),
            'num_hyphens': url.count('-'),
            'num_underscores': url.count('_'),
            'num_slashes': url.count('/'),
            'num_digits': sum(c.isdigit() for c in url),
            'has_suspicious_keywords': 1 if any(kw in url.lower() for kw in ['login', 'verify', 'free', 'winner']) else 0,
            'url_length': len(url),
            'attack_type': attack,
            'risk_level': risk,
            'country': random.choice(countries),
            'timestamp': f'2025-01-{random.randint(1,28):02d} {random.randint(0,23):02d}:{random.randint(0,59):02d}:00'
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"[OK] Created sample dataset with {num_samples} records at {output_path}")
    return df


if __name__ == "__main__":
    # Test the preprocessor
    print("Testing DataPreprocessor...")
    
    preprocessor = DataPreprocessor()
    
    # Test URL feature extraction
    test_urls = [
        'https://google.com/search',
        'http://192.168.1.1/admin',
        'http://paypal-secure-login.tk/verify',
        'https://microsoft.com/security'
    ]
    
    print("\nURL Feature Extraction Test:")
    for url in test_urls:
        features = preprocessor.extract_url_features(url)
        print(f"\n{url}")
        print(f"  Features: {features}")
