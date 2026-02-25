"""
Cyber-Sight: URL Safety Checker Module
======================================
This module provides comprehensive URL safety analysis using
both heuristic rules and ML-based classification.
"""

import re
import os
import joblib
import numpy as np
from urllib.parse import urlparse
from typing import Dict, Any, Tuple, List
from dataclasses import dataclass


@dataclass
class URLAnalysisResult:
    """Data class for URL analysis results."""
    url: str
    safety_status: str  # 'SAFE', 'SUSPICIOUS', 'MALICIOUS'
    risk_level: str  # 'Low', 'Medium', 'High'
    threat_type: str  # 'safe', 'phishing', 'malware', 'hacking', 'unknown'
    confidence: float
    reasons: List[str]
    recommendations: List[str]


class URLSafetyChecker:
    """
    Comprehensive URL safety checker combining heuristic rules
    and ML-based classification.
    """
    
    def __init__(self, model_path: str = None):
        """
        Initialize the URL safety checker.
        
        Args:
            model_path: Path to the trained ML model
        """
        self.model = None
        self.preprocessor = None
        
        # Load ML model if available
        if model_path and os.path.exists(model_path):
            try:
                model_data = joblib.load(model_path)
                self.model = model_data.get('attack_model')
                self.risk_model = model_data.get('risk_model')
                self.scaler = model_data.get('scaler')
                self.label_encoders = model_data.get('label_encoders', {})
                self.feature_columns = model_data.get('feature_columns', [])
                print("[OK] ML model loaded successfully")
            except Exception as e:
                print(f"⚠ Could not load ML model: {e}")
        
        # Suspicious keywords commonly found in malicious URLs
        self.suspicious_keywords = [
            'login', 'verify', 'secure', 'account', 'update', 'confirm',
            'banking', 'paypal', 'ebay', 'amazon', 'apple', 'microsoft',
            'password', 'credential', 'signin', 'signup', 'auth',
            'free', 'winner', 'prize', 'claim', 'reward', 'gift',
            'urgent', 'suspended', 'locked', 'expired', 'alert',
            'hack', 'crack', 'exploit', 'malware', 'virus', 'trojan',
            'download', 'install', 'setup', 'exe', 'zip', 'rar'
        ]
        
        # Known safe domains (whitelist)
        self.safe_domains = [
            'google.com', 'google.co.in', 'microsoft.com', 'apple.com', 
            'amazon.com', 'amazon.in', 'facebook.com', 'twitter.com', 
            'linkedin.com', 'github.com', 'stackoverflow.com', 'wikipedia.org',
            'reddit.com', 'youtube.com', 'netflix.com', 'spotify.com',
            'dropbox.com', 'zoom.us', 'slack.com', 'notion.so',
            'figma.com', 'canva.com', 'adobe.com', 'salesforce.com',
            'paypal.com', 'stripe.com', 'cloudflare.com', 'aws.amazon.com',
            'azure.microsoft.com', 'cloud.google.com'
        ]
        
        # Suspicious TLDs often used in phishing
        self.suspicious_tlds = [
            '.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top', '.work',
            '.click', '.link', '.biz', '.online', '.site', '.club', 
            '.win', '.download', '.stream', '.racing', '.party', '.review',
            '.country', '.science', '.date', '.faith', '.accountant'
        ]
        
        # Executable file extensions
        self.dangerous_extensions = [
            '.exe', '.dll', '.bat', '.cmd', '.msi', '.vbs', '.js',
            '.jar', '.scr', '.pif', '.com', '.hta', '.wsf', '.ps1'
        ]
        
        # Known phishing brand keywords
        self.brand_keywords = [
            'paypal', 'apple', 'microsoft', 'amazon', 'google', 'facebook',
            'netflix', 'bank', 'chase', 'wells', 'fargo', 'citi', 'hsbc',
            'barclays', 'santander', 'amex', 'visa', 'mastercard'
        ]
    
    def check_url(self, url: str) -> URLAnalysisResult:
        """
        Perform comprehensive URL safety analysis.
        
        Args:
            url: The URL to analyze
            
        Returns:
            URLAnalysisResult object with analysis details
        """
        reasons = []
        risk_score = 0  # 0-100 scale
        
        # Normalize URL
        if not url.startswith(('http://', 'https://')):
            url = 'http://' + url
        
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            path = parsed.path.lower()
            full_url = url.lower()
        except Exception:
            return URLAnalysisResult(
                url=url,
                safety_status='SUSPICIOUS',
                risk_level='Medium',
                threat_type='unknown',
                confidence=0.5,
                reasons=['Unable to parse URL structure'],
                recommendations=['Avoid visiting this URL']
            )
        
        # ============ HEURISTIC CHECKS ============
        
        # 1. Check HTTPS
        if parsed.scheme != 'https':
            risk_score += 10
            reasons.append("[!] URL does not use HTTPS encryption")
        else:
            reasons.append("[OK] URL uses HTTPS encryption")
        
        # 2. Check for IP address instead of domain
        ip_pattern = r'^(\d{1,3}\.){3}\d{1,3}(:\d+)?$'
        if re.match(ip_pattern, domain):
            risk_score += 30
            reasons.append("[ALERT] URL uses direct IP address instead of domain name")
        
        # 3. Check domain against safe list
        is_safe_domain = any(safe in domain for safe in self.safe_domains)
        if is_safe_domain:
            risk_score -= 30
            reasons.append("[OK] Domain is from a known trusted source")
        
        # 4. Check for suspicious TLD
        if any(domain.endswith(tld) for tld in self.suspicious_tlds):
            risk_score += 25
            reasons.append("[!] URL uses a suspicious top-level domain (TLD)")
        
        # 5. Check domain length (long domains are suspicious)
        if len(domain) > 30:
            risk_score += 15
            reasons.append("⚠ Unusually long domain name")
        
        # 6. Check for suspicious keywords
        suspicious_found = [kw for kw in self.suspicious_keywords if kw in full_url]
        if suspicious_found and not is_safe_domain:
            risk_score += min(len(suspicious_found) * 5, 25)
            reasons.append(f"⚠ Contains suspicious keywords: {', '.join(suspicious_found[:5])}")
        
        # 7. Check for brand impersonation
        brands_found = [b for b in self.brand_keywords if b in domain]
        if brands_found and not is_safe_domain:
            risk_score += 30
            reasons.append(f"🚨 Possible brand impersonation: {', '.join(brands_found)}")
        
        # 8. Check for dangerous file extensions
        if any(path.endswith(ext) for ext in self.dangerous_extensions):
            risk_score += 35
            reasons.append("🚨 URL points to a potentially dangerous file type")
        
        # 9. Check for excessive special characters
        special_chars = sum(1 for c in domain if c in '-_.')
        if special_chars > 3:
            risk_score += 10
            reasons.append("⚠ Domain contains multiple special characters")
        
        # 10. Check for @ symbol in URL (credential injection)
        if '@' in url:
            risk_score += 30
            reasons.append("🚨 URL contains @ symbol (possible credential injection)")
        
        # 11. Check URL length
        if len(url) > 100:
            risk_score += 10
            reasons.append("⚠ Unusually long URL")
        
        # 12. Check for multiple subdomains
        subdomain_count = len(domain.split('.')) - 2
        if subdomain_count > 2:
            risk_score += 15
            reasons.append(f"⚠ Multiple subdomains detected ({subdomain_count})")
        
        # 13. Check for URL encoding abuse
        if '%' in url:
            encoded_count = url.count('%')
            if encoded_count > 5:
                risk_score += 15
                reasons.append("⚠ Excessive URL encoding detected")
        
        # ============ ML-BASED CLASSIFICATION ============
        
        ml_threat_type = None
        ml_risk_level = None
        ml_confidence = 0.0
        
        if self.model is not None:
            try:
                features = self._extract_features(url)
                features_scaled = self.scaler.transform([features])
                
                # Predict threat type
                threat_pred = self.model.predict(features_scaled)[0]
                threat_proba = self.model.predict_proba(features_scaled)[0]
                ml_confidence = max(threat_proba)
                
                # Decode prediction
                if 'attack_type' in self.label_encoders:
                    ml_threat_type = self.label_encoders['attack_type'].inverse_transform([threat_pred])[0]
                else:
                    ml_threat_type = str(threat_pred)
                
                # Predict risk level
                if self.risk_model is not None:
                    risk_pred = self.risk_model.predict(features_scaled)[0]
                    if 'risk_level' in self.label_encoders:
                        ml_risk_level = self.label_encoders['risk_level'].inverse_transform([risk_pred])[0]
                
                # Adjust risk score based on ML prediction
                if ml_threat_type in ['phishing', 'malware', 'hacking']:
                    risk_score += int(ml_confidence * 40)
                    reasons.append(f"🤖 ML Model detected: {ml_threat_type} (confidence: {ml_confidence:.1%})")
                elif ml_threat_type == 'safe':
                    risk_score -= int(ml_confidence * 20)
                    reasons.append(f"🤖 ML Model assessment: Likely safe (confidence: {ml_confidence:.1%})")
                    
            except Exception as e:
                reasons.append(f"⚠ ML analysis unavailable: {str(e)}")
        
        # ============ DETERMINE FINAL VERDICT ============
        
        # Clamp risk score
        risk_score = max(0, min(100, risk_score))
        
        # Determine safety status and risk level
        if risk_score < 25:
            safety_status = 'SAFE'
            risk_level = 'Low'
        elif risk_score < 50:
            safety_status = 'SUSPICIOUS'
            risk_level = 'Medium'
        else:
            safety_status = 'MALICIOUS'
            risk_level = 'High'
        
        # Override with ML prediction if confident
        if ml_risk_level:
            risk_level = ml_risk_level.capitalize()
        
        # Determine threat type
        if ml_threat_type and ml_threat_type != 'safe':
            threat_type = ml_threat_type
        elif safety_status == 'SAFE':
            threat_type = 'safe'
        elif any(kw in full_url for kw in ['login', 'verify', 'account', 'password']):
            threat_type = 'phishing'
        elif any(ext in path for ext in self.dangerous_extensions):
            threat_type = 'malware'
        elif any(kw in full_url for kw in ['hack', 'exploit', 'crack']):
            threat_type = 'hacking'
        else:
            threat_type = 'unknown'
        
        # Generate recommendations
        recommendations = self._generate_recommendations(safety_status, threat_type)
        
        # Calculate confidence
        confidence = min(0.95, 0.5 + (risk_score / 200))
        if ml_confidence > 0:
            confidence = (confidence + ml_confidence) / 2
        
        return URLAnalysisResult(
            url=url,
            safety_status=safety_status,
            risk_level=risk_level,
            threat_type=threat_type,
            confidence=confidence,
            reasons=reasons,
            recommendations=recommendations
        )
    
    def _extract_features(self, url: str) -> List[float]:
        """Extract ML features from URL."""
        try:
            parsed = urlparse(url if '://' in url else f'http://{url}')
            domain = parsed.netloc or parsed.path.split('/')[0]
            
            features = {
                'domain_length': len(domain),
                'has_ip': 1 if re.match(r'^(\d{1,3}\.){3}\d{1,3}$', domain) else 0,
                'has_https': 1 if parsed.scheme == 'https' else 0,
                'num_dots': url.count('.'),
                'num_hyphens': url.count('-'),
                'num_underscores': url.count('_'),
                'num_slashes': url.count('/'),
                'num_digits': sum(c.isdigit() for c in url),
                'has_suspicious_keywords': 1 if any(kw in url.lower() for kw in self.suspicious_keywords) else 0,
                'url_length': len(url)
            }
            
            # Return features in the order expected by the model
            if self.feature_columns:
                return [features.get(col, 0) for col in self.feature_columns]
            else:
                return list(features.values())
                
        except Exception:
            return [0] * 10  # Default features
    
    def _generate_recommendations(self, safety_status: str, threat_type: str) -> List[str]:
        """Generate safety recommendations based on analysis."""
        recommendations = []
        
        if safety_status == 'SAFE':
            recommendations.append("[OK] This URL appears to be safe to visit")
            recommendations.append("-> Always verify you're on the correct website before entering sensitive information")
        
        elif safety_status == 'SUSPICIOUS':
            recommendations.append("[!] Exercise caution before visiting this URL")
            recommendations.append("-> Verify the URL carefully before clicking")
            recommendations.append("-> Do not enter personal or financial information")
            recommendations.append("-> Consider using a URL scanner service for additional verification")
        
        else:  # MALICIOUS
            recommendations.append("[X] DO NOT visit this URL")
            recommendations.append("-> This URL shows multiple indicators of malicious activity")
            
            if threat_type == 'phishing':
                recommendations.append("-> This appears to be a phishing attempt to steal your credentials")
                recommendations.append("→ Report this URL to your IT department or anti-phishing services")
            elif threat_type == 'malware':
                recommendations.append("→ This URL may attempt to download malware to your device")
                recommendations.append("→ Ensure your antivirus software is up to date")
            elif threat_type == 'hacking':
                recommendations.append("→ This URL is associated with hacking tools or activities")
                recommendations.append("→ Report this URL to cybersecurity authorities")
            
            recommendations.append("→ If you've already visited this URL, scan your device for malware")
        
        return recommendations
    
    def batch_check(self, urls: List[str]) -> List[URLAnalysisResult]:
        """
        Check multiple URLs at once.
        
        Args:
            urls: List of URLs to analyze
            
        Returns:
            List of URLAnalysisResult objects
        """
        return [self.check_url(url) for url in urls]
    
    def get_threat_summary(self, result: URLAnalysisResult) -> str:
        """
        Generate a human-readable summary of the URL analysis.
        
        Args:
            result: URLAnalysisResult object
            
        Returns:
            Formatted summary string
        """
        summary = f"""
╔══════════════════════════════════════════════════════════════╗
║                    URL SAFETY ANALYSIS                        ║
╠══════════════════════════════════════════════════════════════╣
║ URL: {result.url[:55]}{'...' if len(result.url) > 55 else ''}
╠══════════════════════════════════════════════════════════════╣
║ STATUS: {result.safety_status:12} │ RISK LEVEL: {result.risk_level:8}       ║
║ THREAT TYPE: {result.threat_type:10} │ CONFIDENCE: {result.confidence:.1%}        ║
╠══════════════════════════════════════════════════════════════╣
║ ANALYSIS DETAILS:                                             ║
"""
        for reason in result.reasons[:8]:
            summary += f"║ • {reason[:58]:<58} ║\n"
        
        summary += """╠══════════════════════════════════════════════════════════════╣
║ RECOMMENDATIONS:                                              ║
"""
        for rec in result.recommendations[:5]:
            summary += f"║ {rec[:60]:<60} ║\n"
        
        summary += "╚══════════════════════════════════════════════════════════════╝"
        
        return summary


def analyze_domain(domain: str) -> Dict[str, Any]:
    """
    Analyze a domain name for potential threats.
    
    Args:
        domain: Domain name to analyze
        
    Returns:
        Dictionary with analysis results
    """
    checker = URLSafetyChecker()
    result = checker.check_url(f"https://{domain}")
    
    return {
        'domain': domain,
        'safety_status': result.safety_status,
        'risk_level': result.risk_level,
        'threat_type': result.threat_type,
        'confidence': result.confidence,
        'reasons': result.reasons,
        'recommendations': result.recommendations
    }


def quick_check(url: str) -> str:
    """
    Quick URL safety check returning a simple status.
    
    Args:
        url: URL to check
        
    Returns:
        Safety status string
    """
    checker = URLSafetyChecker()
    result = checker.check_url(url)
    return f"{result.safety_status} ({result.risk_level} risk)"


if __name__ == "__main__":
    # Test the URL checker
    print("Testing URL Safety Checker...")
    
    checker = URLSafetyChecker()
    
    test_urls = [
        'https://google.com/search',
        'http://192.168.1.1/admin',
        'http://paypal-secure-login.tk/verify',
        'https://microsoft.com/security',
        'http://free-iphone-winner.xyz/claim',
        'http://malware-download.net/virus.exe'
    ]
    
    for url in test_urls:
        result = checker.check_url(url)
        print(checker.get_threat_summary(result))
        print()
