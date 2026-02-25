"""
Cyber-Sight: Live Threat Monitoring Module
==========================================
Simulates real-time cyber threat feeds, hacking alerts,
and tampering detection for law enforcement use.
"""

import random
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import queue


class ThreatSeverity(Enum):
    """Threat severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ThreatType(Enum):
    """Types of cyber threats."""
    RANSOMWARE = "Ransomware Attack"
    DDOS = "DDoS Attack"
    PHISHING = "Phishing Campaign"
    DATA_BREACH = "Data Breach"
    MALWARE = "Malware Detection"
    INTRUSION = "Network Intrusion"
    SQL_INJECTION = "SQL Injection"
    XSS = "Cross-Site Scripting"
    BRUTE_FORCE = "Brute Force Attack"
    ZERO_DAY = "Zero-Day Exploit"
    APT = "Advanced Persistent Threat"
    CRYPTOJACKING = "Cryptojacking"
    MAN_IN_MIDDLE = "Man-in-the-Middle"
    DNS_SPOOFING = "DNS Spoofing"
    INSIDER_THREAT = "Insider Threat"


@dataclass
class LiveThreat:
    """Data class for live threat information."""
    id: str
    timestamp: datetime
    threat_type: ThreatType
    severity: ThreatSeverity
    source_ip: str
    source_country: str
    target_sector: str
    target_location: str
    description: str
    ioc: List[str]  # Indicators of Compromise
    status: str  # 'active', 'mitigated', 'investigating'
    affected_systems: int
    estimated_impact: str


class LiveThreatGenerator:
    """
    Generates simulated live threat data for demonstration.
    In production, this would connect to real threat intelligence feeds.
    """
    
    # Indian cities for target locations
    INDIAN_CITIES = [
        'Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai',
        'Kolkata', 'Pune', 'Ahmedabad', 'Jaipur', 'Lucknow',
        'Chandigarh', 'Bhopal', 'Kochi', 'Indore', 'Nagpur',
        'Coimbatore', 'Thiruvananthapuram', 'Guwahati', 'Patna', 'Ranchi'
    ]
    
    # Attacker source countries (common in threat intelligence)
    SOURCE_COUNTRIES = [
        'Russia', 'China', 'North Korea', 'Iran', 'Pakistan',
        'Unknown', 'Nigeria', 'Romania', 'Brazil', 'Vietnam',
        'Turkey', 'India', 'Indonesia', 'Ukraine', 'Netherlands'
    ]
    
    # Target sectors
    TARGET_SECTORS = [
        'Banking & Finance', 'Government', 'Healthcare', 'Education',
        'E-commerce', 'Telecom', 'Energy', 'Manufacturing', 'IT/ITES',
        'Defense', 'Transportation', 'Media', 'Legal', 'Retail', 'Insurance'
    ]
    
    # Sample IOCs (Indicators of Compromise)
    SAMPLE_IOCS = {
        'ips': [
            '185.220.101.', '45.155.205.', '91.219.236.', '194.147.140.',
            '103.136.42.', '202.65.145.', '123.58.196.', '61.177.172.'
        ],
        'domains': [
            'malicious-update.com', 'secure-login-verify.tk', 'payment-process.xyz',
            'account-verify-now.ml', 'urgent-action-required.ga', 'free-prize-claim.cf'
        ],
        'hashes': [
            'a1b2c3d4e5f6', 'deadbeef1234', 'cafebabe5678', 'f00dface9012'
        ]
    }
    
    # Threat descriptions templates
    DESCRIPTIONS = {
        ThreatType.RANSOMWARE: [
            "Ransomware variant '{name}' detected attempting to encrypt systems",
            "Active ransomware campaign targeting {sector} sector",
            "New ransomware strain spreading via phishing emails"
        ],
        ThreatType.DDOS: [
            "Volumetric DDoS attack detected - {volume} Gbps traffic spike",
            "Layer 7 DDoS attack targeting web services",
            "Distributed attack from botnet involving {count} IPs"
        ],
        ThreatType.PHISHING: [
            "Mass phishing campaign impersonating {brand}",
            "Spear phishing targeting {sector} executives",
            "Credential harvesting site detected"
        ],
        ThreatType.DATA_BREACH: [
            "Potential data exfiltration detected - {size} GB transferred",
            "Unauthorized database access from external IP",
            "Sensitive data exposure detected in {sector}"
        ],
        ThreatType.INTRUSION: [
            "Unauthorized access attempt to critical infrastructure",
            "Lateral movement detected in network",
            "Privilege escalation attempt blocked"
        ],
        ThreatType.MALWARE: [
            "Trojan '{name}' detected on endpoint",
            "Backdoor installation attempt blocked",
            "Spyware activity detected on {count} systems"
        ],
        ThreatType.APT: [
            "APT group '{name}' activity detected",
            "Nation-state threat actor targeting {sector}",
            "Sophisticated multi-stage attack in progress"
        ],
        ThreatType.ZERO_DAY: [
            "Zero-day exploit targeting {software} detected",
            "Unpatched vulnerability exploitation attempt",
            "Novel attack vector identified"
        ]
    }
    
    RANSOMWARE_NAMES = ['LockBit', 'BlackCat', 'Conti', 'REvil', 'Hive', 'Royal', 'Play']
    APT_GROUPS = ['APT28', 'APT29', 'Lazarus', 'Kimsuky', 'Cobalt', 'FIN7', 'Turla']
    BRANDS = ['HDFC Bank', 'SBI', 'Paytm', 'Amazon India', 'Flipkart', 'IRCTC', 'Aadhaar']
    
    def __init__(self):
        """Initialize the threat generator."""
        self.threat_counter = 0
        self.active_threats = []
    
    def _generate_ip(self) -> str:
        """Generate a random IP address."""
        prefix = random.choice(self.SAMPLE_IOCS['ips'])
        return f"{prefix}{random.randint(1, 254)}"
    
    def _generate_iocs(self, threat_type: ThreatType) -> List[str]:
        """Generate IOCs for a threat."""
        iocs = []
        
        # Add IP
        iocs.append(f"IP: {self._generate_ip()}")
        
        # Add domain for relevant threats
        if threat_type in [ThreatType.PHISHING, ThreatType.MALWARE, ThreatType.DATA_BREACH]:
            iocs.append(f"Domain: {random.choice(self.SAMPLE_IOCS['domains'])}")
        
        # Add file hash for malware-related threats
        if threat_type in [ThreatType.RANSOMWARE, ThreatType.MALWARE, ThreatType.APT]:
            hash_val = ''.join(random.choices('0123456789abcdef', k=32))
            iocs.append(f"MD5: {hash_val}")
        
        return iocs
    
    def _generate_description(self, threat_type: ThreatType, sector: str) -> str:
        """Generate threat description."""
        templates = self.DESCRIPTIONS.get(threat_type, ["Cyber threat detected"])
        template = random.choice(templates)
        
        return template.format(
            name=random.choice(self.RANSOMWARE_NAMES if threat_type == ThreatType.RANSOMWARE else self.APT_GROUPS),
            sector=sector,
            brand=random.choice(self.BRANDS),
            volume=random.randint(10, 500),
            count=random.randint(100, 10000),
            size=random.randint(1, 100),
            software=random.choice(['Apache', 'Windows', 'Linux', 'Oracle', 'SAP'])
        )
    
    def generate_threat(self) -> LiveThreat:
        """Generate a single live threat."""
        self.threat_counter += 1
        
        threat_type = random.choice(list(ThreatType))
        
        # Severity distribution (more low/medium, fewer critical)
        severity_weights = [0.05, 0.15, 0.35, 0.30, 0.15]  # CRITICAL, HIGH, MEDIUM, LOW, INFO
        severity = random.choices(list(ThreatSeverity), weights=severity_weights)[0]
        
        sector = random.choice(self.TARGET_SECTORS)
        
        threat = LiveThreat(
            id=f"THR-{datetime.now().strftime('%Y%m%d')}-{self.threat_counter:05d}",
            timestamp=datetime.now() - timedelta(seconds=random.randint(0, 300)),
            threat_type=threat_type,
            severity=severity,
            source_ip=self._generate_ip(),
            source_country=random.choice(self.SOURCE_COUNTRIES),
            target_sector=sector,
            target_location=random.choice(self.INDIAN_CITIES),
            description=self._generate_description(threat_type, sector),
            ioc=self._generate_iocs(threat_type),
            status=random.choice(['active', 'active', 'investigating', 'mitigated']),
            affected_systems=random.randint(1, 500),
            estimated_impact=random.choice(['Low', 'Medium', 'High', 'Critical'])
        )
        
        return threat
    
    def generate_batch(self, count: int = 10) -> List[LiveThreat]:
        """Generate a batch of threats."""
        return [self.generate_threat() for _ in range(count)]
    
    def get_threat_stats(self, threats: List[LiveThreat]) -> Dict:
        """Calculate statistics from threat list."""
        if not threats:
            return {}
        
        severity_counts = {}
        type_counts = {}
        country_counts = {}
        sector_counts = {}
        
        for t in threats:
            severity_counts[t.severity.value] = severity_counts.get(t.severity.value, 0) + 1
            type_counts[t.threat_type.value] = type_counts.get(t.threat_type.value, 0) + 1
            country_counts[t.source_country] = country_counts.get(t.source_country, 0) + 1
            sector_counts[t.target_sector] = sector_counts.get(t.target_sector, 0) + 1
        
        return {
            'total_threats': len(threats),
            'critical_count': severity_counts.get('CRITICAL', 0),
            'high_count': severity_counts.get('HIGH', 0),
            'active_threats': len([t for t in threats if t.status == 'active']),
            'severity_distribution': severity_counts,
            'threat_types': type_counts,
            'top_source_countries': dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'targeted_sectors': dict(sorted(sector_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'total_affected_systems': sum(t.affected_systems for t in threats)
        }


class ThreatAlertSystem:
    """
    Alert system for critical threats.
    """
    
    def __init__(self):
        self.alerts = queue.Queue()
        self.alert_history = []
    
    def create_alert(self, threat: LiveThreat) -> Dict:
        """Create an alert from a threat."""
        alert = {
            'id': f"ALT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(1000, 9999)}",
            'timestamp': datetime.now().isoformat(),
            'threat_id': threat.id,
            'severity': threat.severity.value,
            'title': f"{threat.severity.value}: {threat.threat_type.value} Detected",
            'message': threat.description,
            'location': threat.target_location,
            'sector': threat.target_sector,
            'action_required': self._get_action(threat),
            'acknowledged': False
        }
        
        self.alert_history.append(alert)
        return alert
    
    def _get_action(self, threat: LiveThreat) -> str:
        """Get recommended action for threat."""
        actions = {
            ThreatSeverity.CRITICAL: "IMMEDIATE ACTION REQUIRED: Isolate affected systems, notify CERT-In, initiate incident response",
            ThreatSeverity.HIGH: "URGENT: Block source IPs, scan for IOCs, prepare incident report",
            ThreatSeverity.MEDIUM: "Monitor closely, update signatures, alert security team",
            ThreatSeverity.LOW: "Log for analysis, update threat intelligence",
            ThreatSeverity.INFO: "No immediate action required, add to threat database"
        }
        return actions.get(threat.severity, "Investigate and document")
    
    def get_unacknowledged_alerts(self) -> List[Dict]:
        """Get all unacknowledged alerts."""
        return [a for a in self.alert_history if not a['acknowledged']]
    
    def acknowledge_alert(self, alert_id: str):
        """Mark an alert as acknowledged."""
        for alert in self.alert_history:
            if alert['id'] == alert_id:
                alert['acknowledged'] = True
                break


class TamperingDetector:
    """
    Detects potential tampering and unauthorized modifications.
    """
    
    TAMPERING_EVENTS = [
        "Unauthorized configuration change detected on firewall",
        "System log files modified outside of maintenance window",
        "Database schema altered without approval",
        "Security policy disabled on endpoint",
        "Privileged account created without authorization",
        "Audit log tampering detected",
        "File integrity check failed on critical system",
        "Registry modification detected on server",
        "Certificate store modified",
        "Backup schedule altered",
        "Network ACL modified",
        "Service account password changed",
        "Admin group membership modified",
        "Encryption keys accessed from unusual location",
        "Multi-factor authentication disabled"
    ]
    
    def __init__(self):
        self.tampering_events = []
    
    def generate_tampering_event(self) -> Dict:
        """Generate a simulated tampering event."""
        event = {
            'id': f"TMP-{datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(100, 999)}",
            'timestamp': datetime.now().isoformat(),
            'event_type': random.choice(self.TAMPERING_EVENTS),
            'system': f"SRV-{random.choice(['WEB', 'DB', 'APP', 'FW', 'DC'])}-{random.randint(1, 50):02d}",
            'user': random.choice(['unknown', 'system', f'user{random.randint(100, 999)}']),
            'source_ip': f"192.168.{random.randint(1, 254)}.{random.randint(1, 254)}",
            'severity': random.choice(['CRITICAL', 'HIGH', 'MEDIUM']),
            'details': f"Change detected at {datetime.now().strftime('%H:%M:%S')}",
            'remediation': "Investigate immediately, verify with system owner, restore if unauthorized"
        }
        
        self.tampering_events.append(event)
        return event
    
    def get_recent_events(self, count: int = 10) -> List[Dict]:
        """Get recent tampering events."""
        return self.tampering_events[-count:]


# Global instances for use in app
threat_generator = LiveThreatGenerator()
alert_system = ThreatAlertSystem()
tampering_detector = TamperingDetector()


def get_live_threat_feed(count: int = 20) -> List[LiveThreat]:
    """Get current live threat feed."""
    return threat_generator.generate_batch(count)


def get_threat_statistics(threats: List[LiveThreat]) -> Dict:
    """Get statistics from threats."""
    return threat_generator.get_threat_stats(threats)


if __name__ == "__main__":
    print("Testing Live Threat Monitoring Module...")
    
    # Generate sample threats
    threats = get_live_threat_feed(10)
    
    print(f"\nGenerated {len(threats)} threats:")
    for t in threats[:5]:
        print(f"  [{t.severity.value}] {t.threat_type.value} - {t.target_location}")
    
    # Get stats
    stats = get_threat_statistics(threats)
    print(f"\nStatistics:")
    print(f"  Critical: {stats['critical_count']}")
    print(f"  High: {stats['high_count']}")
    print(f"  Active: {stats['active_threats']}")
    
    # Generate tampering event
    tampering = tampering_detector.generate_tampering_event()
    print(f"\nTampering Event: {tampering['event_type']}")
