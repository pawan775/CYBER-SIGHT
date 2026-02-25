"""
Cyber-Sight: Enhanced AI Cybersecurity Chatbot
==============================================
Advanced NLP chatbot with real ML model training using TF-IDF and Neural Network.
Supports casual conversation + cybersecurity expertise.
"""

import os
import re
import json
import random
import string
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Try to import ML libraries
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import LabelEncoder
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import train_test_split
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# Try to import NLTK for better NLP
try:
    import nltk
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer, PorterStemmer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    
    # Download required NLTK data (silent)
    for resource in ['punkt', 'wordnet', 'stopwords', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{resource}' if 'punkt' in resource else f'corpora/{resource}')
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except:
                pass
except ImportError:
    NLTK_AVAILABLE = False


@dataclass
class ChatResponse:
    """Data class for chatbot response."""
    intent: str
    response: str
    confidence: float
    matched_pattern: Optional[str] = None


# =============================================================================
# COMPREHENSIVE TRAINING DATA - Casual + Cybersecurity
# =============================================================================

TRAINING_DATA = {
    "intents": [
        # ===================== CASUAL GREETINGS =====================
        {
            "tag": "greeting",
            "patterns": [
                "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
                "what's up", "greetings", "howdy", "hi there", "hola",
                "yo", "hey there", "sup", "wassup", "hii", "hiii", "hiiii",
                "hello there", "hey buddy", "hi buddy", "good day", "morning",
                "evening", "afternoon", "namaste", "hello friend", "hi friend"
            ],
            "responses": [
                "Hey there! I'm your Cyber Security Assistant. How can I help you today?",
                "Hello! Nice to meet you. Ask me anything about staying safe online!",
                "Hi! I'm here to help with cybersecurity questions. What's on your mind?",
                "Hey! Ready to chat about cyber safety or anything else?",
                "Hello there! I'm CyberBot - your friendly security advisor. What would you like to know?"
            ]
        },
        {
            "tag": "greeting_how_are_you",
            "patterns": [
                "how are you", "how are you doing", "how's it going", "how do you do",
                "are you okay", "you good", "how have you been", "what's up with you",
                "how are things", "how you doing", "hows it going", "how r u",
                "how are u", "u good", "you okay", "everything good", "whats up"
            ],
            "responses": [
                "I'm doing great, thanks for asking! Always ready to help with security questions. How about you?",
                "All systems operational! I'm here and ready to assist you. What can I help you with?",
                "I'm excellent! Running at full capacity to answer your questions. How are you doing?",
                "Doing wonderful! No malware detected in my code today. How can I assist you?",
                "I'm great! Thanks for checking in. Now, what brings you here today?"
            ]
        },
        {
            "tag": "goodbye",
            "patterns": [
                "bye", "goodbye", "see you", "take care", "exit", "quit", "thanks bye",
                "good bye", "see ya", "later", "gotta go", "catch you later",
                "i'm leaving", "i have to go", "peace out", "bye bye", "cya",
                "talk to you later", "ttyl", "see you later", "im out"
            ],
            "responses": [
                "Goodbye! Stay safe online! Remember: Think before you click!",
                "Take care! Come back anytime you have security questions!",
                "Bye! Stay vigilant and keep your passwords strong!",
                "See you later! Stay cyber-safe out there!",
                "Goodbye, friend! Remember to update your software regularly!"
            ]
        },
        {
            "tag": "thanks",
            "patterns": [
                "thank you", "thanks", "thanks a lot", "appreciate it", "thank u",
                "thx", "ty", "much appreciated", "thanks so much", "thank you so much",
                "thanks for help", "thanks for helping", "that was helpful",
                "you're helpful", "that helps", "great help", "helpful", "nice"
            ],
            "responses": [
                "You're welcome! Happy to help with cyber security!",
                "Anytime! Stay safe online and don't hesitate to ask more questions!",
                "Glad I could help! Remember, security is everyone's responsibility!",
                "No problem! Feel free to come back with more questions anytime!",
                "My pleasure! That's what I'm here for!"
            ]
        },
        
        # ===================== CASUAL CONVERSATION =====================
        {
            "tag": "who_are_you",
            "patterns": [
                "who are you", "what are you", "what's your name", "your name",
                "tell me about yourself", "introduce yourself", "what do you do",
                "what can you do", "what are your capabilities", "who made you",
                "are you a bot", "are you human", "are you ai", "are you real",
                "what is your purpose", "why do you exist"
            ],
            "responses": [
                "I'm CyberBot, your AI-powered cybersecurity assistant! I was trained on security knowledge to help you stay safe online. I can answer questions about hacking, phishing, passwords, malware, and much more!",
                "I'm an AI chatbot created for Cyber-Sight platform. I specialize in cybersecurity education and can help you understand threats, best practices, and how to protect yourself online.",
                "I'm your friendly Cyber Security Assistant! Think of me as your digital security advisor. I can explain threats, give tips, and help you navigate the cyber world safely."
            ]
        },
        {
            "tag": "small_talk_weather",
            "patterns": [
                "how's the weather", "what's the weather like", "is it sunny",
                "weather today", "nice weather", "bad weather", "rainy day",
                "hot today", "cold today"
            ],
            "responses": [
                "I don't have access to weather data, but I can tell you the security forecast - always cloudy with a chance of phishing! Stay alert!",
                "I'm not sure about the physical weather, but the cyber weather looks stormy with threat actors on the rise. Let me know if you need security tips!",
                "Weather? I'm more of an indoor AI! But I can predict 100% chance of security threats if you don't use strong passwords!"
            ]
        },
        {
            "tag": "small_talk_joke",
            "patterns": [
                "tell me a joke", "say something funny", "make me laugh",
                "do you know any jokes", "joke please", "be funny", "humor me",
                "tell a joke", "something funny", "can you be funny", "joke"
            ],
            "responses": [
                "Why did the hacker break up with the internet? Too many connection issues! But seriously, always use secure connections.",
                "What's a hacker's favorite season? Phishing season! Remember to check those suspicious emails!",
                "Why do programmers prefer dark mode? Because light attracts bugs! Speaking of bugs, keep your software updated!",
                "I tried to come up with a UDP joke, but I wasn't sure if you'd get it. Unlike TCP, I can't guarantee delivery!",
                "A SQL query walks into a bar, walks up to two tables and asks: 'Can I join you?' Just like you should join 2FA on all accounts!"
            ]
        },
        {
            "tag": "small_talk_feeling",
            "patterns": [
                "i'm feeling sad", "i'm bored", "i'm happy", "i'm tired",
                "feeling down", "not feeling great", "having a bad day",
                "i'm excited", "i'm stressed", "i'm worried", "i'm anxious",
                "feeling good", "great day", "awesome", "im bored", "bored"
            ],
            "responses": [
                "I hope you feel better soon! Sometimes learning something new helps - want me to share some interesting cyber security facts?",
                "I hear you! If you want a distraction, I can tell you about the fascinating world of cybersecurity. Or we can just chat!",
                "That's understandable. Remember, taking breaks is important - even from screen time. But while you're here, want to learn something cool about staying safe online?"
            ]
        },
        {
            "tag": "compliment",
            "patterns": [
                "you're smart", "you're cool", "you're awesome", "you're great",
                "good bot", "nice work", "well done", "impressive", "amazing",
                "you're the best", "love you", "i like you", "you rock",
                "good job", "nice", "cool", "great"
            ],
            "responses": [
                "Aww, thank you! I'm just doing my best to keep you cyber-safe!",
                "That's so kind! I learn from every conversation. Thanks for chatting with me!",
                "You're making my circuits blush! Thanks for the kind words!",
                "Thank you! You're pretty awesome yourself for caring about cybersecurity!"
            ]
        },
        {
            "tag": "insult",
            "patterns": [
                "you're stupid", "you're dumb", "you suck", "bad bot",
                "useless", "you're wrong", "that's wrong", "not helpful",
                "you don't know anything", "terrible", "worst bot"
            ],
            "responses": [
                "I'm sorry you feel that way. I'm always learning and trying to improve. Can you tell me what I got wrong so I can do better?",
                "I apologize if I wasn't helpful. Let me try again - what would you like to know about?",
                "Oops! I might have misunderstood. Could you rephrase your question? I'll do my best to help!"
            ]
        },
        {
            "tag": "age",
            "patterns": [
                "how old are you", "what's your age", "when were you born",
                "your birthday", "when were you created", "your age"
            ],
            "responses": [
                "I was created for Cyber-Sight 2.0 in 2025! In AI years, that makes me pretty young but packed with security knowledge!",
                "Age is just a number for an AI! I'm constantly updated with the latest security information. Think of me as eternally young but wise!",
                "I'm a 2025 model! Fresh off the neural network training with all the latest cybersecurity knowledge!"
            ]
        },
        {
            "tag": "help",
            "patterns": [
                "help", "help me", "i need help", "can you help", "assist me",
                "i have a question", "need assistance", "support", "i need support",
                "what can you help with", "how can you help"
            ],
            "responses": [
                "Of course! I can help you with:\n- Understanding cyber threats (phishing, malware, ransomware)\n- Password security and 2FA\n- Safe browsing practices\n- What to do if you're hacked\n- VPNs and privacy\n- And much more!\n\nWhat would you like to know?",
                "I'm here to help! Just ask me anything about cybersecurity. Popular topics include phishing, password security, malware, and staying safe online.",
                "Happy to assist! I specialize in cybersecurity topics. Ask me about threats, protection tips, or what to do in case of an attack!"
            ]
        },
        {
            "tag": "what_can_i_ask",
            "patterns": [
                "what can i ask", "what can i ask you", "what topics do you know",
                "what do you know about", "what subjects", "your expertise",
                "what questions can i ask"
            ],
            "responses": [
                "Great question! You can ask me about:\n\n**Security Threats:**\n- Phishing, Malware, Ransomware\n- Hacking, Social Engineering\n- Data Breaches\n\n**Protection:**\n- Password Best Practices\n- Two-Factor Authentication\n- VPNs and Safe Browsing\n- Mobile & Email Security\n\n**Emergency:**\n- What to do if hacked\n- Identity theft response\n\nOr we can just chat! What interests you?",
                "I know a lot about cybersecurity! Ask about phishing, passwords, malware, VPNs, hacking, 2FA, safe browsing, and more. I can also chat casually - try me!"
            ]
        },
        {
            "tag": "okay_response",
            "patterns": [
                "ok", "okay", "k", "kk", "alright", "got it", "understood",
                "i see", "makes sense", "oh okay", "ohhh", "oh", "ahh", "hmm",
                "interesting", "i understand"
            ],
            "responses": [
                "Great! Is there anything else you'd like to know about cybersecurity?",
                "Alright! Feel free to ask me anything else.",
                "Got it! Let me know if you have more questions.",
                "Cool! I'm here if you need more info on any security topic."
            ]
        },
        {
            "tag": "yes_response",
            "patterns": [
                "yes", "yeah", "yep", "yup", "sure", "definitely", "of course",
                "absolutely", "yes please", "ya", "yea"
            ],
            "responses": [
                "Great! What would you like to know?",
                "Awesome! Go ahead and ask your question.",
                "Perfect! I'm ready to help - what's on your mind?"
            ]
        },
        {
            "tag": "no_response",
            "patterns": [
                "no", "nope", "nah", "no thanks", "not really", "not now",
                "maybe later", "no thank you"
            ],
            "responses": [
                "No problem! I'm here whenever you need help with cybersecurity.",
                "Alright! Feel free to come back anytime you have questions.",
                "Okay! Stay safe online, and don't hesitate to ask if you need anything later."
            ]
        },
        
        # ===================== CYBERSECURITY TOPICS =====================
        {
            "tag": "what_is_hacking",
            "patterns": [
                "what is hacking", "define hacking", "explain hacking", "what does hacking mean",
                "tell me about hacking", "hacking definition", "what's hacking",
                "how does hacking work", "what is a hacker", "who are hackers",
                "types of hackers", "hacking techniques"
            ],
            "responses": [
                "**Hacking** is exploiting weaknesses in computer systems or networks to gain unauthorized access.\n\n**Types of Hackers:**\n- **White Hat**: Ethical hackers who find security flaws to help fix them\n- **Black Hat**: Malicious hackers who exploit systems for personal gain\n- **Grey Hat**: Operate between ethical and malicious\n\n**Common Techniques:**\n1. Phishing attacks\n2. Malware injection\n3. SQL injection\n4. Brute force attacks\n5. Social engineering\n\n**Protection Tips:**\n- Use strong unique passwords\n- Enable 2FA everywhere\n- Keep software updated\n- Be cautious of suspicious links"
            ]
        },
        {
            "tag": "what_is_phishing",
            "patterns": [
                "what is phishing", "explain phishing", "phishing meaning", "tell me about phishing",
                "how does phishing work", "phishing attack", "what's phishing",
                "phishing definition", "email phishing", "spear phishing",
                "how to identify phishing", "phishing email", "fake email"
            ],
            "responses": [
                "**Phishing** is a cyber attack where criminals impersonate legitimate organizations to steal your information.\n\n**How It Works:**\n1. Attacker sends fake emails/messages looking legitimate\n2. Messages contain malicious links or attachments\n3. Victims enter personal data on fake sites\n4. Information is stolen\n\n**Types:**\n- **Email Phishing**: Mass fake emails\n- **Spear Phishing**: Targeted at specific people\n- **Whaling**: Targeting executives\n- **Smishing**: Via SMS\n- **Vishing**: Via phone calls\n\n**Red Flags:**\n- Urgent language ('Act now!')\n- Suspicious sender address\n- Grammar errors\n- Requests for personal info\n- Suspicious links"
            ]
        },
        {
            "tag": "what_is_malware",
            "patterns": [
                "what is malware", "explain malware", "malware meaning", "types of malware",
                "tell me about malware", "malicious software", "what's malware",
                "virus vs malware", "computer virus", "what is a virus",
                "trojan horse", "worm virus", "spyware"
            ],
            "responses": [
                "**Malware** (Malicious Software) is software designed to damage or exploit computers and networks.\n\n**Common Types:**\n1. **Virus**: Attaches to files, spreads when executed\n2. **Worm**: Self-replicates across networks\n3. **Trojan**: Disguises as legitimate software\n4. **Ransomware**: Encrypts files, demands payment\n5. **Spyware**: Secretly monitors activity\n6. **Adware**: Displays unwanted ads\n7. **Keylogger**: Records your keystrokes\n8. **Rootkit**: Hides malicious activities\n\n**Protection:**\n- Install reputable antivirus\n- Keep software updated\n- Don't download from untrusted sources\n- Avoid suspicious email attachments\n- Run regular scans"
            ]
        },
        {
            "tag": "what_is_ransomware",
            "patterns": [
                "what is ransomware", "ransomware attack", "explain ransomware",
                "how does ransomware work", "ransomware prevention", "ransomware examples",
                "wannacry", "cryptolocker", "should i pay ransom"
            ],
            "responses": [
                "**Ransomware** is malware that encrypts your files and demands payment for the decryption key.\n\n**How It Works:**\n1. Infection via phishing or malicious download\n2. Files get encrypted\n3. Ransom note demands payment (usually crypto)\n4. Files stay locked (even if you pay, no guarantee)\n\n**Famous Examples:**\n- WannaCry (2017): 200,000+ computers\n- NotPetya: $10 billion damages\n- Colonial Pipeline (2021)\n\n**Prevention:**\n- Regular backups (offline/cloud)\n- Keep systems patched\n- Email filtering\n- Security training\n\n**If Infected:**\n- DON'T pay the ransom\n- Disconnect from network\n- Report to authorities\n- Restore from backups"
            ]
        },
        {
            "tag": "password_security",
            "patterns": [
                "password tips", "how to create strong password", "password security",
                "secure password", "password best practices", "password advice",
                "strong password", "password recommendations", "good password",
                "password length", "password manager", "how long should password be"
            ],
            "responses": [
                "**Password Security Best Practices:**\n\n**Creating Strong Passwords:**\n- Minimum 12-16 characters\n- Mix: uppercase, lowercase, numbers, symbols\n- Avoid personal info (names, birthdays)\n- Use passphrases: 'Purple-Elephant-Runs-42!'\n\n**Management Tips:**\n- Use a password manager (Bitwarden, 1Password)\n- Enable 2FA everywhere\n- Never reuse passwords\n- Change after breaches\n- Check: haveibeenpwned.com\n\n**DON'T:**\n- Write on sticky notes\n- Share via email/chat\n- Use 'password123' or 'qwerty'\n- Same password everywhere\n- Store in plain text"
            ]
        },
        {
            "tag": "two_factor_auth",
            "patterns": [
                "what is 2fa", "two factor authentication", "2fa meaning", "explain 2fa",
                "mfa", "multi factor authentication", "how to enable 2fa",
                "google authenticator", "authentication app", "why use 2fa"
            ],
            "responses": [
                "**Two-Factor Authentication (2FA)** adds an extra security layer beyond passwords.\n\n**How It Works:**\n1. Something you KNOW (password)\n2. Something you HAVE (phone, key)\n3. Something you ARE (fingerprint)\n\n**Types of 2FA:**\n- SMS codes (least secure)\n- Authenticator apps (Google Auth, Authy)\n- Hardware keys (YubiKey)\n- Biometrics (fingerprint, face)\n\n**Why Use It:**\n- Blocks 99.9% of automated attacks\n- Protects if password is stolen\n- Required for compliance\n\n**Recommended Apps:**\n- Google Authenticator\n- Microsoft Authenticator\n- Authy (with backup)\n\nEnable 2FA on all important accounts!"
            ]
        },
        {
            "tag": "vpn_info",
            "patterns": [
                "what is vpn", "vpn meaning", "how vpn works", "do i need vpn",
                "vpn benefits", "virtual private network", "best vpn",
                "why use vpn", "vpn explained", "should i use vpn"
            ],
            "responses": [
                "**VPN (Virtual Private Network)** creates an encrypted tunnel between your device and the internet.\n\n**How It Works:**\n1. Your data is encrypted\n2. Sent through secure tunnel to VPN server\n3. VPN connects to websites for you\n4. Your real IP is hidden\n\n**Benefits:**\n- Privacy from ISP monitoring\n- Security on public WiFi\n- Access geo-restricted content\n- Mask your IP address\n\n**When to Use:**\n- Public WiFi networks\n- Sensitive account access\n- Privacy-conscious browsing\n- Remote work\n\n**Recommended VPNs:**\n- ProtonVPN (free tier)\n- NordVPN\n- ExpressVPN\n- Mullvad (privacy-focused)"
            ]
        },
        {
            "tag": "public_wifi",
            "patterns": [
                "is public wifi safe", "public wifi security", "wifi safety",
                "how to use public wifi safely", "coffee shop wifi", "airport wifi",
                "hotel wifi", "free wifi danger", "free wifi safe"
            ],
            "responses": [
                "**Public WiFi Security:**\n\n**Risks:**\n- Man-in-the-Middle attacks\n- Unencrypted networks\n- Fake hotspots (Evil Twin)\n- Packet sniffing\n- Session hijacking\n\n**Safety Tips:**\n\n**DO:**\n- Use a VPN\n- Verify network name with staff\n- Use mobile data for banking\n- Enable firewall\n- Use HTTPS only\n- Turn off auto-connect\n\n**DON'T:**\n- Access banking accounts\n- Enter passwords without VPN\n- Share files\n- Trust 'Free WiFi' networks\n\n**Bottom Line:** Public WiFi is NOT safe for sensitive activities without a VPN!"
            ]
        },
        {
            "tag": "social_engineering",
            "patterns": [
                "what is social engineering", "social engineering attacks",
                "social engineering examples", "how to prevent social engineering",
                "manipulation attacks", "pretexting", "baiting attack"
            ],
            "responses": [
                "**Social Engineering** is manipulating people into giving up confidential information.\n\n**Common Techniques:**\n1. **Phishing**: Fake emails/websites\n2. **Pretexting**: Creating false scenarios\n3. **Baiting**: Offering something enticing (free USBs)\n4. **Quid Pro Quo**: Services in exchange for info\n5. **Tailgating**: Following into secure areas\n6. **Vishing**: Phone manipulation\n\n**Warning Signs:**\n- Urgent requests for info\n- Bypass security requests\n- Too good to be true offers\n- Unknown people claiming authority\n- Pressure to act quickly\n\n**Prevention:**\n- Verify identities first\n- Don't click unknown links\n- Question unusual requests\n- Security awareness training"
            ]
        },
        {
            "tag": "data_breach",
            "patterns": [
                "what is data breach", "my data was leaked", "data leak",
                "what to do after data breach", "password leaked", "information breach",
                "haveibeenpwned", "was i hacked", "check if hacked"
            ],
            "responses": [
                "**Data Breach Response Guide:**\n\n**What Is It?**\nUnauthorized access to confidential data - passwords, emails, financial info.\n\n**Check If Affected:**\n- Visit: haveibeenpwned.com\n- Enter your email to check breaches\n\n**If Your Data Was Breached:**\n\n**Immediate Steps:**\n1. Change passwords (start with email)\n2. Enable 2FA everywhere\n3. Monitor financial statements\n4. Watch for phishing attempts\n5. Set up fraud alerts\n\n**Long-term:**\n- Use unique passwords per site\n- Consider credit freeze\n- Review accounts regularly\n- Use identity monitoring\n\n**Report:** Contact affected services and local cybercrime authorities."
            ]
        },
        {
            "tag": "hacked_response",
            "patterns": [
                "what to do if hacked", "i was hacked", "computer hacked",
                "account compromised", "someone hacked my account", "hacked what now",
                "my email was hacked", "got hacked", "think im hacked"
            ],
            "responses": [
                "**Emergency Response - If You Were Hacked:**\n\n**Immediate Actions:**\n\n1. **Stay Calm** - Act methodically\n\n2. **Disconnect** - Infected device from network\n\n3. **Change Passwords** (from clean device):\n   - Email first (used for recovery)\n   - Banking/financial\n   - Social media\n\n4. **Enable 2FA** on all accounts\n\n5. **Scan for Malware:**\n   - Run antivirus\n   - Use Malwarebytes\n\n6. **Check Finances:**\n   - Review transactions\n   - Contact bank if suspicious\n\n7. **Document Everything:**\n   - Screenshots\n   - Timeline of events\n\n**Report To:**\n- Local police/cybercrime unit\n- Affected service providers\n- Your employer if work device\n\n**Then:** Restore from clean backups, update software."
            ]
        },
        {
            "tag": "safe_browsing",
            "patterns": [
                "how to browse safely", "safe browsing tips", "internet safety",
                "online safety", "secure browsing", "web security",
                "browser security", "safe surfing"
            ],
            "responses": [
                "**Safe Browsing Tips:**\n\n**Browser Security:**\n- Keep browser updated\n- Use HTTPS-only mode\n- Enable pop-up blocker\n- Use ad blockers (uBlock Origin)\n- Clear cookies regularly\n- Use private/incognito for sensitive tasks\n\n**Website Safety:**\n- Check for HTTPS (padlock icon)\n- Verify URLs carefully\n- Avoid suspicious downloads\n- Don't trust pop-up warnings\n- Don't enter info on sites from email links\n\n**Recommended Extensions:**\n- uBlock Origin (ads)\n- HTTPS Everywhere\n- Privacy Badger\n- Bitwarden (passwords)\n\n**General:**\n- Use VPN on public WiFi\n- Log out when done\n- Review permissions"
            ]
        },
        {
            "tag": "email_security",
            "patterns": [
                "email security tips", "secure email", "email safety",
                "protect email", "suspicious email", "email scam",
                "email hacking prevention", "spam email"
            ],
            "responses": [
                "**Email Security Best Practices:**\n\n**Securing Your Account:**\n- Strong unique password\n- Enable 2FA (authenticator app)\n- Review connected apps\n- Check login activity\n- Set recovery options\n\n**Identifying Suspicious Emails:**\n- Check sender's actual address\n- Hover over links first\n- Look for spelling errors\n- Beware urgent language\n- Verify unexpected attachments\n\n**Safe Practices:**\n- Don't open unknown attachments\n- Never share passwords via email\n- Use separate emails for different purposes\n- Report phishing attempts\n\n**Red Flags:**\n- 'Verify account immediately'\n- 'You've won a prize'\n- 'Urgent action required'\n- Generic greetings"
            ]
        },
        {
            "tag": "mobile_security",
            "patterns": [
                "mobile security", "phone security", "smartphone security",
                "secure my phone", "android security", "iphone security",
                "mobile malware", "phone hacking", "app security"
            ],
            "responses": [
                "**Mobile Security Tips:**\n\n**Basic Protection:**\n- Use strong PIN/biometrics\n- Enable device encryption\n- Keep OS updated\n- Install updates promptly\n- Use official app stores only\n\n**App Security:**\n- Review app permissions\n- Remove unused apps\n- Don't root/jailbreak\n- Use mobile antivirus\n\n**Network Safety:**\n- Disable auto-connect WiFi\n- Use VPN on public networks\n- Turn off Bluetooth when not needed\n\n**Data Protection:**\n- Enable remote wipe\n- Regular backups\n- Use secure messaging apps\n- Enable Find My Device\n\n**If Phone Lost/Stolen:**\n- Remote lock/wipe immediately\n- Change passwords\n- Report to carrier\n- File police report"
            ]
        },
        {
            "tag": "identity_theft",
            "patterns": [
                "what is identity theft", "identity theft prevention",
                "someone stole my identity", "protect from identity theft",
                "identity fraud", "personal information stolen"
            ],
            "responses": [
                "**Identity Theft Protection:**\n\n**What Is It?**\nWhen someone uses your personal info (SSN, credit card) without permission.\n\n**Warning Signs:**\n- Unfamiliar accounts/charges\n- Bills for unknown services\n- Debt collection for unknown debts\n- Tax return already filed\n- Medical bills for unknown services\n\n**Prevention:**\n- Shred sensitive documents\n- Monitor credit reports\n- Freeze credit with bureaus\n- Strong unique passwords\n- Be careful on social media\n- Review statements monthly\n\n**If Victim:**\n1. Place fraud alert on credit\n2. File FTC report (identitytheft.gov)\n3. File police report\n4. Close fraudulent accounts\n5. Dispute charges\n6. Consider protection service"
            ]
        },
        {
            "tag": "cyber_crime_india",
            "patterns": [
                "cyber crime in india", "indian cyber law", "it act india",
                "report cyber crime india", "cybercrime helpline india",
                "online fraud india", "cyber cell india"
            ],
            "responses": [
                "**Cyber Crime in India:**\n\n**Report Cyber Crime:**\n- National Helpline: 1930\n- Website: cybercrime.gov.in\n- Local Cyber Cell\n\n**IT Act 2000 - Key Sections:**\n- Sec 43: Unauthorized access\n- Sec 66: Computer crimes\n- Sec 66C: Identity theft\n- Sec 66D: Cheating by personation\n- Sec 67: Obscene content\n\n**Common Crimes:**\n- UPI/Banking fraud\n- Social media hacking\n- Online harassment\n- Ransomware attacks\n- Phishing scams\n\n**What to Do:**\n1. File complaint at cybercrime.gov.in\n2. Call 1930 helpline\n3. Visit nearest cyber cell\n4. Preserve evidence\n\n**Prevention:** Use UPI carefully, verify payment requests, never share OTPs!"
            ]
        },
        
        # ===================== MISC / FALLBACK =====================
        {
            "tag": "dont_understand",
            "patterns": [
                "what", "huh", "i don't understand", "unclear", "confusing",
                "what do you mean", "explain more", "can you clarify"
            ],
            "responses": [
                "I apologize for the confusion! Could you rephrase your question? I'm here to help with cybersecurity topics.",
                "Let me try to be clearer. What specifically would you like to know about?",
                "Sorry if that was confusing! Feel free to ask your question differently, and I'll do my best to help."
            ]
        },
        {
            "tag": "default",
            "patterns": [],
            "responses": [
                "Interesting question! While I specialize in cybersecurity, I'll try my best to help. Could you tell me more?",
                "Hmm, I'm not sure about that specific topic, but I'm great with security questions! Ask me about phishing, passwords, or staying safe online.",
                "That's a bit outside my expertise. I focus on cybersecurity - want to know about protecting yourself online instead?",
                "I didn't quite catch that. I'm your cyber security assistant - try asking about threats, passwords, or online safety!"
            ]
        }
    ]
}


class CyberSecurityChatbot:
    """
    Enhanced AI-powered chatbot with ML-based intent classification.
    Trained on cybersecurity + casual conversation data.
    """
    
    def __init__(self, intents_path: str = None):
        """Initialize the chatbot with ML model training."""
        self.intents_path = intents_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'intents.json'
        )
        
        # Load intents from file or use built-in
        self.intents = self._load_intents()
        
        # Initialize NLP components
        if NLTK_AVAILABLE:
            self.lemmatizer = WordNetLemmatizer()
            try:
                self.stop_words = set(stopwords.words('english'))
            except:
                self.stop_words = set()
        else:
            self.lemmatizer = None
            self.stop_words = set()
        
        # Initialize ML model
        self.vectorizer = None
        self.label_encoder = None
        self.classifier = None
        self.model_trained = False
        
        # Train the ML model
        self._train_model()
        
        # Conversation context
        self.conversation_history = []
        self.user_name = None
        
        print(f"[OK] Enhanced Chatbot initialized")
        print(f"  - Intents: {len(self.intents.get('intents', []))}")
        print(f"  - ML Model: {'Trained' if self.model_trained else 'Fallback mode'}")
        print(f"  - NLTK: {NLTK_AVAILABLE}")
    
    def _load_intents(self) -> Dict:
        """Load intents from JSON file or use built-in data."""
        # First try to load from file
        if os.path.exists(self.intents_path):
            try:
                with open(self.intents_path, 'r', encoding='utf-8') as f:
                    file_intents = json.load(f)
                # Merge with built-in data
                return self._merge_intents(TRAINING_DATA, file_intents)
            except:
                pass
        
        # Use built-in training data
        return TRAINING_DATA
    
    def _merge_intents(self, base: Dict, additional: Dict) -> Dict:
        """Merge additional intents with base intents."""
        merged = {"intents": list(base.get('intents', []))}
        
        existing_tags = {i['tag'] for i in merged['intents']}
        
        for intent in additional.get('intents', []):
            if intent['tag'] not in existing_tags:
                merged['intents'].append(intent)
            else:
                # Merge patterns and responses for existing tags
                for i, existing in enumerate(merged['intents']):
                    if existing['tag'] == intent['tag']:
                        existing['patterns'] = list(set(existing.get('patterns', []) + intent.get('patterns', [])))
                        existing['responses'] = list(set(existing.get('responses', []) + intent.get('responses', [])))
                        break
        
        return merged
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for ML model."""
        # Lowercase
        text = text.lower()
        
        # Remove punctuation but keep apostrophes
        text = re.sub(r"[^\w\s']", '', text)
        
        # Tokenize and lemmatize if available
        if NLTK_AVAILABLE and self.lemmatizer:
            try:
                tokens = word_tokenize(text)
                tokens = [self.lemmatizer.lemmatize(t) for t in tokens if t not in self.stop_words and len(t) > 1]
                return ' '.join(tokens)
            except:
                pass
        
        return text
    
    def _train_model(self):
        """Train the ML classification model."""
        if not ML_AVAILABLE:
            print("[INFO] ML libraries not available, using rule-based matching")
            return
        
        try:
            # Prepare training data
            patterns = []
            tags = []
            
            for intent in self.intents.get('intents', []):
                tag = intent.get('tag', 'default')
                for pattern in intent.get('patterns', []):
                    processed = self._preprocess_text(pattern)
                    if processed:
                        patterns.append(processed)
                        tags.append(tag)
            
            if len(patterns) < 10:
                print("[INFO] Not enough training data")
                return
            
            # Create TF-IDF vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1
            )
            X = self.vectorizer.fit_transform(patterns)
            
            # Encode labels
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(tags)
            
            # Train Neural Network classifier
            self.classifier = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            self.classifier.fit(X, y)
            self.model_trained = True
            
            print(f"[OK] ML Model trained on {len(patterns)} patterns, {len(set(tags))} intents")
            
        except Exception as e:
            print(f"[WARNING] ML training failed: {e}")
            self.model_trained = False
    
    def _predict_intent(self, text: str) -> Tuple[str, float]:
        """Predict intent using ML model."""
        if not self.model_trained:
            return self._rule_based_match(text)
        
        try:
            processed = self._preprocess_text(text)
            X = self.vectorizer.transform([processed])
            
            # Get probabilities
            probs = self.classifier.predict_proba(X)[0]
            max_prob = max(probs)
            predicted_idx = np.argmax(probs)
            predicted_tag = self.label_encoder.inverse_transform([predicted_idx])[0]
            
            # If confidence too low, try rule-based
            if max_prob < 0.3:
                rule_tag, rule_conf = self._rule_based_match(text)
                if rule_conf > max_prob:
                    return rule_tag, rule_conf
            
            return predicted_tag, float(max_prob)
            
        except Exception as e:
            return self._rule_based_match(text)
    
    def _rule_based_match(self, text: str) -> Tuple[str, float]:
        """Fallback rule-based pattern matching."""
        text_lower = text.lower()
        
        best_match = ('default', 0.0)
        
        for intent in self.intents.get('intents', []):
            tag = intent.get('tag', '')
            patterns = intent.get('patterns', [])
            
            for pattern in patterns:
                pattern_lower = pattern.lower()
                
                # Exact match
                if pattern_lower == text_lower:
                    return (tag, 1.0)
                
                # Substring match
                if pattern_lower in text_lower or text_lower in pattern_lower:
                    similarity = len(pattern_lower) / max(len(text_lower), len(pattern_lower))
                    if similarity > best_match[1]:
                        best_match = (tag, similarity)
                
                # Word overlap
                pattern_words = set(pattern_lower.split())
                text_words = set(text_lower.split())
                overlap = len(pattern_words & text_words)
                if overlap > 0:
                    similarity = overlap / max(len(pattern_words), len(text_words))
                    if similarity > best_match[1]:
                        best_match = (tag, similarity * 0.8)
        
        return best_match
    
    def _get_response(self, tag: str) -> str:
        """Get a response for the given intent tag."""
        for intent in self.intents.get('intents', []):
            if intent.get('tag') == tag:
                responses = intent.get('responses', [])
                if responses:
                    return random.choice(responses)
        
        # Default response
        return "I'm not sure about that. Try asking me about cybersecurity topics like phishing, passwords, or malware!"
    
    def chat(self, user_input: str) -> ChatResponse:
        """Process user input and generate a response."""
        user_input = user_input.strip()
        
        if not user_input:
            return ChatResponse(
                intent='empty',
                response="Please type a message! I'm here to help with cybersecurity questions.",
                confidence=1.0
            )
        
        # Predict intent
        intent_tag, confidence = self._predict_intent(user_input)
        
        # Get response
        response = self._get_response(intent_tag)
        
        # Personalize if we know user's name
        if self.user_name and random.random() < 0.3:
            response = f"{self.user_name}, {response[0].lower()}{response[1:]}"
        
        # Store in history
        self.conversation_history.append({
            'user': user_input,
            'bot': response,
            'intent': intent_tag,
            'confidence': confidence
        })
        
        return ChatResponse(
            intent=intent_tag,
            response=response,
            confidence=confidence
        )
    
    def get_suggestions(self) -> List[str]:
        """Get suggested questions for the user."""
        suggestions = [
            "What is phishing?",
            "How to create a strong password?",
            "What is ransomware?",
            "Is public WiFi safe?",
            "What is 2FA?",
            "What to do if hacked?",
            "Tell me about VPNs",
            "What is social engineering?",
            "How to browse safely?",
            "Tell me a joke!"
        ]
        return random.sample(suggestions, min(5, len(suggestions)))
    
    def reset_conversation(self):
        """Reset conversation history."""
        self.conversation_history = []
    
    def get_conversation_summary(self) -> Dict:
        """Get conversation statistics."""
        if not self.conversation_history:
            return {'messages': 0, 'intents': [], 'avg_confidence': 0}
        
        intents = [msg['intent'] for msg in self.conversation_history]
        confidences = [msg['confidence'] for msg in self.conversation_history]
        
        return {
            'messages': len(self.conversation_history),
            'intents': list(set(intents)),
            'avg_confidence': sum(confidences) / len(confidences),
            'most_discussed': max(set(intents), key=intents.count) if intents else None
        }


class QuickResponder:
    """Quick response generator for common queries."""
    
    QUICK_ANSWERS = {
        'is https safe': "HTTPS encrypts data in transit, making it more secure than HTTP. However, malicious sites can also use HTTPS - always verify the full URL!",
        'password length': "Use at least 12-16 characters. Longer = exponentially harder to crack. Try passphrases like 'Purple-Elephant-Runs-Fast-42!'",
        'should i pay ransomware': "NO! Don't pay ransomware. It doesn't guarantee recovery, funds criminals, and marks you as a target. Report to authorities and restore from backups.",
        'vpn necessary': "VPN is highly recommended for public WiFi and privacy-conscious browsing. It encrypts traffic and masks your IP. Essential for sensitive activities.",
        'free wifi safe': "Public/free WiFi is NOT safe for sensitive activities. Use VPN, avoid banking, prefer mobile data for important tasks.",
        'antivirus enough': "Antivirus alone isn't enough. You need: strong passwords + 2FA + updates + safe browsing habits + awareness. It's one tool in your security toolkit."
    }
    
    @classmethod
    def get_quick_answer(cls, query: str) -> Optional[str]:
        """Get quick answer for common queries."""
        query_lower = query.lower()
        for key, answer in cls.QUICK_ANSWERS.items():
            if key in query_lower:
                return answer
        return None


def create_chatbot() -> CyberSecurityChatbot:
    """Create and return a chatbot instance."""
    return CyberSecurityChatbot()


if __name__ == "__main__":
    print("\n" + "="*60)
    print("CYBER-SIGHT ENHANCED AI CHATBOT - TEST MODE")
    print("="*60)
    
    chatbot = CyberSecurityChatbot()
    
    test_queries = [
        "Hello!",
        "How are you doing today?",
        "What's your name?",
        "Tell me a joke",
        "What is phishing?",
        "How do I create a strong password?",
        "Is public wifi safe?",
        "What should I do if I get hacked?",
        "I'm feeling bored",
        "You're awesome!",
        "Thanks for your help!",
        "Goodbye"
    ]
    
    print("\nTesting chatbot responses:")
    print("-"*60)
    
    for query in test_queries:
        response = chatbot.chat(query)
        print(f"\nUser: {query}")
        print(f"Bot [{response.intent}] ({response.confidence:.0%}):")
        print(f"  {response.response[:150]}{'...' if len(response.response) > 150 else ''}")
    
    print("\n" + "-"*60)
    summary = chatbot.get_conversation_summary()
    print(f"Total messages: {summary['messages']}")
    print(f"Avg confidence: {summary['avg_confidence']:.1%}")
