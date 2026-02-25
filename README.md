# 🔒 Cyber-Sight: Global ML & AI Based Cyber Crime Detection and Safety Platform

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/License-Educational-green.svg)

<p align="center">
  <img src="https://img.icons8.com/fluency/96/security-checked.png" alt="Cyber-Sight Logo" width="120">
</p>

## 📋 Overview

**Cyber-Sight** is a comprehensive, ML-powered web application designed for global cyber crime detection and security awareness. Built with Python and Streamlit, it provides real-time threat analysis, URL safety checking, and an AI-powered cybersecurity chatbot.

### ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Threat Detection** | ML-based classification of cyber threats (phishing, malware, hacking) |
| 🔗 **URL Safety Checker** | Comprehensive URL analysis with heuristic + ML hybrid approach |
| 🤖 **AI Chatbot** | NLP-powered cybersecurity Q&A assistant |
| 📊 **Dataset Insights** | Interactive visualizations of global cyber threat data |
| 🌍 **Global Coverage** | Country-agnostic design for worldwide threat detection |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### Installation

1. **Clone or Download the Project**

```bash
cd "d:\pawan project\cyber_sight"
```

2. **Create Virtual Environment (Recommended)**

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

4. **Download NLTK Data (First time only)**

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

5. **Train the ML Model**

```bash
python model/train_model.py
```

6. **Run the Application**

```bash
streamlit run app.py
```

7. **Open in Browser**

Navigate to `http://localhost:8501`

---

## 📁 Project Structure

```
cyber_sight/
│
├── app.py                    # Main Streamlit web application
│
├── model/
│   ├── __init__.py
│   ├── train_model.py        # ML model training script
│   └── threat_model.pkl      # Trained model (generated after training)
│
├── chatbot/
│   ├── __init__.py
│   ├── chatbot.py            # AI chatbot logic
│   └── intents.json          # Chatbot knowledge base
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── url_checker.py        # URL safety analysis module
│
├── data/
│   └── cybercrime_dataset.csv # Training dataset
│
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

---

## 🔧 Configuration

### Model Training Options

Edit `model/train_model.py` to customize:

```python
# Change test/train split ratio
train_test_split(X, y, test_size=0.2)  # Default: 20% test

# Modify model parameters
RandomForestClassifier(
    n_estimators=100,    # Number of trees
    max_depth=10,        # Maximum tree depth
    random_state=42      # For reproducibility
)
```

### Adding Custom Intents (Chatbot)

Edit `chatbot/intents.json`:

```json
{
  "tag": "new_topic",
  "patterns": ["question 1", "question 2"],
  "responses": ["Answer to the questions"]
}
```

---

## 📖 Usage Guide

### 1. 🎯 Cyber Threat Detection

- Navigate to "Threat Detection" from the sidebar
- Enter a URL or provide threat information
- View ML-based classification results
- Get risk level and recommendations

### 2. 🔗 URL Safety Checker

- Go to "URL Checker" tab
- Enter single URL or multiple URLs (batch mode)
- Review comprehensive safety analysis
- Check detailed feature breakdown

### 3. 🤖 AI Chatbot

- Select "AI Chatbot" from navigation
- Ask cybersecurity questions in natural language
- Topics include: phishing, malware, passwords, VPN, etc.
- Download chat history for reference

### 4. 📊 Dataset Insights

- Explore "Dataset Insights" tab
- View attack type distribution
- Analyze geographic patterns
- Study risk level statistics

---

## 🔒 Security Features

### URL Analysis Checks

| Check | Description |
|-------|-------------|
| HTTPS Validation | Verifies secure protocol usage |
| IP Detection | Flags direct IP address URLs |
| Domain Analysis | Checks length and patterns |
| Keyword Scanning | Detects suspicious terms |
| TLD Verification | Identifies risky top-level domains |
| Brand Impersonation | Detects fake brand URLs |

### Threat Classification

- **Safe** - Legitimate, trusted websites
- **Phishing** - Credential theft attempts
- **Malware** - Malicious software distribution
- **Hacking** - Attack tools and exploits

---

## 📊 ML Model Details

### Training Dataset

The model is trained on URL-based features:

| Feature | Description |
|---------|-------------|
| `domain_length` | Length of domain name |
| `has_https` | HTTPS protocol usage |
| `has_ip` | Direct IP in URL |
| `num_dots` | Count of dots |
| `num_hyphens` | Count of hyphens |
| `num_slashes` | Count of slashes |
| `num_digits` | Count of digits |
| `url_length` | Total URL length |
| `has_suspicious_keywords` | Presence of phishing terms |

### Models Used

1. **Random Forest Classifier** - Primary threat detection
2. **Gradient Boosting** - Alternative classifier
3. **Logistic Regression** - Baseline model

### Performance Metrics

After training, check `model/training_report.txt` for:
- Accuracy scores
- F1 scores
- Cross-validation results
- Feature importance rankings

---

## 🌍 Global Coverage

Cyber-Sight is designed for worldwide use:

- ✅ Country-agnostic threat detection
- ✅ Supports URLs from any region
- ✅ Multi-language URL patterns
- ✅ Global cyber threat statistics
- ✅ International reporting resources

---

## ⚠️ Disclaimer

> **IMPORTANT**: This application is for **educational and awareness purposes only**.
> 
> - Does NOT perform actual hacking
> - Does NOT store personal data
> - For detection and analysis only
> - Not a replacement for professional security tools

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📝 License

This project is released for educational purposes. Please use responsibly.

---

## 📞 Support

For issues and questions:
- Check the FAQ in the chatbot
- Review error messages in console
- Ensure all dependencies are installed
- Verify model is trained

---

## 🎓 Learning Resources

The chatbot covers these cybersecurity topics:

- What is hacking?
- How phishing works
- Password security best practices
- Two-factor authentication
- VPN usage and benefits
- Social engineering attacks
- Data breach response
- Mobile device security
- Safe browsing tips
- Ransomware protection

---

<p align="center">
  <b>🔒 Stay Safe Online with Cyber-Sight 🔒</b>
  <br>
  <i>Global Cyber Crime Detection and Safety Platform</i>
</p>
