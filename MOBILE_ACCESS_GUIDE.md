# 📱 How to Use Cyber-Sight on Mobile & Desktop

## 🖥️ DESKTOP - Easy Launch (No Terminal Needed!)

### Option 1: Double-Click to Start
1. Go to `D:\pawan project\cyber_sight\`
2. Double-click **`START_CYBER_SIGHT.bat`**
3. Wait for server to start
4. Open browser: **http://localhost:8501**

### Option 2: Create Desktop Shortcut
1. Right-click on `START_CYBER_SIGHT.bat`
2. Select "Create shortcut"
3. Move shortcut to Desktop
4. Rename it to "Cyber-Sight"
5. (Optional) Right-click > Properties > Change Icon

---

## 📱 MOBILE - Access Like an App

### Step 1: Find Your Computer's IP
When you run the server, it shows:
```
Network URL: http://10.69.44.195:8501
```
This IP may change. Check it each time you start.

### Step 2: Connect Mobile to Same WiFi
Your phone and computer MUST be on the **same WiFi network**.

### Step 3: Open in Mobile Browser
Open Chrome/Safari on your phone and go to:
```
http://YOUR_COMPUTER_IP:8501
```
Example: `http://10.69.44.195:8501`

### Step 4: Add to Home Screen (Like an App!)

#### For Android (Chrome):
1. Open the URL in Chrome
2. Tap the **3 dots menu** (⋮) in top-right
3. Tap **"Add to Home screen"**
4. Name it "Cyber-Sight"
5. Tap "Add"
Now it appears like an app on your home screen!

#### For iPhone (Safari):
1. Open the URL in Safari
2. Tap the **Share button** (□↑)
3. Scroll down and tap **"Add to Home Screen"**
4. Name it "Cyber-Sight"
5. Tap "Add"
Now it appears like an app!

---

## ⚠️ Important Notes

1. **Server Must Be Running**: Your computer must be ON and the server running for mobile access.

2. **Same Network Required**: Phone and computer must be on the same WiFi.

3. **IP Address Changes**: Your Network IP may change. Check the terminal output each time.

4. **Firewall**: If mobile can't connect, your Windows Firewall might be blocking it.
   - Go to Windows Defender Firewall
   - Allow "Python" through the firewall

---

## 🚀 Quick Start Guide

1. **Double-click** `START_CYBER_SIGHT.bat` on your computer
2. **Note the Network URL** shown (e.g., http://10.69.44.195:8501)
3. **Open that URL** on your mobile browser
4. **Add to Home Screen** for app-like experience!

---

## 🛑 To Stop the Server

- Double-click **`STOP_SERVER.bat`**
- Or close the command window

---

## 💡 Want It Always Running?

For 24/7 access, you'd need to:
1. Deploy to cloud (Streamlit Cloud, Heroku, AWS)
2. Or keep your computer always on

Contact your developer for cloud deployment options!
