# ==========================================================
#  Enhanced Url_Detection Dashboard
#  Url_Detection Monitoring with Advanced Features
#  Developed by RAJ SHARMA - Enhanced Version
# ==========================================================
import os
import sys
import subprocess

print("=" * 70)
print("      IOT SUSPICIOUS URL DETECTION - QUICK START INSTALLER")
print("=" * 70)

# =========================================================
# 1) CHECK DEPENDENCIES
# =========================================================
print("\n📦 Checking dependencies...")

required_packages = [
    "pandas", "numpy", "scikit-learn", "joblib", "matplotlib",
    "seaborn", "pyttsx3", "SpeechRecognition", "requests", "python-whois"
]

missing = []

for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
    except ImportError:
        missing.append(pkg)

if missing:
    print(f"❌ Missing packages detected: {missing}")
    print("📥 Installing now...\n")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
    print("✅ All missing packages have been installed!")
else:
    print("✅ All required dependencies already installed!")

# =========================================================
# 2) GENERATE IOT URL DATASET
# =========================================================
print("\n📊 Generating IoT Suspicious URL Dataset...")

GEN_SCRIPT = r"C:\Users\sonu seethu\OneDrive\Desktop\IOT-SUSPICIOUS-URL-DETECTION-main\generate_iot_url_dataset.py"

if not os.path.exists(GEN_SCRIPT):
    print("\n❌ ERROR: Dataset generator script not found:")
    print(GEN_SCRIPT)
    sys.exit(1)

exec(open(GEN_SCRIPT, encoding="utf-8").read())     

print("📁 Dataset generated successfully!")

# =========================================================
# 3) TRAIN MODEL USING HYBRID FEATURE SELECTION (FILTER + RFE + GA)
# =========================================================
print("\n🧠 Training IoT URL Model using Hybrid Feature Selection...")

TRAIN_SCRIPT = r"C:\Users\sonu seethu\OneDrive\Desktop\IOT-SUSPICIOUS-URL-DETECTION-main\Train_IoT_URL_Model.py"

if not os.path.exists(TRAIN_SCRIPT):
    print("\n❌ ERROR: Training script not found:")
    print(TRAIN_SCRIPT)
    sys.exit(1)

exec(open(TRAIN_SCRIPT, encoding="utf-8").read())

# =========================================================
# 4) FINISHED – PRINT NEXT STEPS
# =========================================================
print("\n" + "=" * 70)
print("           🚀 SETUP COMPLETE – SYSTEM READY")
print("=" * 70)

print("""
📌 NEXT STEPS:

1️⃣  Open your final GUI file:
    💡 SuspiciousURL_Dashboard.py  (or the 3500-line file ChatGPT generated)

2️⃣  In the GUI:
       • Click:  📁 Load Model
         Load → iot_url_model.pkl
       • Click:  📁 Load Scaler
         Load → iot_url_scaler.pkl

3️⃣  Enter URL → Click “Extract Features”
4️⃣  Click “Predict URL Safety”
5️⃣  Enjoy the complete IoT Suspicious URL Detection dashboard! 🎯

""")

print("=" * 70)
print("   DONE ✔ Your IoT URL Detection environment is fully prepared!")
print("=" * 70)
