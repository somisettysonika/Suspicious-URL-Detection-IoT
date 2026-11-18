# ==========================================================
#  Enhanced Url_Detection Dashboard
#  Url_Detection Monitoring with Advanced Features
#  Developed by RAJ SHARMA - Enhanced Version
# ==========================================================
import pandas as pd
import numpy as np
import random
import string

np.random.seed(42)
random.seed(42)

# ============================
#  CONFIG
# ============================
N_SAMPLES = 1000
OUTPUT_PATH = r"C:\Users\sonu seethu\OneDrive\Desktop\IOT-SUSPICIOUS-URL-DETECTION-main\dataset\iot_url_dataset.csv"

# ============================
#  URL GENERATORS
# ============================

def random_url():
    domain = ''.join(random.choices(string.ascii_lowercase, k=random.randint(6,14)))
    tld = random.choice([".com",".net",".org",".io",".in"])
    path = ''.join(random.choices(string.ascii_lowercase+string.digits, k=random.randint(4,12)))
    return f"http://{domain}{tld}/{path}"

def suspicious_url():
    phishing_keywords = ["login", "secure", "verify", "payment", "update", "bank", "unlock", "gift", "free", "offer"]
    domain = ''.join(random.choices(string.ascii_lowercase+string.digits, k=random.randint(8,18)))
    tld = random.choice([".xyz",".top",".link",".click",".ru",".cn"])
    keyword = random.choice(phishing_keywords)
    random_digits = ''.join(random.choices(string.digits, k=random.randint(2,5)))
    return f"http://{domain}{tld}/{keyword}{random_digits}"

# ============================
#  FEATURE EXTRACTOR
# ============================

def extract_features(url):
    return {
        "length": len(url),
        "num_digits": sum(c.isdigit() for c in url),
        "num_special_chars": sum(c in "-_/.:?&=" for c in url),
        "has_ip": 1 if any(char.isdigit() for char in url.split("/")[2].split(".")) else 0,
        "contains_https": 1 if "https" in url else 0,
        "domain_age": random.randint(1, 15),         # Fake WHOIS
        "alexa_rank": random.randint(5000, 500000)   # Fake Alexa Rank
    }

# ============================
#  DATA BUILDING
# ============================

rows = []

for _ in range(N_SAMPLES):

    if random.random() < 0.5:
        url = random_url()
        label = "benign"
    else:
        url = suspicious_url()
        label = "suspicious"

    feats = extract_features(url)
    feats["url"] = url
    feats["label"] = label

    rows.append(feats)

df = pd.DataFrame(rows)

df.to_csv(OUTPUT_PATH, index=False)

print("=====================================")
print(f"✅ IoT URL Dataset Created: {len(df)} rows")
print("Saved to:", OUTPUT_PATH)
print("=====================================")
print(df.head())
