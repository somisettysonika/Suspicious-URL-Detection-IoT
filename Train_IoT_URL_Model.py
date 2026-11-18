# ==========================================================
#  Enhanced Url_Detection Dashboard
#  Url_Detection Monitoring with Advanced Features
#  Developed by RAJ SHARMA - Enhanced Version
# ==========================================================
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import chi2, mutual_info_classif
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from sklearn.utils import shuffle
import random

# =============================
# PATHS
# =============================
DATA_PATH = r"C:\Users\sonu seethu\OneDrive\Desktop\IOT-SUSPICIOUS-URL-DETECTION-main\dataset\iot_url_dataset.csv"

MODEL_PATH = r"C:\Users\sonu seethu\OneDrive\Desktop\IOT-SUSPICIOUS-URL-DETECTION-main\models\iot_url_model.pkl"
SCALER_PATH = r"C:\Users\sonu seethu\OneDrive\Desktop\IOT-SUSPICIOUS-URL-DETECTION-main\models\iot_url_scaler.pkl"

# =============================
# LOAD DATA
# =============================
df = pd.read_csv(DATA_PATH)
df = shuffle(df)

FEATURES = [
    "length", "num_digits", "num_special_chars",
    "has_ip", "contains_https", "domain_age", "alexa_rank"
]

TARGET = "label"
df[TARGET] = df[TARGET].map({"benign": 0, "suspicious": 1})

X = df[FEATURES]
y = df[TARGET]

print("📌 Dataset Loaded")
print(df.head())

# =============================
# FILTER SELECTION
# =============================
print("\n🎯 Running Filter-based Selection...")

mi_scores = mutual_info_classif(X, y)
chi_scores, _ = chi2(X, y)

filter_scores = mi_scores + chi_scores
selected_filter = [FEATURES[i] for i in np.argsort(filter_scores)[-5:]]

print("🧪 Filter Selected Features:", selected_filter)

# =============================
# WRAPPER SELECTION (RFE)
# =============================
print("\n🔍 Running RFE Wrapper Selection...")

base_model = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(base_model, n_features_to_select=4)
rfe.fit(X[selected_filter], y)

selected_wrapper = list(X[selected_filter].columns[rfe.support_])

print("🧰 RFE Selected Features:", selected_wrapper)

# =============================
# GENETIC ALGORITHM OPTIMIZATION
# =============================
print("\n🧬 Running Genetic Algorithm Optimization...")

POP_SIZE = 12
GENERATIONS = 10
MUT_RATE = 0.2

candidate_features = selected_wrapper
N = len(candidate_features)

def random_chromosome():
    return [random.choice([0,1]) for _ in range(N)]

def fitness(chrom):
    chosen = [candidate_features[i] for i in range(N) if chrom[i] == 1]
    if len(chosen) == 0:
        return 0
    
    X_sel = X[chosen]

    X_train, X_test, y_train, y_test = train_test_split(
        X_sel, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    return accuracy_score(y_test, pred)

population = [random_chromosome() for _ in range(POP_SIZE)]

for gen in range(GENERATIONS):
    scores = [fitness(ch) for ch in population]
    ranked = sorted(list(zip(scores, population)), reverse=True)
    
    population = [p for _, p in ranked[:6]]
    
    while len(population) < POP_SIZE:
        parent1 = random.choice(population[:3])
        parent2 = random.choice(population[:3])
        cut = random.randint(1, N-1)
        child = parent1[:cut] + parent2[cut:]
        
        if random.random() < MUT_RATE:
            mut_idx = random.randint(0, N-1)
            child[mut_idx] = 1 - child[mut_idx]
        
        population.append(child)

best_chrom = ranked[0][1]
final_features = [candidate_features[i] for i in range(N) if best_chrom[i] == 1]

print("\n🏆 GA Selected Features:", final_features)

# =============================
# TRAIN FINAL MODEL
# =============================
print("\n🧠 Training Final Model...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[final_features])

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

final_model = RandomForestClassifier(n_estimators=250, random_state=42)
final_model.fit(X_train, y_train)

pred = final_model.predict(X_test)
acc = accuracy_score(y_test, pred)

print("\n===============================")
print(f"🎯 FINAL ACCURACY: {acc:.2%}")
print("===============================")
print(classification_report(y_test, pred, target_names=["Benign", "Suspicious"]))

joblib.dump(final_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print("\n💾 Model Saved:", MODEL_PATH)
print("💾 Scaler Saved:", SCALER_PATH)
print("🚀 READY for IoT URL Detection Dashboard!")
