import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data/cleaned_data.csv")

# =========================
# Remove Data Leakage
# =========================
leakage_cols = [
    "phq_anxiety1",
    "phq_anxiety2",
    "phq_depression1",
    "phq_depression2",
    "phq_total"
]

df = df.drop(columns=leakage_cols)

# Convert all to numeric
df = df.astype(float)

# =========================
# Features & Target
# =========================
TARGET_COL = "label"

X = df.drop(TARGET_COL, axis=1)
y = df[TARGET_COL]

# =========================
# Train-Test Split (SAME AS TRANSFORMER)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# =========================
# Model
# =========================
# =========================
# Model (NO SMOTE)
# =========================
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.9,
    colsample_bytree=0.9,
    scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train),  # 🔥 imbalance handling
    random_state=42,
    eval_metric="logloss"
)

# =========================
# Train
# =========================
model.fit(X_train, y_train)

# =========================
# Predict
# =========================
y_pred = model.predict(X_test)

# =========================
# Evaluation
# =========================
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))