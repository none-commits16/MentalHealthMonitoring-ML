import pandas as pd

df = pd.read_csv("data/final_dataset.csv")

print("Columns:\n", df.columns)

# Check leakage
leakage_cols = [
    "phq_anxiety1",
    "phq_anxiety2",
    "phq_depression1",
    "phq_depression2",
    "phq_total"
]

print("\nLeakage Check:")
for col in leakage_cols:
    if col in df.columns:
        print(f"❌ Leakage column found: {col}")
    else:
        print(f"✅ {col} not present")

# Data types
print("\nData Types:\n", df.dtypes)

# Shape
print("\nShape:", df.shape)

# Label distribution
print("\nLabel Distribution:")
print(df["label"].value_counts())