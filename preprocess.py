import pandas as pd

data = pd.read_csv("data/responses.csv")

data.columns = [
    "timestamp",
    "age",
    "screen_time",
    "social_media",
    "productivity",
    "sleep_hours",
    "living",
    "meditation",
    "academic_pressure",
    "phq_anxiety1",
    "phq_anxiety2",
    "phq_depression1",
    "phq_depression2"
]
print("\nNew Column Names:")
print(data.columns)


print("First 5 rows:")
print(data.head())

print("\nShape of dataset:")
print(data.shape)

data = data.drop("timestamp", axis=1)

print("\nColumns After Dropping Timestamp:")
print(data.columns)

print("\nNew Shape:")
print(data.shape)

print("\nMissing Values Per Column:")
print(data.isnull().sum())

data = data.dropna()

print("\nNew Shape After Dropping Missing Values:")
print(data.shape)

print("\nData Types:")
print(data.dtypes)

print("\nUnique values in screen_time:")
print(data["screen_time"].unique())

print("\nUnique values in social_media:")
print(data["social_media"].unique())

print("\nUnique values in sleep_hours:")
print(data["sleep_hours"].unique())

print("\nUnique values in productivity:")
print(data["productivity"].unique())

print("\nUnique values in meditation:")
print(data["meditation"].unique())

print("\nUnique values in living:")
print(data["living"].unique())

data["screen_time"] = data["screen_time"].replace("10+", "10")
data["screen_time"] = data["screen_time"].astype(int)

print("\nScreen time data type after fix:")
print(data["screen_time"].dtype)

data["social_media"] = data["social_media"].replace("6+", "6")
data["social_media"] = data["social_media"].replace("Less than 2", "1")

data["social_media"] = data["social_media"].astype(int)

print("\nSocial media data type after fix:")
print(data["social_media"].dtype)

data["sleep_hours"] = data["sleep_hours"].replace("10+", "10")
data["sleep_hours"] = data["sleep_hours"].astype(int)

print("\nSleep hours data type after fix:")
print(data["sleep_hours"].dtype)

data = pd.get_dummies(
    data,
    columns=["living", "meditation", "productivity"],
    drop_first=True
)

print("\nColumns After Encoding:")
print(data.columns)

print("\nNew Shape After Encoding:")
print(data.shape)

data["phq_total"] = (
    data["phq_anxiety1"] +
    data["phq_anxiety2"] +
    data["phq_depression1"] +
    data["phq_depression2"]
)

print("\nPHQ Total Score Preview:")
print(data["phq_total"].head())

print("\nPHQ Severity Distribution:")
print(pd.cut(
    data["phq_total"],
    bins=[-1,2,5,8,12],
    labels=["Normal","Mild","Moderate","Severe"]
).value_counts())

data["label"] = data["phq_total"].apply(lambda x: 1 if x >= 6 else 0)

print("\nNew Label Distribution:")
print(data["label"].value_counts())

X = data.drop([
    "phq_anxiety1",
    "phq_anxiety2",
    "phq_depression1",
    "phq_depression2",
    "phq_total",
    "label"
], axis=1)

y = data["label"]

print("\nFeature shape:", X.shape)
print("Label shape:", y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining shape:", X_train.shape)
print("Testing shape:", X_test.shape)

print("\nTraining label distribution:")
print(y_train.value_counts())

print("\nTesting label distribution:")
print(y_test.value_counts())

data.to_csv("data/cleaned_data.csv", index=False)

print("\nCleaned dataset saved successfully.")
