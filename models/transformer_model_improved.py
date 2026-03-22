import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data/cleaned_data.csv")

# Remove leakage
leakage_cols = [
    "phq_anxiety1",
    "phq_anxiety2",
    "phq_depression1",
    "phq_depression2",
    "phq_total"
]
df = df.drop(columns=leakage_cols)

# Convert to numeric
df = df.astype(float)

# =========================
# Features & Target
# =========================
TARGET_COL = "label"

X = df.drop(TARGET_COL, axis=1).values
y = df[TARGET_COL].values

# 🔥 Normalize (VERY IMPORTANT)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Treat each feature as token
X_train = X_train.unsqueeze(2)
X_test = X_test.unsqueeze(2)

# =========================
# Model
# =========================
class TabTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Linear(1, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x  # NO sigmoid

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TabTransformer().to(device)

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# =========================
# 🔥 Class Weight Fix
# =========================
pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / y_train.sum()]).to(device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# Lower LR
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)

# =========================
# Training
# =========================
epochs = 30

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()
    outputs = model(X_train).squeeze()

    loss = criterion(outputs, y_train)

    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# =========================
# Evaluation
# =========================
model.eval()

with torch.no_grad():
    preds = model(X_test).squeeze()

    preds = torch.sigmoid(preds)  # apply sigmoid here
    preds = (preds > 0.5).float()

accuracy = accuracy_score(y_test.cpu(), preds.cpu())

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test.cpu(), preds.cpu()))

# Save model
torch.save(model.state_dict(), "models/transformer_model_improved.pth")