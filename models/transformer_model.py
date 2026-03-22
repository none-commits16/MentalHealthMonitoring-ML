import torch
import torch.nn as nn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# =========================
# Load Dataset
# =========================
df = pd.read_csv("data/cleaned_data.csv")

# =========================
# REMOVE DATA LEAKAGE
# =========================
leakage_cols = [
    "phq_anxiety1",
    "phq_anxiety2",
    "phq_depression1",
    "phq_depression2",
    "phq_total"
]

df = df.drop(columns=leakage_cols)

# =========================
# 🔥 FIX: Convert all to numeric
# =========================
df = df.astype(float)

# =========================
# Features & Target
# =========================
TARGET_COL = "label"

X = df.drop(TARGET_COL, axis=1).values
y = df[TARGET_COL].values

# =========================
# Train-Test Split
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# =========================
# Convert to Tensors
# =========================
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Treat each feature as token
X_train = X_train.unsqueeze(2)   # (batch, features, 1)
X_test = X_test.unsqueeze(2)

# =========================
# Transformer Model
# =========================
class TabTransformer(nn.Module):
    def __init__(self, num_features):
        super().__init__()

        self.embedding = nn.Linear(1, 64)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.fc = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return self.sigmoid(x)

# =========================
# Device (GPU if available)
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TabTransformer(num_features=X_train.shape[1]).to(device)

X_train, X_test = X_train.to(device), X_test.to(device)
y_train, y_test = y_train.to(device), y_test.to(device)

# =========================
# Training Setup
# =========================
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# =========================
# Training Loop
# =========================
epochs = 20

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
    preds = (preds > 0.5).float()

accuracy = accuracy_score(y_test.cpu(), preds.cpu())

print("\nAccuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test.cpu(), preds.cpu()))

# =========================
# Save Model
# =========================
torch.save(model.state_dict(), "models/transformer_model.pth")

print("\nModel saved at models/transformer_model.pth")