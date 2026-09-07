import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import optuna
import shap

# ======================================================
# Full Deterministic Reproducibility Setup
# ======================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================
# Data Ingestion & Target Splitting
# ======================================================
dataset = pd.read_csv("dataset_shuffle.csv")

drop_cols = ["Structure", "Time (h)", "O", "L", "a", "b", "x", "Reference"]
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

X = dataset.drop(drop_cols, axis=1).to_numpy()
y = dataset["x"].to_numpy()

# ======================================================
# Feature & Target Normalization
# ======================================================
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

# ======================================================
# Classical Quantum-like Layer Architecture
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

class ClassicalQuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(n_qubits, n_qubits))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

clayer = ClassicalQuantumLayer(n_qubits, n_layers)

# ======================================================
# Optuna Hyperparameter Optimization
# ======================================================
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)

def objective(trial):
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical(
        "hidden_units", [8, 9, 10, 11, 12, 13, 14, 15, 16, 32]
    )
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Linear(X.shape[1], n_qubits)
            self32)
)

# ======================================================
# Classical Quantum-like Layer Architecture
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

class ClassicalQuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(n_qubits, n_qubits))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

clayer = ClassicalQuantumLayer(n_qubits, n_layers)

# ======================================================
# Optuna Hyperparameter Optimization
# ======================================================
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize", sampler=sampler)

def objective(trial):
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical(
        "hidden_units", [8, 9, 10, 11, 12, 13, 14, 15, 16, 32]
    )
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Linear(X.shape[1], n_qubits)
            selfq = clayer
        self.c2 = nn.Linear(n_qubits, best_params["hidden_units"])
        self.a2 = nn.LeakyReLU(0.1)
        self.d = nn.Dropout(best_params["dropout"])
        self.out = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):
        x = self.a1(self.c1(x))
        x = self.q(x)
        x = self.a2(self.c2(x))
        x = self.d(x)
        return self.out(x).view(-1)

model = HybridNNFinal()
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
criterion = nn.MSELoss()

loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True
)

for _ in range(100):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

# ======================================================
# SHAP Interpretability Analysis
# ======================================================
def model_predict(x_np):
    x_t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        return model(x_t).cpu().numpy().reshape(-1)

background = X_train[np.random.choice(X_train.shape[0], 30, replace=False)]

explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(X_test, nsamples=200)

shap_values = np.array(shap_values)
assert shap_values.shape == X_test.shape

# ------------------------------------------------------
# SHAP Summary Beeswarm Plot (All Features)
# ------------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)
plt.tight_layout()
plt.savefig("SHAP_summary_all_features.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ SHAP summary plot saved for all features.")

# ------------------------------------------------------
# SHAP Mean Absolute Value Bar Plot
# ------------------------------------------------------
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)
plt.tight_layout()
plt.savefig("SHAP_bar_mean_abs.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ SHAP bar plot (mean |SHAP|) saved.")

# ------------------------------------------------------
# SHAP Feature Dependence Plots (Top 3 Features)
# ------------------------------------------------------
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_k = min(3, len(feature_names))
top_features_idx = sorted_idx[:top_k]

print("\nTop features identified for dependence plots:")
for rank, idx in enumerate(top_features_idx, 1):
    print(f"{rank}. {feature_names[idx]}")

for idx in top_features_idx:
    plt.figure(figsize=(7, 5))
    shap.dependence_plot(
        idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
        interaction_index="auto"
    )
    plt.tight_layout()
    fname = f"SHAP_dependence_{feature_names[idx]}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP dependence saved for: {feature_names[idx]}")






