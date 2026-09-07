import os
import random
import numpy as np
import torch
import pennylane as qml
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
# Deterministic Seeding & Reproducibility Setup
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

optuna.logging.set_verbosity(optuna.logging.WARNING)

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

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

# ======================================================
# Quantum Hardware-Efficient Ansatz (HEA) Layer Setup
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def enhanced_hea(inputs, scale, shift, rot):
    for i in range(n_qubits):
        qml.RY(inputs[i] * scale[i] + shift[i], wires=i)

    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(rot[l, i, 0], wires=i)
            qml.RY(rot[l, i, 1], wires=i)
            qml.RZ(rot[l, i, 2], wires=i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])

    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

weight_shapes = {
    "scale": n_qubits,
    "shift": n_qubits,
    "rot": (n_layers, n_qubits, 3)
}

qnode = qml.QNode(enhanced_hea, dev, interface="torch", diff_method="backprop")
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ======================================================
# Optuna Hyperparameter Optimization
# ======================================================
def objective(trial):
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Linear(X.shape[1], n_qubits)
            self.a1 = nn.LeakyReLU(0.1)
            self.q = qlayer
            self.c2 = nn.Linear(n_qubits, hidden_units)
            self.a2 = nn.LeakyReLU(0.1)
            self.d = nn.Dropout(dropout)
            self.out = nn.Linear(hidden_units, 1)

        def forward(self, x):
            x = self.a1(self.c1(x))
            if x.ndim == 1:
                x = self.q(x)
            else:
                x = torch.stack([self.q(xi) for xi in x])
            x = self.a2(self.c2(x))
            x = self.d(x)
            return self.out(x).view(-1)

    model = HybridNNOptuna()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=Truedropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()
            self.c1 = nn.Linear(X.shape[1], n_qubits)
            self.a1 = nn.LeakyReLU(0.1)
            self.q = qlayer
            self.c2 = nn.Linear(n_qubits, hidden_units)
            self.a2 = nn.LeakyReLU(0.1)
            self.d = nn.Dropout(dropout)
            self.out = nn.Linear(hidden_units, 1)

        def forward(self, x):
            x = self.a1(self.c1(x))
            if x.ndim == 1:
                x = self.q(x)
            else:
                x = torch.stack([self.q(xi) for xi in x])
            x = self.a2(self.c2(x))
            x = self.d(x)
            return self.out(x).view(-1)

    model = HybridNNOptuna()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda wid: np.random.seed(SEED + wid)
    )

    for _ in range(epochs):
        model.train()
        for bx, by in loader(x))
        x = self.d(x)
        return self.out(x).view(-1)

model = HybridNNFinal()
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
criterion = nn.MSELoss()

loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    worker_init_fn=lambda wid: np.random.seed(SEED + wid)
)

for _ in range(100):
    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

# ======================================================
# Model Evaluation
# ======================================================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(X_test_tensor).cpu().numpy()

y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n===== TEST EVALUATION METRICS =====")
print(f"MSE: {mse:.5f}")
print(f"MAE: {mae:.5f}")
print(f"R² : {r2:.5f}")
print("====================================")

# ======================================================
# SHAP Interpretability Analysis
# ======================================================
def model_predict(x_np):
    x_t = torch.tensor(x_np, dtype=torch.float32)
    model.eval()
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
plt.savefig("SHAP_summary_HEA_Enhanced.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ SHAP summary plot saved (Enhanced HEA).")

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
plt.savefig("SHAP_bar_mean_abs_quantum.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ SHAP bar plot (mean |SHAP|) saved (Quantum model).")

# ------------------------------------------------------
# SHAP Feature Dependence Plots (Top 3 Features)
# ------------------------------------------------------
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_k = min(3, len(feature_names))
top_features_idx = sorted_idx[:top_k]

print("\nTop features identified for dependence plots (Quantum model):")
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
    fname = f"SHAP_dependence_quantum_{feature_names[idx]}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP dependence saved for: {feature_names[idx]} (Quantum model)")





