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
# Reproducibility Setup
# ======================================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================
# Data Loading & Preparation
# ======================================================
dataset = pd.read_csv("dataset_shuffle.csv")

# Metadata and target columns to exclude from features
drop_cols = [
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume"
]

feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

X = dataset.drop(drop_cols, axis=1).to_numpy()
y = dataset["band_gap"].to_numpy()

# ======================================================
# Feature & Target Normalization
# ======================================================
# Scale input features to [0, pi] for quantum angle encoding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target variable to [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Convert test sets to PyTorch tensors
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create training TensorDataset
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

# ======================================================
# Quantum Enhanced HEA Layer Setup
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

# Initialize quantum device
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def enhanced_hea(inputs, scale, shift, rot):
    """
    Enhanced Hardware-Efficient Ansatz (HEA):
    - Scaled and shifted RY feature encoding
    - Parameterized single-qubit rotations (RX, RY, RZ) per layer
    - All-to-all CNOT entangling gates
    """
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

    # Return Pauli-Z expectation values for all qubits
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

weight_shapes = {
    "scale": n_qubits,
    "shift": n_qubits,
    "rot": (n_layers, n_qubits, 3)
}

# Create PennyLane QNode and PyTorch layer interface
qnode = qml.QNode(enhanced_hea, dev, interface="torch", diff_method="backprop")
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ======================================================
# Hyperparameter Optimization (Optuna)
# ======================================================
def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Hybrid QNN architecture for hyperparameter search
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

    # Training loop per trial
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()

    # Evaluate validation loss on test set
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        return loss_fn(preds, y_test_tensor).item()

study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print("Best params:", best_params)

# ======================================================
# Final Model Training
# ======================================================
class HybridNNFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Linear(X.shape[1], n_qubits)
        self.a1 = nn.LeakyReLU(0.1)
        self.q = qlayer
        self.c2 = nn.Linear(n_qubits, best_params["hidden_units"])
        self.a2 = nn.LeakyReLU(0.1)
        self.d = nn.Dropout(best_params["dropout"])
        self.out = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):
        x = self.a1(self.c1(x))
        if x.ndim == 1:
            x = self.q(x)
        else:
            x = torch.stack([self.q(xi) for xi in x])
        x = self.a2(self.c2(x))
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

# Full training loop (100 epochs)
for _ in range(100):
    model.train()
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

# ======================================================
# SHAP Interpretability Analysis
# ======================================================
def model_predict(x_np):
    """Wrapper function to get predictions from PyTorch model as NumPy array."""
    x_t = torch.tensor(x_np, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(x_t).cpu().numpy().reshape(-1)

# Sample background dataset for KernelExplainer
background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(X_test, nsamples=200)

shap_values = np.array(shap_values)
assert shap_values.shape == X_test.shape

# Global summary plot (beeswarm plot)
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

# ======================================================
# Additional SHAP Analysis
# ======================================================

# ---------- SHAP bar plot (mean |SHAP|) ----------
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

# ---------- Determine top features ----------
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_k = min(3, len(feature_names))  # top 3 features (safe)
top_features_idx = sorted_idx[:top_k]

print("Top features for dependence plots (Quantum model):")
for rank, idx in enumerate(top_features_idx, 1):
    print(f"{rank}. {feature_names[idx]}")

# ---------- SHAP dependence plots ----------
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






