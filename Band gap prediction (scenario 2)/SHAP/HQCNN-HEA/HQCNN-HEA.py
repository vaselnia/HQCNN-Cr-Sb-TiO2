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

# Ensure reproducibility across random number generators and hardware backends
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress verbose Optuna logging outputs
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================
# Data Loading & Physics-Informed Feature Engineering
# ======================================================
dataset = pd.read_csv("dataset_shuffle.csv")

# ---------- Physics-Aware Feature Engineering ----------
# Extract band edge references and Fermi energy level
cbm = dataset["cbm"].values
vbm = dataset["vbm"].values
efermi = dataset["efermi"].values

# dataset["delta_c"] = cbm - efermi          # CBM - EF
# dataset["delta_v"] = efermi - vbm          # EF - VBM
# Calculate band edge asymmetry relative to Fermi level
dataset["asym"] = (cbm + vbm)/2 - efermi   # asymmetry

# ---------- Drop Non-Predictive & Target-Redundant Columns ----------
drop_cols = [
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume",
    "cbm", "vbm", "efermi"
]

# Preserve feature column names for SHAP interpretability
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

# Feature matrix and regression target
X = dataset.drop(drop_cols, axis=1).to_numpy()

y = dataset["band_gap"].to_numpy()

# ======================================================
# Normalization & Dataset Splitting
# ======================================================
# Scale input features to [0, pi] for quantum rotational embedding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target band gaps to [-1, 1] matching Pauli-Z expectation values
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Partition dataset into train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Convert test set to PyTorch tensors for evaluation
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Build training TensorDataset
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

# ======================================================
# Quantum Hardware-Efficient Ansatz (HEA) Layer
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

# Initialize deterministic PennyLane state-vector simulator device
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def enhanced_hea(inputs, scale, shift, rot):
    # Parameterized initial state encoding with learnable scaling and shifting
    for i in range(n_qubits):
        qml.RY(inputs[i] * scale[i] + shift[i], wires=i)

    # Entangling variational layers with all-to-all CNOT topology
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(rot[l, i, 0], wires=i)
            qml.RY(rot[l, i, 1], wires=i)
            qml.RZ(rot[l, i, 2], wires=i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])

    # Measure Pauli-Z expectation value on each wire
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Parameter tensor dimensions for quantum ansatz
weight_shapes = {
    "scale": n_qubits,
    "shift": n_qubits,
    "rot": (n_layers, n_qubits, 3)
}

# Construct differentiable QNode and bind to PyTorch TorchLayer interface
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

    # Hybrid Quantum-Classical architecture for Optuna trial
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
            # Handle batch execution through quantum layer sequentially
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

    # Trial training loop
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()

    # Validation evaluation on held-out test split
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        return loss_fn(preds, y_test_tensor).item()

# Create Optuna study with TPE sampler and median pruner
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print("d = nn.Dropout(dropout)
            self.out = nn.Linear(hidden_units, 1)

        def forward(self, x):
            x = self.a1(self.c1(x))
            # Handle batch execution through quantum layer sequentially
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

    # Trial training loop
    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()

    # Validation evaluation on held-out test split
    model.eval()
    with torch.no_grad():
        preds = model(X_test_tensor)
        return loss_fn(preds, y_test_tensor).item()

# Create Optuna study with TPE sampler and median pruner
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner()
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

best_params = study.best_params
print("), by)
        loss.backward()
        optimizer.step()

# ======================================================
# SHAP Global & Local Interpretability Analysis
# ======================================================
# Prediction wrapper for SHAP KernelExplainer
def model_predict(x_np):
    x_t = torch.tensor(x_np, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(x_t).cpu().numpy().reshape(-1)

# Sample background reference dataset for expectation value calculations
background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

# Compute SHAP values across test samples
explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(X_test, nsamples=200)

shap_values = np.array(shap_values)
assert shap_values.shape == X_test.shape

# Generate and save SHAP summary beeswarm plot
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
# ADDITIONAL SHAP ANALYSIS (DO NOT MODIFY EXISTING CODE)
# ======================================================

# ---------- SHAP Feature Importance Bar Plot (Mean |SHAP|) ----------
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


# ---------- Determine Top Predictive Features ----------
# Rank features by overall mean absolute SHAP attribution
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_k = min(3, len(feature_names))  # Top 3 most influential features
top_features_idx = sorted_idx[:top_k]

print("Top features for dependence plots (Quantum model):")
for rank, idx in enumerate(top_features_idx, 1):
    print(f"{rank}. {feature_names[idx]}")

# ---------- SHAP Dependence Plots with Feature Interactions ----------
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






