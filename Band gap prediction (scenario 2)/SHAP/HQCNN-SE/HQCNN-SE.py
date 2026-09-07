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
# Reproducibility & Deterministic Setup
# ======================================================
SEED = 42

# Enforce deterministic behavior across all RNG backends
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================
# Data Loading & Physics-Informed Feature Engineering
# ======================================================
dataset = pd.read_csv("dataset_shuffle.csv")

# ---------- Physics-Aware Feature Engineering ----------
# Extract band edge extrema and Fermi energy level
cbm = dataset["cbm"].values
vbm = dataset["vbm"].values
efermi = dataset["efermi"].values

# dataset["delta_c"] = cbm - efermi          # CBM - EF
# dataset["delta_v"] = efermi - vbm          # EF - VBM
# Compute band edge asymmetry relative to the Fermi level
dataset["asym"] = (cbm + vbm)/2 - efermi   # asymmetry

# ---------- Drop Redundant & Non-Predictive Columns ----------
drop_cols = [
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume",
    "cbm", "vbm", "efermi"
]

# Preserve feature column names for SHAP interpretability analysis
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

# Extract feature matrix and continuous regression targets
X = dataset.drop(drop_cols, axis=1).to_numpy()

y = dataset["band_gap"].to_numpy()

# ======================================================
# Normalization & Dataset Splitting
# ======================================================
# Scale input features to [0, pi] for quantum rotational angle embedding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target band gaps to [-1, 1] aligned with Pauli-Z expectation values
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Partition dataset into reproducible train (80%) and test (20%) subsets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Construct PyTorch TensorDataset for training
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

# ======================================================
# Quantum Layer (Strongly Entangling Ansatz)
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

# Initialize deterministic PennyLane default.qubit simulator
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def quantum_layer(inputs, weights):
    # Rotational angle embedding for input vectors
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # Multi-layer parameterized strongly entangling circuit
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measure expectation values of Pauli-Z operator on each qubit
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Parameter tensor shape definition for variational weights
weight_shapes = {"weights": (n_layers, n_qubits, 3)}
# Construct differentiable PyTorch quantum layer via TorchLayer interface
qlayer = qml.qnn.TorchLayer(qml.QNode(quantum_layer, dev), weight_shapes)

# ======================================================
# Hyperparameter Optimization (Optuna)
# ======================================================
optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    # Categorical search spaces for hyperparameters
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Hybrid Quantum-Classical architecture for Optuna trial evaluation
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
            x = self.q(x)
            x = self.a2(self.c2(x))
            x = self.d(x)
            return self.out(x).view(-1)  # <<< FIX

    model = HybridNNOptuna()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda wid: np.random.seed(SEED + wid)
    )

    # Mini-batch training loop per trial
    for _ in range(epochs):
        for bx, by in loader:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            opt.step()

    # Compute validation loss on the test set
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32))
        return loss_fn(preds, torch.tensor(y_test, dtype=torch.float32)).item()

# Initialize Optuna study with TPE sampler and median pruner
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
# Instantiate hybrid model using optimal hyperparameters found by Optuna
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
        x = self.q(x)
        x = self.a2(self.c2(x))
        x = self.d(x)
        return self.out(x).view(-1)  # <<< FIX

model = HybridNNFinal()
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
criterion = nn.MSELoss()

loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    worker_init_fn=lambda wid: np.random.seed(SEED + wid)
)

# Full training cycle of optimized model for 100 epochs
for _ in range(100):
    for bx, by in loader:
        optimizer.zero_grad()
        loss = criterion(model(bx), by)
        loss.backward()
        optimizer.step()

# ======================================================
# SHAP (GUARANTEED STABLE)
# ======================================================
# Model prediction wrapper for SHAP KernelExplainer
def model_predict(x_np):
    x_t = torch.tensor(x_np, dtype=torch.float32)
    with torch.no_grad():
        return model(x_t).cpu().numpy().reshape(-1)

# Sample background reference dataset for expectation value calculations
background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

# Calculate SHAP values across test samples
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
plt.savefig("SHAP_summary_all_features_quantum.png", dpi=300, bbox_inches="tight")
plt.close()

print("✅ SHAP summary plot saved for all features (Quantum model).")



# ======================================================
# ADDITIONAL SHAP ANALYSIS (DO NOT MODIFY EXISTING CODE)
# ======================================================

# ---------- SHAP bar plot (mean |SHAP|) ----------
# Feature importance ranking based on mean absolute SHAP values
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
# Rank features by overall mean absolute attribution
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_k = min(3, len(feature_names))  # top 3 features (safe)
top_features_idx = sorted_idx[:top_k]

print("Top features for dependence plots (Quantum model):")
for rank, idx in enumerate(top_features_idx, 1):
    print(f"{rank}. {feature_names[idx]}")

# ---------- SHAP dependence plots ----------
# Generate and save SHAP dependence plots for top-k features with auto-interaction
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





