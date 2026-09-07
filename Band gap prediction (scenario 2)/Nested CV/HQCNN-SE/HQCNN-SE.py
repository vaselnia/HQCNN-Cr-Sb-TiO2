import os
import random
import numpy as np
import torch
import pennylane as qml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import optuna

# ======================================================
# 🔒 Global Deterministic Reproducibility Setup
# ======================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# Seed Python, NumPy, and PyTorch random number generators
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enforce deterministic algorithm execution across CPU and CUDA backends
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress verbose Optuna trial logging for clean terminal output
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------
# Data Loading & Physics-Aware Feature Engineering
# ------------------------------------------------------
dataset = pd.read_csv("dataset_shuffle.csv")

# Extract electronic band structure reference levels
cbm = dataset["cbm"].values
vbm = dataset["vbm"].values
efermi = dataset["efermi"].values

# Calculate band edge asymmetry relative to the Fermi energy level
# dataset["delta_c"] = cbm - efermi          # CBM - EF
# dataset["delta_v"] = efermi - vbm          # EF - VBM
dataset["asym"] = (cbm + vbm) / 2.0 - efermi  # Band edge asymmetry feature

# Drop non-predictive identifiers, metadata, and raw target-redundant band extrema
X = dataset.drop([
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume",
    "cbm", "vbm", "efermi"
], axis=1).to_numpy()

y = dataset["band_gap"].to_numpy()

# ======================================================
# Feature & Target Normalization
# ======================================================
# Scale input features to [0, pi] for quantum rotational angle embedding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X = scaler_X.fit_transform(X)

# Scale regression target to [-1, 1] matching Pauli-Z expectation value spectrum
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ======================================================
# Quantum Layer (Strongly Entangling Ansatz)
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

# Initialize deterministic PennyLane default qubit simulator device
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def quantum_layer(inputs, weights):
    # Encode classical inputs into quantum state via angle rotation
    qml.AngleEmbedding(inputs, wires=range(n_qubits))
    # Apply parameterized strongly entangling variational layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
    # Measure expectation values of Pauli-Z observable across all qubits
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Shape of variational parameters: (n_layers, n_qubits, 3 Euler angles)
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# ======================================================
# 🔵 Nested Cross-Validation (5 Outer × 3 Inner Folds)
# ======================================================
outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

# Metric tracking lists across outer cross-validation folds
outer_mse, outer_mae, outer_r2 = [], [], []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"\n🔁 Outer Fold {outer_fold + 1}/5")

    # Partition outer training and testing sets
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    train_outer_ds = TensorDataset(
        torch.tensor(X_train_outer, dtype=torch.float32),
        torch.tensor(y_train_outer, dtype=torch.float32)
    )

    # ==================================================
    # 🔵 Inner CV + Optuna Hyperparameter Optimization
    # ==================================================
    def objective(trial):
        # Define hyperparameter search space
        lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
        hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_categorical("epochs", [15, 20])

        fold_losses = []

        # Inner 3-fold cross-validation loop
        for tr_idx, val_idx in inner_cv.split(X_train_outer):
            X_tr, X_val = X_train_outer[tr_idx], X_train_outer[val_idx]
            y_tr, y_val = y_train_outer[tr_idx], y_train_outer[val_idx]

            tr_ds = TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32)
            )

            # Instantiate quantum TorchLayer interface for inner trial
            qlayer = qml.qnn.TorchLayer(
                qml.QNode(quantum_layer, dev),
                weight_shapes
            )

            # Define hybrid quantum-classical network architecture
            class HybridQNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(n_qubits, n_qubits)
                    self.act1 = nn.LeakyReLU(0.1)
                    self.q = qlayer
                    self.fc2 = nn.Linear(n_qubits, hidden_units)
                    self.act2 = nn.LeakyReLU(0.1)
                    self.drop = nn.Dropout(dropout)
                    self.out = nn.Linear(hidden_units, 1)

                def forward(self, x):
                    x = self.act1(self.fc1(x))
                    x = self.q(x)
                    x = self.act2(self.fc2(x))
                    x = self.drop(x)
                    return self.out(x).squeeze()

            model = HybridQNN()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            loader = DataLoader(
                tr_ds,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=lambda wid: np.random.seed(SEED + wid)
            )

            # Inner fold training loop
            for _ in range(epochs):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            # Inner fold validation evaluation
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32))
                fold_losses.append(
                    criterion(preds, torch.tensor(y_val, dtype=torch.float32)).item()
                )

        # Return mean validation loss across inner folds
        return np.mean(fold_losses)

    # Execute hyperparameter optimization study using TPE sampler
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best = study.best_params

    # ==================================================
    # 🔵 Retrain Final Model on Full Outer-Train Fold
    # ==================================================
    # Instantiate dedicated quantum layer for outer fold training
    qlayer_final = qml.qnn.TorchLayer(
        qml.QNode(quantum_layer, dev),
        weight_shapes
    )

    # Define final hybrid architecture using optimal hyperparameters
    class FinalQNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_qubits, n_qubits)
            self.act1 = nn.LeakyReLU(0.1)
            self.q = qlayer_final
            self.fc2 = nn.Linear(n_qubits, best["hidden_units"])
            self.act2 = nn.LeakyReLU(0.1)
            self.drop = nn.Dropout(best["dropout"])
            self.out = nn.Linear(best["hidden_units"], 1)

        def forward(self, x):
            x = self.act1(self.fc1(x))
            x = self.q(x)
            x = self.act2(self.fc2(x))
            x = self.drop(x)
            return self.out(x).squeeze()

    model = FinalQNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best["lr"])
    criterion = nn.MSELoss()

    loader = DataLoader(
        train_outer_ds,
        batch_size=best["batch_size"],
        shuffle=True,
        worker_init_fn=lambda wid: np.random.seed(SEED + wid)
    )

    # Train final model on full outer training partition (100 epochs)
    for _ in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # ==================================================
    # 🔵 Outer Fold Generalization Evaluation
    # ==================================================
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_outer, dtype=torch.float32)).numpy()

    # Invert target scaling back to physical band gap units (eV)
    y_true = scaler_y.inverse_transform(y_test_outer.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()

    # Compute and record evaluation metrics for the current outer fold
    outer_mse.append(mean_squared_error(y_true, y_pred))
    outer_mae.append(mean_absolute_error(y_true, y_pred))
    outer_r2.append(r2_score(y_true, y_pred))

# ======================================================
# 🔴 Final Nested Cross-Validation Summary Report
# ======================================================
print("\n===== NESTED 5×3 CV (HYBRID QNN) =====")
print(f"MSE: {np.mean(outer_mse):.5f} ± {np.std(outer_mse):.5f}")
print(f"MAE: {np.mean(outer_mae):.5f} ± {np.std(outer_mae):.5f}")
print(f"R² : {np.mean(outer_r2):.5f} ± {np.std(outer_r2):.5f}")
print("====================================")







