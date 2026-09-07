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
# Reproducibility Setup (Deterministic Mode)
# ======================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# Seed all standard, numerical, and deep learning libraries
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enforce deterministic PyTorch operations
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress verbose Optuna logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================
# Data Loading & Preprocessing
# ======================================================
dataset = pd.read_csv("dataset_shuffle.csv")

# Extract features by dropping metadata and non-predictive columns
X = dataset.drop(
    ["Structure", "Time (h)", "O", "L", "a", "b", "x", "Reference"], axis=1
).to_numpy()

# Target variable
y = dataset["x"].to_numpy()

# ======================================================
# Feature & Target Normalization
# ======================================================
# Scale input features to [0, pi] for angle embedding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X = scaler_X.fit_transform(X)

# Scale target values to [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ======================================================
# Quantum Layer Setup (Enhanced Hardware Efficient Ansatz)
# ======================================================
n_qubits = X.shape[1]
n_layers = 4

# Initialize quantum device
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def enhanced_hea(inputs, scale, shift, rot):
    """
    Enhanced Hardware-Efficient Ansatz (HEA) with trainable feature encoding
    (scaled/shifted RY rotations) and all-to-all CNOT entangling blocks.
    """
    # Trainable feature encoding layer
    for i in range(n_qubits):
        qml.RY(inputs[i] * scale[i] + shift[i], wires=i)

    # Variational layers: single-qubit rotations followed by all-to-all entangling gates
    for l in range(n_layers):
        for i in range(n_qubits):
            qml.RX(rot[l, i, 0], wires=i)
            qml.RY(rot[l, i, 1], wires=i)
            qml.RZ(rot[l, i, 2], wires=i)
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])

    # Return expectation values of Pauli-Z on all qubits
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Parameter shape mapping for PennyLane TorchLayer
weight_shapes = {
    "scale": n_qubits,
    "shift": n_qubits,
    "rot": (n_layers, n_qubits, 3)
}

# ======================================================
# Nested Cross-Validation (5x3 Scheme)
# ======================================================
# Outer CV for generalized performance evaluation (5 folds)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
# Inner CV for hyperparameter optimization (3 folds)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

outer_mse, outer_mae, outer_r2 = [], [], []

for fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"\n🔁 Outer Fold {fold + 1}/5")

    # Split outer training and testing sets
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    train_outer_ds = TensorDataset(
        torch.tensor(X_train_outer, dtype=torch.float32),
        torch.tensor(y_train_outer, dtype=torch.float32)
    )

    # ==================================================
    # Inner CV + Optuna Hyperparameter Optimization
    # ==================================================
    def objective(trial):
        # Hyperparameter search space
        lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
        hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_categorical("epochs", [15, 20])

        fold_losses = []

        # Inner cross-validation loop
        for tr_idx, val_idx in inner_cv.split(X_train_outer):
            X_tr, X_val = X_train_outer[tr_idx], X_train_outer[val_idx]
            y_tr, y_val = y_train_outer[tr_idx], y_train_outer[val_idx]

            tr_ds = TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32)
            )

            # Fresh quantum layer per trial model
            qlayer = qml.qnn.TorchLayer(
                qml.QNode(enhanced_hea, dev, interface="torch"),
                weight_shapes
            )

            # Hybrid Quantum-Classical architecture
            class HybridHEA(nn.Module):
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

            model = HybridHEA()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            loader = DataLoader(
                tr_ds,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=lambda wid: np.random.seed(SEED + wid)
            )

            # Train on inner training split
            for _ in range(epochs):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            # Evaluate on inner validation split
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32))
                fold_losses.append(
                    criterion(preds, torch.tensor(y_val, dtype=torch.float32)).item()
                )

        # Average loss across inner folds serves as the objective metric
        return np.mean(fold_losses)

    # Optimize hyperparameters using TPE sampler
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=25, show_progress_bar=False)

    best = study.best_params

    # ==================================================
    # Train Final Model on Outer Training Split
    # ==================================================
    qlayer_final = qml.qnn.TorchLayer(
        qml.QNode(enhanced_hea, dev, interface="torch"),
        weight_shapes
    )

    class FinalHEA(nn.Module):
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

    model = FinalHEA()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best["lr"])
    criterion = nn.MSELoss()

    loader = DataLoader(
        train_outer_ds,
        batch_size=best["batch_size"],
        shuffle=True,
        worker_init_fn=lambda wid: np.random.seed(SEED + wid)
    )

    # Full training loop for the best configured model
    for _ in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # ==================================================
    # Outer Test Evaluation & Metrics
    # ==================================================
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_outer, dtype=torch.float32)).numpy()

    # Revert target normalization to original scale
    y_true = scaler_y.inverse_transform(y_test_outer.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()

    # Accumulate outer fold scores
    outer_mse.append(mean_squared_error(y_true, y_pred))
    outer_mae.append(mean_absolute_error(y_true, y_pred))
    outer_r2.append(r2_score(y_true, y_pred))

# ======================================================
# Final Results Summary
# ======================================================
print("\n===== NESTED 5×3 CV (ENHANCED HEA-QNN) =====")
print(f"MSE: {np.mean(outer_mse):.5f} ± {np.std(outer_mse):.5f}")
print(f"MAE: {np.mean(outer_mae):.5f} ± {np.std(outer_mae):.5f}")
print(f"R² : {np.mean(outer_r2):.5f} ± {np.std(outer_r2):.5f}")
print("==========================================")


