import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import optuna

# ======================================================
# Reproducibility Setup (Strict Deterministic Mode)
# ======================================================
SEED = 42

# Ensure reproducibility across hash seeds and cuBLAS operations
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Seed all random number generators
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enforce deterministic PyTorch and cuDNN algorithms
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------
# Data Loading & Preprocessing
# ------------------------------
# Load dataset
dataset = pd.read_csv("dataset_shuffle.csv")

# Separate features by dropping target and metadata columns
X = dataset.drop([
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume"
], axis=1).to_numpy()

# Target variable (band gap)
y = dataset["band_gap"].to_numpy()

# ------------------------------
# Feature & Target Scaling
# ------------------------------
# Scale input features to [0, pi]
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X = scaler_X.fit_transform(X)

# Scale target values to [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ------------------------------
# Classical Quantum-like Layer
# ------------------------------
n_qubits = X.shape[1]
n_layers = 4

class ClassicalQuantumLayer(nn.Module):
    """
    Classical feed-forward approximation of a parameterized quantum circuit
    using stacked linear projections and non-linear Tanh activations.
    """
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(nn.Linear(n_qubits, n_qubits))
            layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ======================================================
# Nested Cross-Validation (5x3 Scheme)
# ======================================================
# Outer CV for unbiased performance estimation (5 folds)
outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
# Inner CV for hyperparameter tuning (3 folds)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

outer_mse, outer_mae, outer_r2 = [], [], []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"\n🔁 Outer Fold {outer_fold + 1}/5")

    # Split outer training and test sets
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
        hidden_units = trial.suggest_categorical(
            "hidden_units", [4, 8, 16, 32]
        )
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

            # Hybrid model architecture for inner validation
            class HybridNN(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc1 = nn.Linear(n_qubits, n_qubits)
                    self.act1 = nn.LeakyReLU(0.1)
                    self.q = ClassicalQuantumLayer(n_qubits, n_layers)
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

            model = HybridNN()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            loader = DataLoader(
                tr_ds, batch_size=batch_size, shuffle=True
            )

            # Training on inner fold
            for _ in range(epochs):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            # Evaluation on validation fold
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32))
                fold_losses.append(
                    criterion(preds, torch.tensor(y_val, dtype=torch.float32)).item()
                )

        # Objective value is mean loss across inner folds
        return np.mean(fold_losses)

    # Initialize and execute Optuna study
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=30, show_progress_bar=False)

    best = study.best_params

    # ==================================================
    # Retrain Final Model on Full Outer-Train Fold
    # ==================================================
    class FinalModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(n_qubits, n_qubits)
            self.act1 = nn.LeakyReLU(0.1)
            self.q = ClassicalQuantumLayer(n_qubits, n_layers)
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

    model = FinalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best["lr"])
    criterion = nn.MSELoss()

    loader = DataLoader(
        train_outer_ds,
        batch_size=best["batch_size"],
        shuffle=True
    )

    # Full training loop on outer training split
    for _ in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # ==================================================
    # Outer Test Set Evaluation
    # ==================================================
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_outer, dtype=torch.float32)).numpy()

    # Revert target scaling
    y_true = scaler_y.inverse_transform(y_test_outer.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()

    # Collect outer fold metrics
    outer_mse.append(mean_squared_error(y_true, y_pred))
    outer_mae.append(mean_absolute_error(y_true, y_pred))
    outer_r2.append(r2_score(y_true, y_pred))

# ======================================================
# Final Nested CV Summary Report
# ======================================================
print("\n===== NESTED 5×3 CV RESULTS =====")
print(f"MSE: {np.mean(outer_mse):.5f} ± {np.std(outer_mse):.5f}")
print(f"MAE: {np.mean(outer_mae):.5f} ± {np.std(outer_mae):.5f}")
print(f"R² : {np.mean(outer_r2):.5f} ± {np.std(outer_r2):.5f}")
print("================================")





