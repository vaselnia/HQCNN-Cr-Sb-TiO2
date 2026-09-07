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

# ------------------------------------------------------
# Reproducibility Setup
# ------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------
# Optuna Logging & Seed Configuration
# ------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------
# Data Loading & Preparation
# ------------------------------------------------------
dataset = pd.read_csv("dataset_shuffle.csv")

# ---------- Physics-aware feature engineering ----------
cbm = dataset["cbm"].values
vbm = dataset["vbm"].values
efermi = dataset["efermi"].values

# dataset["delta_c"] = cbm - efermi          # CBM - EF
# dataset["delta_v"] = efermi - vbm          # EF - VBM
dataset["asym"] = (cbm + vbm)/2 - efermi   # Band edge asymmetry relative to Fermi level

# ---------- Drop trivial band edges & metadata columns ----------
X = dataset.drop([
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume",
    "cbm", "vbm", "efermi"
], axis=1).to_numpy()

y = dataset["band_gap"].to_numpy()

# ------------------------------------------------------
# Feature & Target Normalization
# ------------------------------------------------------
# Scale input features to [0, pi] for quantum angle embedding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X).utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import optuna

# ------------------------------------------------------
# Reproducibility Setup
# ------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------------------------------
# Optuna Logging & Seed Configuration
# ------------------------------------------------------
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------
# Data Loading & Preparation
# ------------------------------------------------------
dataset = pd.read_csv("dataset_shuffle.csv")

# ---------- Physics-aware feature engineering ----------
cbm = dataset["cbm"].values
vbm = dataset["vbm"].values
efermi = dataset["efermi"].values

# dataset["delta_c"] = cbm - efermi          # CBM - EF
# dataset["delta_v"] = efermi - vbm          # EF - VBM
dataset["asym"] = (cbm + vbm)/2 - efermi   # Band edge asymmetry relative to Fermi level

# ---------- Drop trivial band edges & metadata columns ----------
X = dataset.drop([
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume",
    "cbm", "vbm", "efermi"
], axis=1).to_numpy()

y = dataset["band_gap"].to_numpy()

# ------------------------------------------------------
# Feature & Target Normalization
# ------------------------------------------------------
# Scale input features to [0, pi] for quantum angle embedding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)# ----------- Undecomposed Circuit (Top Level) -----------
fig1, ax1 = qml.draw_mpl(qnode_for_plot, level="top")(sample_input, weights_sample)
undecomposed_filename = "quantum_circuit_undecomposed_300dpi.png"
fig1.savefig(undecomposed_filename, dpi=300, bbox_inches="tight")
print(f"Undecomposed circuit saved as: {undecomposed_filename}")

# ----------- Fully Decomposed Circuit (Device Level) -----------
fig2, ax2 = qml.draw_mpl(qnode_for_plot, level="device")(sample_input, weights_sample)
decomposed_filename = "quantum_circuit_decomposed_300dpi.png"
fig2.savefig(decomposed_filename, dpi=300, bbox_inches="tight")
print(f"Decomposed circuit saved as: {decomposed_filename}")

plt.show()

# ============================================================
# Optuna Hyperparameter Optimization
# ============================================================
def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])  # Reduced epochs for tuning

    # Hybrid Quantum-Classical architecture for Optuna trials
    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super(HybridNNOptuna, self).__init__()
            self.classical1 = nn.Linear(X.shape[1], n_qubits)
            self.act1 = nn.LeakyReLU(0.1)
            self.quantum = qlayer
            self.classical2 = nn.Linear(n_qubits, hidden_units)
            self.act2 = nn.LeakyReLU(0.1)
            self.dropout = nn.Dropout(dropout)
            self.classical3 = nn.Linear(hidden_units, 1)

        def forward(self, x):
            x = self.act1(self.classical1(x))
            x = self.quantum(x)
            x = self.act2(self.classical2(x))
            x = self.dropout(x)
            x = self.classical3(x)
            return x.squeeze()

    model = HybridNNOptuna()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    # Reproducible DataLoader with worker seed initialization
    loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
    )

    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Report validation loss per epoch to Optuna
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32))
            val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()
        trial.report(val_loss, step=epoch)

        # Early stopping / pruning for underperforming trials
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

# ------------------------------------------------------
# Run Optuna Study with Fixed Random Seed
# ------------------------------------------------------
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\n🎯 Best Hyperparameters Found by Optuna:")
print(study.best_params)
print(f"Best validation loss = {study.best_value:.6f}\n")

# ============================================================
# Final Model Training with Optimal Hyperparameters
# ============================================================
best_params = study.best_params
final_epochs = 100  # Full final training phase

class HybridNNFinal(nn.Module):
    def __init__(self):
        super(HybridNNFinal, self).__init__()
        self.classical1 = nn.Linear(X.shape[1], n_qubits)
        self.act1 = nn.LeakyReLU(0.1)
        self.quantum = qlayer
        self.classical2 = nn.Linear(n_qubits, best_params["hidden_units"])
        self.act2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(best_params["dropout"])
        self.classical3 = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):
        x = self.act1(self.classical1(x))
        x = self.quantum(x)
        x = self.act2(self.classical2(x))
        x = self.dropout(x)
        x = self.classical3(x)
        return x.squeeze()

model = HybridNNFinal()
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
criterion = nn.MSELoss()
train_loader = DataLoader(
    train_dataset, 
    batch_size=best_params["batch_size"], 
    shuffle=True,
    worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
)
epochs = final_epochs

# Full training loop
for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Evaluation on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32))
        test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# ------------------------------------------------------
# Prediction & Model Evaluation
# ------------------------------------------------------
y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nPerformance Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ------------------------------------------------------
# Parity Plot (True vs Predicted)
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(y_true, y_pred, 'o', color='blue')

min_val = round(min(min(y_true), min(y_pred)))
max_val = round(max(max(y_true), max(y_pred)))
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', lw=2)

tick_step = 0.1
ax.set_xticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))
ax.set_yticks()
ax.set_ylim(min_val, max_val)

ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', lw=2)

tick_step = 0.1
ax.set_xticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))
ax.set_yticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))

ax.set_xlabel('True $E_{g}$ (eV)')
ax.set_ylabel('Predicted $E_{g}$ (eV)')
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()

plt.show()
fig.savefig('Hybrid NN.png', dpi=300, bbox_inches='tight')






