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

# Suppress verbose Optuna logging outputs
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
X = dataset.drop(X)

# Scale target band gap values to [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1["band_gap"].to_numpy()

# ------------------------------------------------------
# Feature & Target Normalization
# ------------------------------------------------------
# Scale input features to [0, pi] for quantum rotational angle encoding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target band gap values to [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Convert to PyTorch TensorDatasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

# ------------------------------------------------------
# Quantum Layer Setup: Hardware-Efficient Ansatz (HEA) with Full Entanglement
# ------------------------------------------------------
n_qubits = X.shape[1]
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def enhanced_hea(inputs, scale, shift, rot):
    """
    Hardware-Efficient Ansatz (HEA) with:
    1. Trainable linear scaling and shifting for feature encoding.
    2. Multi-layer universal single-qubit rotations (RX, RY, RZ).
    3. Full all-to-all CNOT entanglement pattern.
    """
    # Trainable Feature Encoding
    for i in range(n_qubits):
        qml.RY(inputs[i] * scale[i] + shift[i], wires=i)

    # Multi-layer HEA
    for l in range(n_layers):
        # Single-qubit parameterized rotations
        for i in range(n_qubits):
            qml.RX(rot[l, i, 0], wires=i)
            qml.RY(rot[l, i, 1], wires=i)
            qml.RZ(rot[l, i, 2], wires=i)
        # Full Entanglement (all-to-all CNOT gates)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])

    # Return Pauli-Z expectation values across all qubits
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

weight_shapes = {
    "scale": n_qubits,
    "shift": n_qubits,
    "rot": (n_layers, n_qubits, 3)
}

# Create PennyLane TorchLayer interface
qnode = qml.QNode(enhanced_hea, dev, interface="torch", diff_method="backprop")
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ----------- Quantum Circuit Visualization -----------
qnode_for_plot = qml.QNode(enhanced_hea, dev)
sample_input = np.zeros(n_qubits)
np.random.seed(SEED)
weights_sample = {
    "scale": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "shift": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "rot": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits, 3))
}

# Save undecomposed circuit diagram
fig1, ax1 = qml.draw_mpl(qnode_for_plot, level="top")(sample_input, **weights_sample)
fig1.savefig("quantum_circuit_undecomposed_300dpi.png", dpi=300, bbox_inches="tight")

# Save decomposed device-level circuit diagram
fig2, ax2 = qml.draw_mpl(qnode_for_plot, level="device")(sample_input, **weights_sample)
fig2.savefig("quantum_circuit_decomposed_300dpi.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================
# Optuna Hyperparameter Optimization
# ============================================================
def objective(trial):
    # Search space definition
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Hybrid Quantum-Classical architecture for Optuna trials
    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()
            self.classical1 = nn.Linear(X.shape[1], n_qubits)
            self.act1 = nn.LeakyReLU(0.1)
            self.quantum = qlayer
            self.classical2 = nn.Linear(n_qubits, hidden_units)
            self.act2 = nn.LeakyReLU(0.1)
            self.dropout = nn.Dropout(dropout)
            self.classical3 = nn.Linear(hidden_units, 1)

        def forward(self, x):
            x = self.act1(self.classical1(x))
            if x.ndim == 1:
                x = self.quantum(x)
            else:
                x = torch.stack([self.quantum(xi) for xi in x])
            x = self.act2(self.classical2(x))
            x = self.dropout(x)
            x = self.classical3(x)
            return x.squeeze()

    model = HybridNNOptuna()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
    )

    # Trial training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32))
            val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()
            
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest Hyperparameters Found by Optuna:")
print(study.best_params)
print(f"Best validation loss = {study.best_value:.6f}\n")

# ============================================================
# Final Model Training
# ============================================================
best_params = study.best_params
final_epochs = 100

class HybridNNFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.classical1 = nn.Linear(X.shape[1], n_qubits)
        self.act1 = nn.LeakyReLU(0.1)
        self.quantum = qlayer
        self.classical2 = nn.Linear(n_qubits, best_params["hidden_units"])
        self.act2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(best_params["dropout"])
        self.classical3 = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):
        x = self.act1(self.classical1(x))
        if x.ndim == 1:
            x = self.quantum(x)
        else:
            x = torch.stack([self.quantum(xi) for xi in x])
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

# Full training loop (100 epochs)
for epoch in range(final_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Model evaluation on test set
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32))
        test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{final_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

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
ax.set_yticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))

ax.set_xlabel('True $E_{g}$ (eV)')
ax.set_ylabel('Predicted $E_{g}$ (eV)')
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()

plt.show()
fig.savefig('Hybrid NN.png', dpi=300, bbox_inches='tight')






