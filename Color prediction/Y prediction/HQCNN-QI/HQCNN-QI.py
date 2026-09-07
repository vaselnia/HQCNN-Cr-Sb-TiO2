# ======================================================
# Required Libraries Import
# ======================================================
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
# Deterministic Seeding and Reproducibility Setup
# ------------------------------------------------------
# Fix global random seeds across Python, NumPy, and PyTorch backends
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress Optuna info/warning logs to keep output clean
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------
# Data Loading and Preprocessing
# ------------------------------------------------------
# Load dataset from the specified CSV file
dataset = pd.read_csv("dataset_shuffle.csv")

# Extract feature matrix X and target vector y (excluding metadata and target columns)
X = dataset.drop(["Structure", "Time (h)", "O", "L", "a", "b", "y", "Reference"], axis=1).to_numpy()
y = dataset["y"].to_numpy()

# ------------------------------------------------------
# Feature and Target Normalization
# ------------------------------------------------------
# Scale input features to [0, pi] for angle embedding in quantum circuits
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target values to [-1, 1] for neural network convergence
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Partition data into 80% train and 20% test splits
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Convert arrays into PyTorch TensorDatasets
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

# ------------------------------------------------------
# Quantum Layer: Trainable Encoding + All-to-All QAOA Ansatz
# ------------------------------------------------------
# Define quantum circuit parameters
n_qubits = X.shape[1]
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def qaoa_all_to_all(inputs, enc_scale, enc_shift, gamma, delta, beta):
    # Trainable feature encoding: linear transformation with learnable scale and shift
    for i in range(n_qubits):
        qml.RY(inputs[i] * enc_scale[i] + enc_shift[i], wires=i)
        
    # All-to-all QAOA ansatz across n_layers
    for l in range(n_layers):
        # All-to-all CNOT entanglement block
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
                
        # Single-qubit parameterized rotations (RX, RY, RZ) per wire
        for i in range(n_qubits):
            qml.RX(gamma[l, i], wires=i)
            qml.RY(delta[l, i], wires=i)
            qml.RZ(beta[l, i], wires=i)
            
    # Measure Pauli-Z expectation values on all qubits
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Shape definitions for trainable quantum parameters
weight_shapes = {
    "enc_scale": n_qubits,
    "enc_shift": n_qubits,
    "gamma": (n_layers, n_qubits),
    "delta": (n_layers, n_qubits),
    "beta": (n_layers, n_qubits)
}

# Construct PyTorch-compatible TorchLayer
qnode = qml.QNode(qaoa_all_to_all, dev, interface="torch", diff_method="backprop")
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ------------------------------------------------------
# Circuit Visualization and Export
# ------------------------------------------------------
# Generate and save top-level and device-level quantum circuit diagrams (300 DPI)
qnode_for_plot = qml.QNode(qaoa_all_to_all, dev)
sample_input = np.zeros(n_qubits)
np.random.seed(SEED)
weights_sample = {
    "enc_scale": np.random.uniform(0, 2 * np.pi, size=n_qubits),
    "enc_shift": np.random.uniform(0, 2 * np.pi, size=n_qubits),
    "gamma": np.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits)),
    "delta": np.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits)),
    "beta": np.random.uniform(0, 2 * np.pi, size=(n_layers, n_qubits))
}

fig1, ax1 = qml.draw_mpl(qnode_for_plot, level="top")(sample_input, **weights_sample)
fig1.savefig("quantum_circuit_undecomposed_300dpi.png", dpi=300, bbox_inches="tight")

fig2, ax2 = qml.draw_mpl(qnode_for_plot, level="device")(sample_input, **weights_sample)
fig2.savefig("quantum_circuit_decomposed_300dpi.png", dpi=300, bbox_inches="tight")
plt.show()

# ======================================================
# Hyperparameter Tuning with Optuna
# ======================================================
def objective(trial):
    # Define hyperparameter search space
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Dynamic trial model architecture
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
            # Batch handling for the quantum layer
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

    # Optuna trial training loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        
        # Validation evaluation step
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32))
            val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()
            
        trial.report(val_loss, step=epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
    
    return val_loss

# Create and run Optuna study
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest Hyperparameters Found:")
print(study.best_params)
print(f"Best Validation Loss: {study.best_value:.6f}\n")

# ======================================================
# Final Model Training
# ======================================================
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
        # Batch processing through quantum circuit
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

# Full training loop for final model
for epoch in range(final_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Track validation loss periodically
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32))
        test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{final_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# ------------------------------------------------------
# Model Evaluation and Predictions
# ------------------------------------------------------
# Compute model predictions on test set and reverse scaling to original domain
y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Calculate regression evaluation metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nPerformance Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ------------------------------------------------------
# Parity Plot Generation (True vs. Predicted)
# ------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 10))

# Scatter plot for test points
ax.plot(y_true, y_pred, 'o', color='green', markersize=6)

# Explicit axis limits
ax.set_xlim(0.3, 0.5)
ax.set_ylim(0.3, 0.5)

# Reference identity line (y = x)
ax.plot([0.3, 0.5], [0.3, 0.5], '--', color='black', lw=2)

# Axis ticks configuration
tick_step = 0.05
ax.set_xticks(np.arange(0.3, 0.51, tick_step))
ax.set_yticks(np.arange(0.3, 0.51, tick_step))

# Axis labels
ax.set_xlabel('True y', fontsize=14)
ax.set_ylabel('Predicted y', fontsize=14)

# Equal aspect ratio for parity alignment
ax.set_aspect('equal', adjustable='box')

ax.grid(alpha=0.3)
plt.tight_layout()

plt.show()
fig.savefig('Hybrid_NN_QAOA.png', dpi=300, bbox_inches='tight')







