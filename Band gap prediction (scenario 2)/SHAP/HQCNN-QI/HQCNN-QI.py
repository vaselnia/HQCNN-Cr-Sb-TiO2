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

# ------------------------------------------------------
# Deterministic Seeding & Environment Configuration
# ------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress Optuna progress and informational logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------------------------------
# Data Ingestion & Physics-Aware Feature Engineering
# ------------------------------------------------------
dataset = pd.read_csv("dataset_shuffle.csv")

# ---------- Physics-Aware Feature Engineering ----------
# Extract band edge extrema and Fermi energy level
cbm = dataset["cbm"].values
vbm = dataset["vbm"].values
efermi = dataset["efermi"].values

# dataset["delta_c"] = cbm - efermi          # CBM - EF
# dataset["delta_v"] = efermi - vbm          # EF - VBM
# Calculate band edge asymmetry relative to Fermi level
dataset["asym"] = (cbm + vbm)/2 - efermi   # asymmetry

# ---------- Drop Redundant & Non-Predictive Columns ----------
drop_cols = [
    "material_id", "formula_pretty", "band_gap", "nelements",
    "density_atomic", "density", "formation_energy_per_atom",
    "energy_above_hull", "total_magnetization", "volume",
    "cbm", "vbm", "efermi"
]

# Preserve feature names for interpretability (SHAP)
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

# Feature matrix and ground truth regression targets
X = dataset.drop(drop_cols, axis=1).to_numpy()

y = dataset["band_gap"].to_numpy()

# ------------------------------------------------------
# Normalization & Dataset Splitting
# ------------------------------------------------------
# Scale input features to [0, pi] for quantum rotational embedding
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target band gaps to [-1, 1] aligned with Pauli-Z spectrum
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Partition dataset into reproducible train (80%) and test (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

# ------------------------------------------------------
# Parameterized Quantum Circuit (QAOA-Style Ansatz)
# ------------------------------------------------------
n_qubits = X.shape[1]
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def qaoa_all_to_all(inputs, enc_scale, enc_shift, gamma, delta, beta):
    # Learnable angle embedding
    for i in range(n_qubits):
        qml.RY(inputs[i] * enc_scale[i] + enc_shift[i], wires=i)

    # Entanglement and variational single-qubit Euler rotations
    for l in range(n_layers):
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])

        for i in range(n_qubits):
            qml.RX(gamma[l, i], wires=i)
            qml.RY(delta[l, i], wires=i)
            qml.RZ(beta[l, i], wires=i)

    # Measure expectation values of Pauli-Z operator
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

weight_shapes = {
    "enc_scale": n_qubits,
    "enc_shift": n_qubits,
    "gamma": (n_layers, n_qubits),
    "delta": (n_layers, n_qubits),
    "beta": (n_layers, n_qubits)
}

# Construct differentiable PyTorch quantum layer
qnode = qml.QNode(qaoa_all_to_all, dev, interface="torch", diff_method="backprop")
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ----------- Circuit Visualization & Export -----------
qnode_for_plot = qml.QNode(qaoa_all_to_all, dev)

sample_input = np.zeros(n_qubits)

np.random.seed(SEED)
weights_sample = {
    "enc_scale": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "enc_shift": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "gamma": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits)),
    "delta": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits)),
    "beta": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits))
}

# Render and save top-level schematic
fig1, ax1 = qml.draw_mpl(qnode_for_plot, level="top")(sample_input, **weights_sample)
fig1.savefig("quantum_circuit_undecomposed_300dpi.png", dpi=300, bbox_inches="tight")

# Render and save fully decomposed gate representation
fig2, ax2 = qml.draw_mpl(qnode_for_plot, level="device")(sample_input, **weights_sample)
fig2.savefig("quantum_circuit_decomposed_300dpi.png", dpi=300, bbox_inches="tight")

plt.show()

# ============================================================
# Hyperparameter Optimization (Optuna)
# ============================================================
def objective(trial):

    # Hyperparameter search space
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Trial hybrid network architecture
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

            # Handle multi-sample batch processing via sequential stack
            if x.ndim == 1:
                x = self.quantum(x)
            else:
                x = torch.stack([self.quantum(xi) for xi in x])

            x = self.act2(self.classical2(x))
            x = self.dropout(x)
            x = self.classical3(x)

            return x.view(-1)

    model = HybridNNOptuna()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))

    # Trial training and pruning loop
    for epoch in range(epochs):

        model.train()

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

        model.eval()

        with torch.no_grad():

            preds = model(torch.tensor(X_test, dtype=torch.float32))

            val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()

        trial.report(val_loss, step=epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


# Initialize Optuna study with TPE sampler and median pruning
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

study.optimize(objective, n_trials=50, show_progress_bar=True)

# ============================================================
# Final Optimized Hybrid Model
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

        return x.view(-1)


model = HybridNNFinal()

optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

criterion = nn.MSELoss()

train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True,
                          worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))


# Full training cycle of optimized model
for epoch in range(final_epochs):

    model.train()

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

    model.eval()

    with torch.no_grad():

        test_preds = model(torch.tensor(X_test, dtype=torch.float32))

        test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))

    if (epoch+1) % 10 == 0:

        print(f"Epoch {epoch+1}/{final_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# ------------------------------------------------------
# Model Evaluation & Metrics Computation
# ------------------------------------------------------

# Inverse-transform predictions and targets to original physical scale (eV)
y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Compute statistical regression performance metrics
mse = mean_squared_error(y_true, y_pred)

mae = mean_absolute_error(y_true, y_pred)

r2 = r2_score(y_true, y_pred)

print("\nPerformance Metrics:")

print(f"MSE: {mse:.4f}")

print(f"MAE: {mae:.4f}")

print(f"R²: {r2:.4f}")

# ------------------------------------------------------
# Parity Plot Generation (Predicted vs. True)
# ------------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(y_true, y_pred, 'o', color='magenta')

min_val = min(min(y_true), min(y_pred))
max_val = max(max(y_true), max(y_pred))

ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

# Plot reference identity parity line
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', lw=2)

tick_step = 0.1

ax.set_xticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))
ax.set_yticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))

ax.set_xlabel('True $E_{g}$ (eV)')
ax.set_ylabel('Predicted $E_{g}$ (eV)')

ax.set_aspect('equal', adjustable='box')

fig.tight_layout()

plt.show()

fig.savefig('Hybrid_NN_QAOA.png', dpi=300, bbox_inches='tight')

# ======================================================
# SHAP Model Interpretability Analysis
# ======================================================

# SHAP prediction wrapper function
def model_predict(x_np):

    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    with torch.no_grad():

        preds = model(x_tensor)

    return preds.numpy().reshape(-1)


# Sample background distribution for reference expectation values
background = X_train[np.random.choice(X_train.shape[0], 50, replace=False)]

explainer = shap.KernelExplainer(model_predict, background)

shap_values = explainer.shap_values(X_test, nsamples=200)

shap_values = np.array(shap_values)

# Global SHAP beeswarm summary plot
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)

plt.tight_layout()

plt.savefig("SHAP_summary_all_features_quantum.png", dpi=300, bbox_inches="tight")

plt.close()

# Global feature importance bar plot (Mean |SHAP|)
plt.figure(figsize=(10,6))

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

# Identify top influential features for dependence analysis
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

sorted_idx = np.argsort(mean_abs_shap)[::-1]

top_features = sorted_idx[:3]

# Generate and save SHAP dependence plots for top-3 features
for idx in top_features:

    plt.figure(figsize=(7,5))

    shap.dependence_plot(
        idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )

    plt.tight_layout()

    plt.savefig(f"SHAP_dependence_quantum_{feature_names[idx]}.png", dpi=300, bbox_inches="tight")

    plt.close()







