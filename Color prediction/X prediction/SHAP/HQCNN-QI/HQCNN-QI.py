import os
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
# Deterministic Seeding & Reproducibility Setup
# ======================================================
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================
# Data Ingestion & Target Splitting
# ======================================================
dataset = pd.read_csv("dataset_shuffle.csv")

drop_cols = ["Structure", "Time (h)", "O", "L", "a", "b", "x", "Reference"]
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

X = dataset.drop(drop_cols, axis=1).to_numpy()
y = dataset["x"].to_numpy()

# ======================================================
# Feature & Target Normalization
# ======================================================
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test, dtype=torch.float32)
)

# ======================================================
# QAOA-Inspired Quantum Layer Setup
# ======================================================
n_qubits = X.shape[1]
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def qaoa_all_to_all(inputs, enc_scale, enc_shift, gamma, delta, beta):
    # Parameterized Angle Encoding
    for i in range(n_qubits):
        qml.RY(inputs[i] * enc_scale[i] + enc_shift[i], wires=i)

    # QAOA-inspired entangling and single-qubit rotation blocks
    for l in range(n_layers):
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])

        for i in range(n_qubits):
            qml.RX(gamma[l, i], wires=i)
            qml.RY(delta[l, i], wires=i)
            qml.RZ(beta[l, i], wires=i)

    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

weight_shapes = {
    "enc_scale": n_qubits,
    "enc_shift": n_qubits,
    "gamma": (n_layers, n_qubits),
    "delta": (n_layers, n_qubits),
    "beta": (n_layers, n_qubits)
}

qnode = qml.QNode(qaoa_all_to_all, dev, interface="torch", diff_method="backprop")
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ======================================================
# Quantum Circuit Visualization
# ======================================================
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
plt.close(fig1)

fig2, ax2 = qml.draw_mpl(qnode_for_plot, level="device")(sample_input, **weights_sample)
fig2.savefig("quantum_circuit_decomposed_300dpi.png", dpi=300, bbox_inches="tight")
plt.close(fig2)

print("✅ Quantum circuit diagrams saved (300 DPI).")

# ======================================================
# Optuna Hyperparameter Optimization
# ======================================================
def objective(trial):
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

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
            return x.view(-1)

    model = HybridNNOptuna()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

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

best_params = study.best_params
print("\nOptimal Hyperparameters:", best_params)

# ======================================================
# Final Model Training
# ======================================================
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

train_loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id)
)

print("\nStarting final model training...")
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

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1:3d}/{final_epochs}] | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")

# ======================================================
# Performance Evaluation
# ======================================================
model.eval()
with torch.no_grad():
    y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).cpu().numpy()

y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\n===== FINAL EVALUATION METRICS =====")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R² : {r2:.4f}")
print("====================================")

# ======================================================
# Parity Plot (True vs. Predicted)
# ======================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(y_true, y_pred, 'o', color='magenta', alpha=0.8, edgecolors='k')

min_val = min(min(y_true), min(y_pred))
max_val = max(max(y_true), max(y_pred))

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
fig.savefig('Hybrid_NN_QAOA.png', dpi=300, bbox_inches='tight')
plt.close(fig)
print("✅ Parity plot saved as 'Hybrid_NN_QAOA.png'.")

# ======================================================
# SHAP Interpretability Analysis
# ======================================================
def model_predict(x_np):
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        preds = model(x_tensor)
    return preds.cpu().numpy().reshape(-1)

background = X_train[np.random.choice(X_train.shape[0], 30, replace=False)]

explainer = shap.KernelExplainer(model_predict, background)
shap_values = explainer.shap_values(X_test, nsamples=200)
shap_values = np.array(shap_values)

# ------------------------------------------------------
# SHAP Summary Beeswarm Plot (All Features)
# ------------------------------------------------------
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
print("✅ SHAP summary beeswarm plot saved.")

# ------------------------------------------------------
# SHAP Mean Absolute Value Bar Plot
# ------------------------------------------------------
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
print("✅ SHAP bar plot (mean |SHAP|) saved.")

# ------------------------------------------------------
# SHAP Feature Dependence Plots (Top 3 Features)
# ------------------------------------------------------
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
sorted_idx = np.argsort(mean_abs_shap)[::-1]
top_features = sorted_idx[:min(3, len(feature_names))]

print("\nTop features selected for dependence plots:")
for rank, idx in enumerate(top_features, 1):
    print(f"{rank}. {feature_names[idx]}")

for idx in top_features:
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
    plt.savefig(f"SHAP_dependence_quantum_{feature_names[idx]}.png", dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✅ SHAP dependence saved for: {feature_names[idx]}")







