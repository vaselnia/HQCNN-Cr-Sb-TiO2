import random
import numpy as np
import torch
import pennylane as qml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import optuna

# ------------------------------------------------------
# Deterministic Seeding & Reproducibility Setup
# ------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------------------------------------
# Data Ingestion & Target Splitting
# ------------------------------------------------------
dataset = pd.read_csv("dataset_shuffle.csv")

X = dataset.drop(
    ["Structure", "Time (h)", "O", "L", "a", "b", "x", "Reference"], axis=1
).to_numpy()
y = dataset["x"].to_numpy()

# ------------------------------------------------------
# Normalization & Train-Test Split
# ------------------------------------------------------
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

# ------------------------------------------------------
# QAOA-Inspired Quantum Layer Setup
# ------------------------------------------------------
n_qubits = X.shape[1]
n_layers = 4
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

def qaoa_all_to_all(inputs, enc_scale, enc_shift, gamma, delta, beta):
    # Parameterized feature encoding
    for i in range(n_qubits):
        qml.RY(inputs[i] * enc_scale[i] + enc_shift[i], wires=i)
        
    # Alternating entangling and single-qubit rotational layers
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
    "beta": (n_layers, n_qubits),
}

qnode = qml.QNode(qaoa_all_to_all, dev, interface="torch", diff_method="backprop")
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ============================================================
# Optuna Hyperparameter Optimization
# ============================================================
def objective(trial):
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(X.shape[1], n_qubits)
            self.act1 = nn.LeakyReLU(0.1)
            self.quantum = qlayer
            self.fc2 = nn.Linear(n_qubits, hidden_units)
            self.act2 = nn.LeakyReLU(0.1)
            self.dropout = nn.Dropout(dropout)
            self.fc3 = nn.Linear(hidden_units, 1)

        def forward(self, x):
            x = self.act1(self.fc1(x))
            x = torch.stack([self.quantum(xi) for xi in x])
            x = self.act2(self.fc2(x))
            x = self.dropout(x)
            return self.fc3(x).squeeze()

    model = HybridNNOptuna()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    for _ in range(epochs):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32))
        val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()

    return val_loss

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

best_params = study.best_params

# ============================================================
# Final Model Architecture
# ============================================================
class HybridNNFinal(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(X.shape[1], n_qubits)
        self.act1 = nn.LeakyReLU(0.1)
        self.quantum = qlayer
        self.fc2 = nn.Linear(n_qubits, best_params["hidden_units"])
        self.act2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(best_params["dropout"])
        self.fc3 = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = torch.stack([self.quantum(xi) for xi in x])
        x = self.act2(self.fc2(x))
        x = self.dropout(x)
        return self.fc3(x).squeeze()

# ============================================================
# Nested Cross-Validation (Outer Evaluation)
# ============================================================
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

mae_list, mse_list, r2_list = [], [], []

for fold, (tr_idx, te_idx) in enumerate(kf.split(X_scaled), 1):
    print(f"\nOuter Fold {fold}")

    X_tr, X_te = X_scaled[tr_idx], X_scaled[te_idx]
    y_tr, y_te = y_scaled[tr_idx], y_scaled[te_idx]

    train_ds = TensorDataset(
        torch.tensor(X_tr, dtype=torch.float32),
        torch.tensor(y_tr, dtype=torch.float32)
    )

    loader = DataLoader(
        train_ds,
        batch_size=best_params["batch_size"],
        shuffle=True
    )

    model = HybridNNFinal()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
    criterion = nn.MSELoss()

    for _ in range(100):
        model.train()
        for bx, by in loader:
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        y_pred_scaled = model(torch.tensor(X_te, dtype=torch.float32)).numpy()

    y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
    y_true = scaler_y.inverse_transform(y_te.reshape(-1, 1)).ravel()

    mae_list.append(mean_absolute_error(y_true, y_pred))
    mse_list.append(mean_squared_error(y_true, y_pred))
    r2_list.append(r2_score(y_true, y_pred))

    print(f"MAE={mae_list[-1]:.4f}, MSE={mse_list[-1]:.4f}, R²={r2_list[-1]:.4f}")

# ============================================================
# Final Performance Summary Report
# ============================================================
print("\n📊 Nested CV Results")
print(f"MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"MSE: {np.mean(mse_list):.4f} ± {np.std(mse_list):.4f}")
print(f"R² : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")







