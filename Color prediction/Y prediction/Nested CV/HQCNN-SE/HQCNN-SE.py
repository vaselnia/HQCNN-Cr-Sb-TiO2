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
# 🔒 Reproducibility
# ======================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)

# Set random seeds for reproducible results
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enable deterministic computation
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress Optuna informational messages
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================================================
# Data
# ======================================================
# Load the dataset from the CSV file
dataset = pd.read_csv("dataset_shuffle.csv")

# Select input features by excluding non-feature columns
X = dataset.drop(
    ["Structure", "Time (h)", "O", "L", "a", "b", "y", "Reference"], axis=1
).to_numpy()
y = dataset["y"].to_numpy()

# ======================================================
# Normalization
# ======================================================
# Scale input features to the range [0, pi]
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X = scaler_X.fit_transform(X)

# Scale target values to the range [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# ======================================================
# Quantum Layer
# ======================================================
# Define the number of qubits based on the number of input features
n_qubits = X.shape[1]
n_layers = 4

# Initialize the quantum device with a fixed random seed
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

# Define the quantum circuit using angle embedding and strongly entangling layers
def quantum_layer(inputs, weights):
    # Encode the classical input features into quantum rotation angles
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # Apply trainable strongly entangling quantum layers
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Measure the Pauli-Z expectation value for each qubit
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Define the trainable parameter shape of the quantum circuit
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# ======================================================
# 🔵 Nested Cross Validation
# ======================================================
# Define the outer and inner cross-validation schemes
outer_cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=SEED)

# Store evaluation metrics for each outer fold
outer_mse, outer_mae, outer_r2 = [], [], []

for outer_fold, (train_idx, test_idx) in enumerate(outer_cv.split(X)):
    print(f"\n🔁 Outer Fold {outer_fold + 1}/5")

    # Split the data into outer training and testing sets
    X_train_outer, X_test_outer = X[train_idx], X[test_idx]
    y_train_outer, y_test_outer = y[train_idx], y[test_idx]

    train_outer_ds = TensorDataset(
        torch.tensor(X_train_outer, dtype=torch.float32),
        torch.tensor(y_train_outer, dtype=torch.float32)
    )

    # ==================================================
    # 🔵 Inner CV + Optuna
    # ==================================================
    # Define the Optuna objective function using inner cross-validation
    def objective(trial):
        # Select hyperparameters from predefined search spaces
        lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
        dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
        hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
        batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
        epochs = trial.suggest_categorical("epochs", [15, 20])

        fold_losses = []

        # Perform three-fold inner cross-validation
        for tr_idx, val_idx in inner_cv.split(X_train_outer):
            X_tr, X_val = X_train_outer[tr_idx], X_train_outer[val_idx]
            y_tr, y_val = y_train_outer[tr_idx], y_train_outer[val_idx]

            tr_ds = TensorDataset(
                torch.tensor(X_tr, dtype=torch.float32),
                torch.tensor(y_tr, dtype=torch.float32)
            )

            # Create a trainable quantum layer for the current inner fold
            qlayer = qml.qnn.TorchLayer(
                qml.QNode(quantum_layer, dev),
                weight_shapes
            )

            # Define the hybrid classical-quantum neural network
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
                    # Process the input through the classical and quantum layers
                    x = self.act1(self.fc1(x))
                    x = self.q(x)
                    x = self.act2(self.fc2(x))
                    x = self.drop(x)
                    return self.out(x).squeeze()

            # Initialize the model, optimizer, and loss function
            model = HybridQNN()
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
            criterion = nn.MSELoss()

            # Create the training data loader with deterministic worker seeding
            loader = DataLoader(
                tr_ds,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=lambda wid: np.random.seed(SEED + wid)
            )

            # Train the model for the selected number of epochs
            for _ in range(epochs):
                model.train()
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(model(xb), yb)
                    loss.backward()
                    optimizer.step()

            # Evaluate the model on the inner validation set
            model.eval()
            with torch.no_grad():
                preds = model(torch.tensor(X_val, dtype=torch.float32))
                fold_losses.append(
                    criterion(preds, torch.tensor(y_val, dtype=torch.float32)).item()
                )

        # Return the mean validation loss across inner folds
        return np.mean(fold_losses)

    # Create the Optuna study for hyperparameter optimization
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=20, show_progress_bar=False)

    # Retrieve the best hyperparameters identified by Optuna
    best = study.best_params

    # ==================================================
    # 🔵 Train Final Model on Outer-Train
    # ==================================================
    # Create the quantum layer for the final outer-fold model
    qlayer_final = qml.qnn.TorchLayer(
        qml.QNode(quantum_layer, dev),
        weight_shapes
    )

    # Define the final hybrid quantum neural network
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
            # Process the input through the classical and quantum components
            x = self.act1(self.fc1(x))
            x = self.q(x)
            x = self.act2(self.fc2(x))
            x = self.drop(x)
            return self.out(x).squeeze()

    # Initialize the final model, optimizer, and loss function
    model = FinalQNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=best["lr"])
    criterion = nn.MSELoss()

    # Create the outer-training data loader using the optimized batch size
    loader = DataLoader(
        train_outer_ds,
        batch_size=best["batch_size"],
        shuffle=True,
        worker_init_fn=lambda wid: np.random.seed(SEED + wid)
    )

    # Train the final model for 100 epochs on the outer training set
    for _ in range(100):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    # ==================================================
    # 🔵 Outer Test Evaluation
    # ==================================================
    # Evaluate the trained model on the unseen outer test set
    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test_outer, dtype=torch.float32)).numpy()

    # Convert normalized targets and predictions back to the original scale
    y_true = scaler_y.inverse_transform(y_test_outer.reshape(-1, 1)).ravel()
    y_pred = scaler_y.inverse_transform(preds.reshape(-1, 1)).ravel()

    # Calculate and store the evaluation metrics for the current outer fold
    outer_mse.append(mean_squared_error(y_true, y_pred))
    outer_mae.append(mean_absolute_error(y_true, y_pred))
    outer_r2.append(r2_score(y_true, y_pred))

# ======================================================
# 🔴 Final Nested CV Results
# ======================================================
# Report the mean and standard deviation of the metrics across outer folds
print("\n===== NESTED 5×3 CV (HYBRID QNN) =====")
print(f"MSE: {np.mean(outer_mse):.5f} ± {np.std(outer_mse):.5f}")
print(f"MAE: {np.mean(outer_mae):.5f} ± {np.std(outer_mae):.5f}")
print(f"R² : {np.mean(outer_r2):.5f} ± {np.std(outer_r2):.5f}")
print("====================================")







