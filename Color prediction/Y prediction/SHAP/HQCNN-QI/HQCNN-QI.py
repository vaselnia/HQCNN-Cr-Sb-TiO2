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

# ------------------------------
# Fixing the random seed for reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Suppress Optuna logging warnings and re-seed numpy/random for consistency
optuna.logging.set_verbosity(optuna.logging.WARNING)
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------
# Data loading
# ------------------------------
# Load the shuffled dataset from CSV
dataset = pd.read_csv("dataset_shuffle.csv")

# Columns to drop (non-feature or non-target columns)
drop_cols = ["Structure", "Time (h)", "O", "L", "a", "b", "y", "Reference"]

# Keep the names of the remaining feature columns
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

# Extract feature matrix X and target vector y as numpy arrays
X = dataset.drop(drop_cols, axis=1).to_numpy()
y = dataset["y"].to_numpy()

# ------------------------------
# Normalization / Scaling
# ------------------------------
# Scale features into [0, pi] (suitable range for quantum rotation angles)
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target into [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split into train and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Wrap data into PyTorch tensors and datasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

# ------------------------------
# Quantum layer definition
# ------------------------------
# Number of qubits equals the number of input features
n_qubits = X.shape[1]
# Number of QAOA-style layers in the ansatz
n_layers = 4
# Create a PennyLane device (simulator) with a fixed seed
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

# Define the all-to-all QAOA-style variational circuit
def qaoa_all_to_all(inputs, enc_scale, enc_shift, gamma, delta, beta):
    # Encode the input features into qubit rotations (RY gates)
    for i in range(n_qubits):
        qml.RY(inputs[i] * enc_scale[i] + enc_shift[i], wires=i)

    # Apply the variational layers
    for l in range(n_layers):
        # Entangling step: CNOT gates between all pairs of qubits (all-to-all)
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])

        # Rotational step: single-qubit rotations with trainable parameters
        for i in range(n_qubits):
            qml.RX(gamma[l, i], wires=i)
            qml.RY(delta[l, i], wires=i)
            qml.RZ(beta[l, i], wires=i)

    # Return the expectation value of PauliZ on each qubit
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Define the shapes of the trainable weight tensors
weight_shapes = {
    "enc_scale": n_qubits,
    "enc_shift": n_qubits,
    "gamma": (n_layers, n_qubits),
    "delta": (n_layers, n_qubits),
    "beta": (n_layers, n_qubits)
}

# Build the QNode with torch interface and backpropagation for gradients
qnode = qml.QNode(qaoa_all_to_all, dev, interface="torch", diff_method="backprop")
# Wrap the QNode as a PyTorch layer
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ----------- Circuit for plotting -----------
# A separate QNode (default interface) used only for drawing the circuit
qnode_for_plot = qml.QNode(qaoa_all_to_all, dev)

# Use a zero input vector for the circuit diagram
sample_input = np.zeros(n_qubits)

# Generate random sample weights for visualization purposes
np.random.seed(SEED)
weights_sample = {
    "enc_scale": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "enc_shift": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "gamma": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits)),
    "delta": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits)),
    "beta": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits))
}

# Draw and save the circuit without decomposition (top-level view)
fig1, ax1 = qml.draw_mpl(qnode_for_plot, level="top")(sample_input, **weights_sample)
fig1.savefig("quantum_circuit_undecomposed_300dpi.png", dpi=300, bbox_inches="tight")

# Draw and save the fully decomposed circuit (device-level view)
fig2, ax2 = qml.draw_mpl(qnode_for_plot, level="device")(sample_input, **weights_sample)
fig2.savefig("quantum_circuit_decomposed_300dpi.png", dpi=300, bbox_inches="tight")

plt.show()

# ============================================================
# Optuna hyperparameter optimization
# ============================================================
# Objective function evaluated by Optuna for each trial
def objective(trial):

    # Suggest hyperparameters to tune
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Define the hybrid quantum-classical neural network for a single trial
    class HybridNNOptuna(nn.Module):

        def __init__(self):
            super().__init__()
            # Classical layer mapping features to qubit count
            self.classical1 = nn.Linear(X.shape[1], n_qubits)
            self.act1 = nn.LeakyReLU(0.1)
            # Quantum layer (PennyLane TorchLayer)
            self.quantum = qlayer
            # Classical layers after the quantum part
            self.classical2 = nn.Linear(n_qubits, hidden_units)
            self.act2 = nn.LeakyReLU(0.1)
            self.dropout = nn.Dropout(dropout)
            self.classical3 = nn.Linear(hidden_units, 1)

        def forward(self, x):

            # First classical transformation + activation
            x = self.act1(self.classical1(x))

            # Apply the quantum layer (handle both single and batched inputs)
            if x.ndim == 1:
                x = self.quantum(x)
            else:
                x = torch.stack([self.quantum(xi) for xi in x])

            # Remaining classical layers
            x = self.act2(self.classical2(x))
            x = self.dropout(x)
            x = self.classical3(x)

            # Flatten output to a 1D vector
            return x.view(-1)

    # Instantiate the model
    model = HybridNNOptuna()

    # Optimizer and loss function
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Data loader for training batches
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))

    # Training loop
    for epoch in range(epochs):

        model.train()

        for batch_X, batch_y in loader:

            optimizer.zero_grad()

            outputs = model(batch_X)

            loss = criterion(outputs, batch_y)

            loss.backward()

            optimizer.step()

        # Validation after each epoch
        model.eval()

        with torch.no_grad():

            preds = model(torch.tensor(X_test, dtype=torch.float32))

            val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()

        # Report validation loss to Optuna for pruning
        trial.report(val_loss, step=epoch)

        # Prune unpromising trials early
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


# Create the Optuna study (minimize validation loss) with pruning and a seeded sampler
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

# Run the optimization for 50 trials
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ============================================================
# Final model training with best hyperparameters
# ============================================================

# Retrieve the best hyperparameters found by Optuna
best_params = study.best_params

# Number of epochs for the final training run
final_epochs = 100


# Define the final hybrid model using the best hyperparameters
class HybridNNFinal(nn.Module):

    def __init__(self):

        super().__init__()

        # Classical input layer
        self.classical1 = nn.Linear(X.shape[1], n_qubits)

        self.act1 = nn.LeakyReLU(0.1)

        # Quantum layer
        self.quantum = qlayer

        # Classical layers with best hidden units
        self.classical2 = nn.Linear(n_qubits, best_params["hidden_units"])

        self.act2 = nn.LeakyReLU(0.1)

        self.dropout = nn.Dropout(best_params["dropout"])

        self.classical3 = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):

        x = self.act1(self.classical1(x))

        # Apply quantum layer (single or batched input)
        if x.ndim == 1:
            x = self.quantum(x)
        else:
            x = torch.stack([self.quantum(xi) for xi in x])

        x = self.act2(self.classical2(x))

        x = self.dropout(x)

        x = self.classical3(x)

        return x.view(-1)


# Instantiate the final model
model = HybridNNFinal()

# Optimizer with the best learning rate
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

# Loss function
criterion = nn.MSELoss()

# Data loader with the best batch size
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True,
                          worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))


# Final training loop
for epoch in range(final_epochs):

    model.train()

    for batch_X, batch_y in train_loader:

        optimizer.zero_grad()

        outputs = model(batch_X)

        loss = criterion(outputs, batch_y)

        loss.backward()

        optimizer.step()

    # Evaluate on the test set after each epoch
    model.eval()

    with torch.no_grad():

        test_preds = model(torch.tensor(X_test, dtype=torch.float32))

        test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))

    # Print progress every 10 epochs
    if (epoch+1) % 10 == 0:

        print(f"Epoch {epoch+1}/{final_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# ------------------------------
# Evaluation
# ------------------------------

# Get scaled predictions on the test set
y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

# Inverse-transform predictions and true values back to original scale
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Compute regression metrics
mse = mean_squared_error(y_true, y_pred)

mae = mean_absolute_error(y_true, y_pred)

r2 = r2_score(y_true, y_pred)

print("\nPerformance Metrics:")

print(f"MSE: {mse:.4f}")

print(f"MAE: {mae:.4f}")

print(f"R²: {r2:.4f}")

# ------------------------------
# Prediction plot
# ------------------------------

# Scatter plot of true vs predicted values
fig, ax = plt.subplots(figsize=(10, 10))

ax.plot(y_true, y_pred, 'o', color='magenta')

# Set equal axis limits based on data range
min_val = min(min(y_true), min(y_pred))
max_val = max(max(y_true), max(y_pred))

ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

# Draw the ideal y=x reference line
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', lw=2)

# Set tick spacing
tick_step = 0.1

ax.set_xticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))
ax.set_yticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))

# Axis labels (band gap in eV)
ax.set_xlabel('True $E_{g}$ (eV)')
ax.set_ylabel('Predicted $E_{g}$ (eV)')

# Keep equal aspect ratio for a fair comparison
ax.set_aspect('equal', adjustable='box')

fig.tight_layout()

plt.show()

# Save the prediction plot
fig.savefig('Hybrid_NN_QAOA.png', dpi=300, bbox_inches='tight')

# ======================================================
# SHAP explainability analysis
# ======================================================

# Wrapper function to make predictions from numpy input (for SHAP)
def model_predict(x_np):

    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    with torch.no_grad():

        preds = model(x_tensor)

    return preds.numpy().reshape(-1)


# Select a random background sample (30 points) from the training set
background = X_train[np.random.choice(X_train.shape[0], 30, replace=False)]

# Create a KernelExplainer for the model
explainer = shap.KernelExplainer(model_predict, background)

# Compute SHAP values for the test set
shap_values = explainer.shap_values(X_test, nsamples=200)

shap_values = np.array(shap_values)

# Summary plot (beeswarm) of SHAP values
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

# Bar plot of mean absolute SHAP values (feature importance)
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

# Compute mean absolute SHAP values and rank features
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

sorted_idx = np.argsort(mean_abs_shap)[::-1]

# Take the top 3 most important features
top_features = sorted_idx[:3]

# Create a dependence plot for each top feature
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





