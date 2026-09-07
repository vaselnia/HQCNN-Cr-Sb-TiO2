import os
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import matplotlib.pyplot as plt
import optuna

# ======================================================
# Reproducibility Setup
# ======================================================

# Set random seed across all libraries for deterministic execution
SEED = 42

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enforce deterministic PyTorch algorithms
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
# Scale input features to [0, pi] (suitable for quantum/rotational encodings)
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale target values to [-1, 1]
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Train/Test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Convert to PyTorch TensorDatasets
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

# ------------------------------
# Classical Quantum-like Layer Definition
# ------------------------------
n_qubits = X.shape[1]
n_layers = 4

# Multi-layer classical network mimicking parameterized quantum circuit behavior
class ClassicalQuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        layers = []
        input_size = n_qubits
        for _ in range(n_layers):
            layers.append(nn.Linear(input_size, n_qubits))
            layers.append(nn.Tanh())
            input_size = n_qubits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Instantiate the layer
clayer = ClassicalQuantumLayer(n_qubits, n_layers)

# ======================================================
# Hyperparameter Optimization (Optuna)
# ======================================================

# Setup deterministic TPE sampler and median pruner
sampler = optuna.samplers.TPESampler(seed=SEED)
study = optuna.create_study(direction="minimize",
                            sampler=sampler,
                            pruner=optuna.pruners.MedianPruner())


def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])
    hidden_units = trial.suggest_categorical("hidden_units", 
                                             [8, 9, 10, 11, 12, 13, 14, 15, 16, 32])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Model architecture for Optuna trials
    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super(HybridNNOptuna, self).__init__()
            self.classical1 = nn.Linear(X.shape[1], n_qubits)
            self.act1 = nn.LeakyReLU(0.1)
            self.quantum_equiv = clayer
            self.classical2 = nn.Linear(n_qubits, hidden_units)
            self.act2 = nn.LeakyReLU(0.1)
            self.dropout = nn.Dropout(dropout)
            self.classical3 = nn.Linear(hidden_units, 1)

        def forward(self, x):
            x = self.act1(self.classical1(x))
            x = self.quantum_equiv(x)
            x = self.act2(self.classical2(x))
            x = self.dropout(x)
            x = self.classical3(x)
            return x.squeeze()

    model = HybridNNOptuna()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Seeded data loader for reproducible batching
    g = torch.Generator()
    g.manual_seed(SEED)

    loader = DataLoader(train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        generator=g)

    # Training and validation loop
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32))
            val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()

        # Report to Optuna and prune unpromising trials
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return val_loss


# Run optimization trials
study.optimize(objective, n_trials=50, show_progress_bar=True)

print("\nBest Hyperparameters:")
print(study.best_params)
print(f"Best validation loss = {study.best_value:.6f}\n")

# ------------------------------
# Train Final Model with Best Parameters
# ------------------------------
best_params = study.best_params
final_epochs = 100

# Final hybrid model architecture
class HybridNNFinal(nn.Module):
    def __init__(self):
        super(HybridNNFinal, self).__init__()
        self.classical1 = nn.Linear(X.shape[1], n_qubits)
        self.act1 = nn.LeakyReLU(0.1)
        self.quantum_equiv = clayer
        self.classical2 = nn.Linear(n_qubits, best_params["hidden_units"])
        self.act2 = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(best_params["dropout"])
        self.classical3 = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):
        x = self.act1(self.classical1(x))
        x = self.quantum_equiv(x)
        x = self.act2(self.classical2(x))
        x = self.dropout(x)
        x = self.classical3(x)
        return x.squeeze()

# Build and configure final model
model = HybridNNFinal()
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])
criterion = nn.MSELoss()

g_final = torch.Generator()
g_final.manual_seed(SEED)

train_loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    generator=g_final
)

# Full training loop
for epoch in range(final_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

    # Track test set loss during training
    model.eval()
    with torch.no_grad():
        test_preds = model(torch.tensor(X_test, dtype=torch.float32))
        test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{final_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# ------------------------------
# Evaluation & Metrics
# ------------------------------
# Generate predictions on test set and revert scaling
y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Calculate regression performance metrics
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

print("\nPerformance Metrics:")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"R²: {r2:.4f}")

# ------------------------------
# Parity Plot (True vs. Predicted)
# ------------------------------
fig, ax = plt.subplots(figsize=(10, 10))
ax.plot(y_true, y_pred, 'o', color='magenta')

# Set equal plot limits based on data range
min_val = round(min(min(y_true), min(y_pred)))
max_val = round(max(max(y_true), max(y_pred)))
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

# Ideal prediction diagonal line
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', lw=2)

tick_step = 0.1
ax.set_xticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))
ax.set_yticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))

ax.set_xlabel('True $E_{g}$ (eV)')
ax.set_ylabel('Predicted $E_{g}$ (eV)')
ax.set_aspect('equal', adjustable='box')
fig.tight_layout()

plt.show()
fig.savefig('Classical NN.png', dpi=300, bbox_inches='tight')

# ------------------------------
# Network Architecture Visualization
# ------------------------------
def plot_nn_professional_save(layer_sizes, filename="NN_architecture.png"):
    """Draw and save a diagram of the neural network architecture."""
    n_layers = len(layer_sizes)
    max_neurons = max(layer_sizes)
    h_spacing = 3.5
    radius = 0.3
    font_size = 10
    vertical_text_offset = 2.0 

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('off')

    neuron_positions = []

    # Assign layer colors and labels
    layer_colors = []
    layer_labels = []
    for i in range(n_layers):
        if i == 0:
            layer_colors.append('#2E2EFF')
            layer_labels.append("Input\nlayer")
        elif i == n_layers-1:
            layer_colors.append('#D1D100')
            layer_labels.append("Output\nlayer")
        elif i == 1:
            layer_colors.append('#AE0000')
            layer_labels.append("Quantum-equivalent\nlayer")
        else:
            layer_colors.append('#2BB02B')
            layer_labels.append("Hidden\nlayer")

    # Plot neurons as circles
    for i, n_neurons in enumerate(layer_sizes):
        layer_pos = []
        y_offset = (max_neurons - n_neurons)/2
        for j in range(n_neurons):
            x = i * h_spacing
            y = j + y_offset
            circle = plt.Circle((x, y), radius, color=layer_colors[i], ec='k', zorder=4)
            ax.add_patch(circle)
            layer_pos.append((x, y))
        neuron_positions.append(layer_pos)

        y_text = -vertical_text_offset
        ax.text(x, y_text, layer_labels[i], ha='center', fontsize=font_size, fontweight='bold', linespacing=1.8)

    # Plot synaptic connections between adjacent layers
    for i in range(n_layers-1):
        for (x0, y0) in neuron_positions[i]:
            for (x1, y1) in neuron_positions[i+1]:
                ax.plot([x0, x1], [y0, y1], color='gray', lw=0.5, zorder=1)

    ax.set_xlim(-1, h_spacing*(n_layers-1)+1)
    ax.set_ylim(-1 - vertical_text_offset, max_neurons+1)
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

    fig.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved figure as {filename}")


# Plot architecture for the current model configuration
layer_sizes = [X.shape[1], X.shape[1], X.shape[1], best_params["hidden_units"], 1]
plot_nn_professional_save(layer_sizes, filename="HybridNN_Architecture.png")



