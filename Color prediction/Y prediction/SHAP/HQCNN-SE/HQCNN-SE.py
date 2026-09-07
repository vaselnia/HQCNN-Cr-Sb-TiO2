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
# Reproducibility
# ======================================================
# Set a fixed random seed to make Python, NumPy, and PyTorch operations reproducible.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ======================================================
# Data
# ======================================================
# Load the shuffled dataset from the CSV file.
dataset = pd.read_csv("dataset_shuffle.csv")

# Define columns that should not be used as model input features.
drop_cols = ["Structure", "Time (h)", "O", "L", "a", "b", "y", "Reference"]

# Extract the names of the remaining input features.
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

# Convert the selected input features and target variable to NumPy arrays.
X = dataset.drop(drop_cols, axis=1).to_numpy()
y = dataset["y"].to_numpy()

# ======================================================
# Scaling
# ======================================================
# Scale input features to the [0, pi] interval for quantum angle encoding.
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale the target variable to the [-1, 1] interval.
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split the scaled dataset into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Convert the training data into a PyTorch TensorDataset.
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

# ======================================================
# Quantum Layer
# ======================================================
# Use one qubit for each input feature.
n_qubits = X.shape[1]

# Define the number of strongly entangling quantum layers.
n_layers = 4

# Create the PennyLane default-qubit simulator with a fixed seed.
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

# Define the variational quantum circuit used as the quantum layer.
def quantum_layer(inputs, weights):
    # Encode classical input values into qubit rotation angles.
    qml.AngleEmbedding(inputs, wires=range(n_qubits))

    # Apply trainable strongly entangling quantum layers.
    qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))

    # Return the expectation value of Pauli-Z for every qubit.
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Specify the trainable quantum weight tensor dimensions.
weight_shapes = {"weights": (n_layers, n_qubits, 3)}

# Convert the PennyLane QNode into a PyTorch-compatible quantum layer.
qlayer = qml.qnn.TorchLayer(qml.QNode(quantum_layer, dev), weight_shapes)

# ======================================================
# Optuna
# ======================================================
# Suppress Optuna informational logs and keep the output concise.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Define the Optuna objective function for hyperparameter optimization.
def objective(trial):
    # Sample the learning rate from the predefined categorical candidates.
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])

    # Sample the dropout rate from the predefined categorical candidates.
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])

    # Sample the number of hidden classical units.
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])

    # Sample the training batch size.
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Sample the number of training epochs for each trial.
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Define the hybrid neural network used during each Optuna trial.
    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()

            # Classical layer that maps the input features to the quantum dimension.
            self.c1 = nn.Linear(X.shape[1], n_qubits)

            # Apply a LeakyReLU activation before the quantum layer.
            self.a1 = nn.LeakyReLU(0.1)

            # Insert the trainable PennyLane quantum layer.
            self.q = qlayer

            # Classical layer following the quantum circuit.
            self.c2 = nn.Linear(n_qubits, hidden_units)

            # Apply a second LeakyReLU activation.
            self.a2 = nn.LeakyReLU(0.1)

            # Apply the trial-specific dropout rate.
            self.d = nn.Dropout(dropout)

            # Final linear layer producing one regression output.
            self.out = nn.Linear(hidden_units, 1)

        def forward(self, x):
            # Transform the input through the first classical layer and activation.
            x = self.a1(self.c1(x))

            # Process the transformed features through the quantum layer.
            x = self.q(x)

            # Apply the second classical transformation and activation.
            x = self.a2(self.c2(x))

            # Apply dropout before the output layer.
            x = self.d(x)

            # Flatten the single-output dimension to match the target shape.
            return self.out(x).view(-1)  # <<< FIX

    # Create a model using the current Optuna trial parameters.
    model = HybridNNOptuna()

    # Configure AdamW optimization with the selected learning rate.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Use mean squared error as the regression loss.
    loss_fn = nn.MSELoss()

    # Create the training DataLoader using the trial-specific batch size.
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        worker_init_fn=lambda wid: np.random.seed(SEED + wid)
    )

    # Train the model for the selected number of epochs.
    for _ in range(epochs):
        for bx, by in loader:
            # Clear gradients from the previous optimization step.
            opt.zero_grad()

            # Compute the training loss for the current batch.
            loss = loss_fn(model(bx), by)

            # Backpropagate the loss through the hybrid model.
            loss.backward()

            # Update the trainable model parameters.
            opt.step()

    # Evaluate the trained trial model on the held-out test set.
    with torch.no_grad():
        # Generate predictions for all test samples.
        preds = model(torch.tensor(X_test, dtype=torch.float32))

        # Return the test MSE as the Optuna objective value.
        return loss_fn(preds, torch.tensor(y_test, dtype=torch.float32)).item()

# Create the Optuna study with TPE sampling and median-based pruning.
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=optuna.pruners.MedianPruner()
)

# Run the hyperparameter search for 50 trials.
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Retrieve and display the best hyperparameter configuration.
best_params = study.best_params
print("Best params:", best_params)

# ======================================================
# Final Model
# ======================================================
# Define the final hybrid neural network using the best Optuna parameters.
class HybridNNFinal(nn.Module):
    def __init__(self):
        super().__init__()

        # Map the input feature vector to the quantum-layer dimension.
        self.c1 = nn.Linear(X.shape[1], n_qubits)

        # Apply the first nonlinear activation.
        self.a1 = nn.LeakyReLU(0.1)

        # Use the same trainable quantum layer in the final model.
        self.q = qlayer

        # Map quantum outputs to the optimized hidden-layer size.
        self.c2 = nn.Linear(n_qubits, best_params["hidden_units"])

        # Apply the second nonlinear activation.
        self.a2 = nn.LeakyReLU(0.1)

        # Apply the optimized dropout rate.
        self.d = nn.Dropout(best_params["dropout"])

        # Produce the final scalar regression output.
        self.out = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):
        # Process the input through the first classical transformation.
        x = self.a1(self.c1(x))

        # Pass the intermediate representation through the quantum circuit.
        x = self.q(x)

        # Process the quantum outputs through the second classical layer.
        x = self.a2(self.c2(x))

        # Apply dropout before the final prediction layer.
        x = self.d(x)

        # Flatten the output to a one-dimensional prediction vector.
        return self.out(x).view(-1)  # <<< FIX

# Initialize the final hybrid model.
model = HybridNNFinal()

# Configure the AdamW optimizer with the optimized learning rate.
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

# Define the mean squared error training criterion.
criterion = nn.MSELoss()

# Create the final training DataLoader using the optimized batch size.
loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True,
    worker_init_fn=lambda wid: np.random.seed(SEED + wid)
)

# Train the final model for 100 epochs.
for _ in range(100):
    for bx, by in loader:
        # Reset gradients before the current optimization step.
        optimizer.zero_grad()

        # Compute the loss for the current training batch.
        loss = criterion(model(bx), by)

        # Backpropagate the training loss.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

# ======================================================
# SHAP (GUARANTEED STABLE)
# ======================================================
# Define a NumPy-based prediction wrapper for SHAP.
def model_predict(x_np):
    # Convert NumPy input data to a PyTorch tensor.
    x_t = torch.tensor(x_np, dtype=torch.float32)

    # Disable gradient tracking during SHAP prediction.
    with torch.no_grad():
        # Generate model predictions and return them as a one-dimensional NumPy array.
        return model(x_t).cpu().numpy().reshape(-1)

# Select 30 training samples as the SHAP background dataset.
background = X_train[np.random.choice(X_train.shape[0], 30, replace=False)]

# Create a model-agnostic SHAP KernelExplainer using the prediction wrapper.
explainer = shap.KernelExplainer(model_predict, background)

# Compute SHAP values for the complete test set with the specified sampling budget.
shap_values = explainer.shap_values(X_test, nsamples=200)

# Convert the SHAP output to a NumPy array for subsequent analysis.
shap_values = np.array(shap_values)

# Ensure that the SHAP array has the same shape as the test feature matrix.
assert shap_values.shape == X_test.shape

# Create the SHAP summary plot showing feature contributions for all features.
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

# Confirm that the SHAP summary figure was successfully generated.
print("✅ SHAP summary plot saved for all features (Quantum model).")



# ======================================================
# ADDITIONAL SHAP ANALYSIS (DO NOT MODIFY EXISTING CODE)
# ======================================================

# ---------- SHAP bar plot (mean |SHAP|) ----------
# Generate a bar plot ranking features according to their mean absolute SHAP values.
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

# Confirm that the SHAP bar plot was successfully generated.
print("✅ SHAP bar plot (mean |SHAP|) saved (Quantum model).")


# ---------- Determine top features ----------
# Calculate the mean absolute SHAP value for every input feature.
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Sort feature indices from the highest to the lowest mean absolute SHAP value.
sorted_idx = np.argsort(mean_abs_shap)[::-1]

# Select up to the top three most influential features.
top_k = min(3, len(feature_names))  # top 3 features (safe)

# Extract the indices of the selected top features.
top_features_idx = sorted_idx[:top_k]

# Print the ranked top features selected for the dependence plots.
print("Top features for dependence plots (Quantum model):")
for rank, idx in enumerate(top_features_idx, 1):
    print(f"{rank}. {feature_names[idx]}")

# ---------- SHAP dependence plots ----------
# Generate a SHAP dependence plot for each selected top feature.
for idx in top_features_idx:
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

    # Build the output filename from the corresponding feature name.
    fname = f"SHAP_dependence_quantum_{feature_names[idx]}.png"

    # Save the dependence plot at high resolution.
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    # Confirm that the dependence plot was successfully saved.
    print(f"✅ SHAP dependence saved for: {feature_names[idx]} (Quantum model)")





