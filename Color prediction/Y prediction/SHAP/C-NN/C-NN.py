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
import shap

# ======================================================
#  ███   Full Deterministic Reproducibility Setup   ███
# ======================================================

# Define a fixed random seed to make the experiment reproducible.
SEED = 42

# Force deterministic hashing and CUDA workspace behavior.
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Set the random seeds for Python, NumPy, and PyTorch.
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Enable deterministic PyTorch algorithms and deterministic cuDNN behavior.
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------
# Data
# ------------------------------

# Load the shuffled dataset from the CSV file.
dataset = pd.read_csv("dataset_shuffle.csv")

# Define columns that should not be used as input features.
drop_cols = ["Structure", "Time (h)", "O", "L", "a", "b", "y", "Reference"]

# Extract the names of the remaining input features.
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

# Extract input features and target values as NumPy arrays.
X = dataset.drop(drop_cols, axis=1).to_numpy()
y = dataset["y"].to_numpy()

# ------------------------------
# Normalization
# ------------------------------

# Scale the input features to the range [0, pi].
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale the target values to the range [-1, 1].
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split the normalized data into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Convert the training data into a PyTorch TensorDataset.
train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32)
)

# ------------------------------
# Classical Quantum-like Layer
# ------------------------------

# Use the number of input features as the number of quantum-like units.
n_qubits = X.shape[1]

# Define the depth of the classical quantum-like transformation.
n_layers = 4

# Define a fully classical neural layer that mimics repeated
# parameterized quantum-style transformations.
class ClassicalQuantumLayer(nn.Module):
    def __init__(self, n_qubits, n_layers):
        super().__init__()
        layers = []

        # Build multiple Linear + Tanh blocks.
        for _ in range(n_layers):
            layers.append(nn.Linear(n_qubits, n_qubits))
            layers.append(nn.Tanh())

        # Combine all blocks into a sequential network.
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Apply the complete quantum-like classical transformation.
        return self.net(x)

# Instantiate the shared classical quantum-like layer.
clayer = ClassicalQuantumLayer(n_qubits, n_layers)

# ======================================================
# Optuna
# ======================================================

# Create a deterministic Optuna TPE sampler.
sampler = optuna.samplers.TPESampler(seed=SEED)

# Create an optimization study that minimizes the validation/test loss.
study = optuna.create_study(direction="minimize", sampler=sampler)

# Define the Optuna objective function.
def objective(trial):

    # Suggest candidate learning rates.
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])

    # Suggest candidate dropout rates.
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])

    # Suggest the number of hidden units.
    hidden_units = trial.suggest_categorical(
        "hidden_units", [8, 9, 10, 11, 12, 13, 14, 15, 16, 32]
    )

    # Suggest the mini-batch size.
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Suggest the number of training epochs.
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Define the neural network used during hyperparameter optimization.
    class HybridNNOptuna(nn.Module):
        def __init__(self):
            super().__init__()

            # First classical projection layer.
            self.c1 = nn.Linear(X.shape[1], n_qubits)

            # First nonlinear activation.
            self.a1 = nn.LeakyReLU(0.1)

            # Insert the classical quantum-like layer.
            self.q = clayer

            # Project the quantum-like representation into the hidden layer.
            self.c2 = nn.Linear(n_qubits, hidden_units)

            # Hidden-layer activation.
            self.a2 = nn.LeakyReLU(0.1)

            # Apply the trial-specific dropout rate.
            self.d = nn.Dropout(dropout)

            # Final regression output layer.
            self.out = nn.Linear(hidden_units, 1)

        def forward(self, x):

            # Apply the first linear transformation and activation.
            x = self.a1(self.c1(x))

            # Apply the quantum-like classical transformation.
            x = self.q(x)

            # Transform into the hidden representation.
            x = self.a2(self.c2(x))

            # Apply dropout regularization.
            x = self.d(x)

            # Produce the scalar regression output.
            return self.out(x).view(-1)  # <<< FIX

    # Initialize the model for the current Optuna trial.
    model = HybridNNOptuna()

    # Use AdamW as the optimizer with the trial-specific learning rate.
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    # Use mean squared error as the optimization objective.
    loss_fn = nn.MSELoss()

    # Create a DataLoader using the trial-specific batch size.
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Train the model for the selected number of epochs.
    for _ in range(epochs):
        for bx, by in loader:
            # Clear the previous gradients.
            opt.zero_grad()

            # Compute the training loss.
            loss = loss_fn(model(bx), by)

            # Backpropagate the loss.
            loss.backward()

            # Update model parameters.
            opt.step()

    # Evaluate the current trial on the held-out test set.
    with torch.no_grad():
        preds = model(torch.tensor(X_test, dtype=torch.float32))

        # Return the test MSE to Optuna for minimization.
        return loss_fn(preds, torch.tensor(y_test, dtype=torch.float32)).item()

# Run 50 Optuna trials and display progress.
study.optimize(objective, n_trials=50, show_progress_bar=True)

# Retrieve the best hyperparameter configuration.
best_params = study.best_params

# ------------------------------
# Final Model
# ------------------------------

# Define the final neural network using the optimized hyperparameters.
class HybridNNFinal(nn.Module):
    def __init__(self):
        super().__init__()

        # First classical projection layer.
        self.c1 = nn.Linear(X.shape[1], n_qubits)

        # First nonlinear activation.
        self.a1 = nn.LeakyReLU(0.1)

        # Reuse the classical quantum-like layer.
        self.q = clayer

        # Hidden layer using the optimized number of units.
        self.c2 = nn.Linear(n_qubits, best_params["hidden_units"])

        # Hidden-layer activation.
        self.a2 = nn.LeakyReLU(0.1)

        # Dropout using the optimized dropout rate.
        self.d = nn.Dropout(best_params["dropout"])

        # Final regression output layer.
        self.out = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):

        # Apply the first classical transformation.
        x = self.a1(self.c1(x))

        # Apply the quantum-like classical transformation.
        x = self.q(x)

        # Apply the hidden layer and nonlinear activation.
        x = self.a2(self.c2(x))

        # Apply dropout regularization.
        x = self.d(x)

        # Return a one-dimensional regression output.
        return self.out(x).view(-1)  # <<< FIX

# Initialize the final model.
model = HybridNNFinal()

# Create the final AdamW optimizer using the optimized learning rate.
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

# Define the final mean squared error loss function.
criterion = nn.MSELoss()

# Create the final training DataLoader using the optimized batch size.
loader = DataLoader(
    train_dataset,
    batch_size=best_params["batch_size"],
    shuffle=True
)

# Train the final model for 100 epochs.
for _ in range(100):
    for bx, by in loader:

        # Clear the accumulated gradients.
        optimizer.zero_grad()

        # Compute the training loss.
        loss = criterion(model(bx), by)

        # Backpropagate the loss.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

# ======================================================
# SHAP – GUARANTEED STABLE
# ======================================================

# Define a prediction wrapper compatible with SHAP.
def model_predict(x_np):

    # Convert NumPy input into a PyTorch tensor.
    x_t = torch.tensor(x_np, dtype=torch.float32)

    # Disable gradient tracking during SHAP predictions.
    with torch.no_grad():

        # Generate model predictions and convert them back to NumPy.
        return model(x_t).cpu().numpy().reshape(-1)

# Randomly select 30 training samples as the SHAP background dataset.
background = X_train[np.random.choice(X_train.shape[0], 30, replace=False)]

# Create a model-agnostic SHAP KernelExplainer.
explainer = shap.KernelExplainer(model_predict, background)

# Calculate SHAP values for the complete test set.
shap_values = explainer.shap_values(X_test, nsamples=200)

# Convert the SHAP output into a NumPy array.
shap_values = np.array(shap_values)

# Verify that the SHAP matrix has the same shape as the test feature matrix.
assert shap_values.shape == X_test.shape

# Create the SHAP summary plot.
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)

# Adjust the figure layout and save the plot at high resolution.
plt.tight_layout()
plt.savefig("SHAP_summary_all_features.png", dpi=300, bbox_inches="tight")
plt.close()

# Confirm that the summary plot was successfully generated.
print("✅ SHAP summary plot saved for all features.")

# ======================================================
# ADDITIONAL SHAP ANALYSIS (DO NOT MODIFY EXISTING CODE)
# ======================================================

# ---------- SHAP bar plot (mean |SHAP|) ----------

# Create a bar plot showing the mean absolute SHAP importance.
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

# Adjust the layout and save the SHAP feature-importance bar plot.
plt.tight_layout()
plt.savefig("SHAP_bar_mean_abs.png", dpi=300, bbox_inches="tight")
plt.close()

# Confirm that the SHAP bar plot was successfully generated.
print("✅ SHAP bar plot (mean |SHAP|) saved.")


# ---------- Determine top features ----------

# Calculate the mean absolute SHAP value for every feature.
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Sort feature indices from highest to lowest SHAP importance.
sorted_idx = np.argsort(mean_abs_shap)[::-1]

# Select up to the three most important features.
top_k = min(3, len(feature_names))  # top 3 features (safe)

# Extract the indices of the top-ranked features.
top_features_idx = sorted_idx[:top_k]

# Display the features selected for the dependence plots.
print("Top features for dependence plots:")

# Print the rank and feature name for each selected feature.
for rank, idx in enumerate(top_features_idx, 1):
    print(f"{rank}. {feature_names[idx]}")

# ---------- SHAP dependence plots ----------

# Generate one SHAP dependence plot for each top feature.
for idx in top_features_idx:

    # Create a new figure for the current feature.
    plt.figure(figsize=(7, 5))

    # Plot the relationship between feature values and their SHAP contributions.
    shap.dependence_plot(
        idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False,
        interaction_index="auto"
    )

    # Adjust the figure layout.
    plt.tight_layout()

    # Construct the output filename using the feature name.
    fname = f"SHAP_dependence_{feature_names[idx]}.png"

    # Save the dependence plot at high resolution.
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()

    # Confirm that the dependence plot was saved.
    print(f"✅ SHAP dependence saved for: {feature_names[idx]}")





