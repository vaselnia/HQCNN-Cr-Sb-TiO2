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
# Reproducibility
# ------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set Optuna logging level to warnings only.
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Reset the random seeds for reproducibility.
np.random.seed(SEED)
random.seed(SEED)

# ------------------------------
# Data
# ------------------------------

# Load the shuffled dataset from the CSV file.
dataset = pd.read_csv("dataset_shuffle.csv")

# Define columns that should be excluded from the input features.
drop_cols = ["Structure", "Time (h)", "O", "L", "a", "b", "y", "Reference"]

# Extract the names of the remaining input features.
feature_names = dataset.drop(drop_cols, axis=1).columns.tolist()

# Extract input features and target values.
X = dataset.drop(drop_cols, axis=1).to_numpy()
y = dataset["y"].to_numpy()

# ------------------------------
# Scaling
# ------------------------------

# Scale the input features to the range [0, pi].
scaler_X = MinMaxScaler(feature_range=(0, np.pi))
X_scaled = scaler_X.fit_transform(X)

# Scale the target variable to the range [-1, 1].
scaler_y = MinMaxScaler(feature_range=(-1, 1))
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

# Split the scaled data into training and testing subsets.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_scaled, test_size=0.2, random_state=SEED
)

# Create a TensorDataset for the training data.
train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                              torch.tensor(y_train, dtype=torch.float32))

# Create a TensorDataset for the test data.
test_dataset = TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                             torch.tensor(y_test, dtype=torch.float32))

# ------------------------------
# Quantum Layer
# ------------------------------

# Set the number of qubits equal to the number of input features.
n_qubits = X.shape[1]

# Define the number of repeated quantum layers.
n_layers = 4

# Create a deterministic default.qubit simulator.
dev = qml.device("default.qubit", wires=n_qubits, seed=SEED)

# Define the QAOA-like all-to-all quantum circuit.
def qaoa_all_to_all(inputs, enc_scale, enc_shift, gamma, delta, beta):

    # Encode each input feature using a parameterized RY rotation.
    for i in range(n_qubits):
        qml.RY(inputs[i] * enc_scale[i] + enc_shift[i], wires=i)

    # Apply repeated entangling and parameterized rotation layers.
    for l in range(n_layers):

        # Create all-to-all CNOT entanglement between the qubits.
        for i in range(n_qubits):
            for j in range(i+1, n_qubits):
                qml.CNOT(wires=[i, j])

        # Apply trainable RX, RY, and RZ rotations to every qubit.
        for i in range(n_qubits):
            qml.RX(gamma[l, i], wires=i)
            qml.RY(delta[l, i], wires=i)
            qml.RZ(beta[l, i], wires=i)

    # Return the Pauli-Z expectation value of every qubit.
    return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

# Define the shapes of all trainable quantum parameters.
weight_shapes = {
    "enc_scale": n_qubits,
    "enc_shift": n_qubits,
    "gamma": (n_layers, n_qubits),
    "delta": (n_layers, n_qubits),
    "beta": (n_layers, n_qubits)
}

# Create the trainable QNode using the PyTorch interface.
qnode = qml.QNode(qaoa_all_to_all, dev, interface="torch", diff_method="backprop")

# Convert the QNode into a PyTorch-compatible trainable layer.
qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

# ----------- Circuit for Plotting -----------

# Create a separate QNode for visualizing the quantum circuit.
qnode_for_plot = qml.QNode(qaoa_all_to_all, dev)

# Use a zero-valued sample input for circuit visualization.
sample_input = np.zeros(n_qubits)

# Reset the NumPy random seed before generating sample quantum parameters.
np.random.seed(SEED)

# Generate deterministic random parameters for circuit visualization.
weights_sample = {
    "enc_scale": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "enc_shift": np.random.uniform(0, 2*np.pi, size=n_qubits),
    "gamma": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits)),
    "delta": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits)),
    "beta": np.random.uniform(0, 2*np.pi, size=(n_layers, n_qubits))
}

# Draw and save the high-level, undecomposed circuit diagram.
fig1, ax1 = qml.draw_mpl(qnode_for_plot, level="top")(sample_input, **weights_sample)
fig1.savefig("quantum_circuit_undecomposed_300dpi.png", dpi=300, bbox_inches="tight")

# Draw and save the device-level decomposed circuit diagram.
fig2, ax2 = qml.draw_mpl(qnode_for_plot, level="device")(sample_input, **weights_sample)
fig2.savefig("quantum_circuit_decomposed_300dpi.png", dpi=300, bbox_inches="tight")

# Display the generated circuit figures.
plt.show()

# ============================================================
# Optuna
# ============================================================

# Define the Optuna objective function for hyperparameter optimization.
def objective(trial):

    # Suggest the learning rate.
    lr = trial.suggest_categorical("lr", [0.0005, 0.001, 0.003])

    # Suggest the dropout rate.
    dropout = trial.suggest_categorical("dropout", [0.0, 0.05, 0.1])

    # Suggest the number of hidden units.
    hidden_units = trial.suggest_categorical("hidden_units", [8, 16, 32])

    # Suggest the training batch size.
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])

    # Suggest the number of training epochs.
    epochs = trial.suggest_categorical("epochs", [15, 20, 25])

    # Define the hybrid classical-quantum model used during optimization.
    class HybridNNOptuna(nn.Module):

        def __init__(self):
            super().__init__()

            # First classical linear transformation.
            self.classical1 = nn.Linear(X.shape[1], n_qubits)

            # First nonlinear activation function.
            self.act1 = nn.LeakyReLU(0.1)

            # Trainable quantum layer.
            self.quantum = qlayer

            # Second classical transformation.
            self.classical2 = nn.Linear(n_qubits, hidden_units)

            # Second nonlinear activation function.
            self.act2 = nn.LeakyReLU(0.1)

            # Dropout regularization layer.
            self.dropout = nn.Dropout(dropout)

            # Final single-output regression layer.
            self.classical3 = nn.Linear(hidden_units, 1)

        def forward(self, x):

            # Apply the first classical layer and activation.
            x = self.act1(self.classical1(x))

            # Apply the quantum layer to either a single sample or a batch.
            if x.ndim == 1:
                x = self.quantum(x)
            else:
                x = torch.stack([self.quantum(xi) for xi in x])

            # Apply the second classical layer and activation.
            x = self.act2(self.classical2(x))

            # Apply dropout regularization.
            x = self.dropout(x)

            # Generate the final regression output.
            x = self.classical3(x)

            # Flatten the output to one value per sample.
            return x.view(-1)

    # Initialize the model for the current Optuna trial.
    model = HybridNNOptuna()

    # Create the AdamW optimizer using the selected learning rate.
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Use mean squared error as the training criterion.
    criterion = nn.MSELoss()

    # Create the training DataLoader using the selected batch size.
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                        worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))

    # Train the model for the selected number of epochs.
    for epoch in range(epochs):

        # Set the model to training mode.
        model.train()

        # Iterate over the training batches.
        for batch_X, batch_y in loader:

            # Clear the accumulated gradients.
            optimizer.zero_grad()

            # Generate predictions for the current batch.
            outputs = model(batch_X)

            # Calculate the training loss.
            loss = criterion(outputs, batch_y)

            # Backpropagate the loss.
            loss.backward()

            # Update the model parameters.
            optimizer.step()

        # Switch the model to evaluation mode for validation.
        model.eval()

        # Evaluate the model without calculating gradients.
        with torch.no_grad():

            # Generate predictions for the test set.
            preds = model(torch.tensor(X_test, dtype=torch.float32))

            # Calculate the test/validation loss.
            val_loss = criterion(preds, torch.tensor(y_test, dtype=torch.float32)).item()

        # Report the validation loss to Optuna for the current epoch.
        trial.report(val_loss, step=epoch)

        # Stop the trial early if the pruning criterion is satisfied.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    # Return the final validation loss for the trial.
    return val_loss


# Create the Optuna study using TPE sampling and median pruning.
study = optuna.create_study(
    direction="minimize",
    pruner=optuna.pruners.MedianPruner(),
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

# Run 50 hyperparameter optimization trials.
study.optimize(objective, n_trials=50, show_progress_bar=True)

# ============================================================
# Final Model
# ============================================================

# Retrieve the best hyperparameters identified by Optuna.
best_params = study.best_params

# Define the number of epochs used to train the final model.
final_epochs = 100


# Define the final hybrid classical-quantum neural network.
class HybridNNFinal(nn.Module):

    def __init__(self):

        super().__init__()

        # First classical linear transformation.
        self.classical1 = nn.Linear(X.shape[1], n_qubits)

        # First nonlinear activation function.
        self.act1 = nn.LeakyReLU(0.1)

        # Trainable quantum layer.
        self.quantum = qlayer

        # Second classical layer using the optimized hidden size.
        self.classical2 = nn.Linear(n_qubits, best_params["hidden_units"])

        # Second nonlinear activation function.
        self.act2 = nn.LeakyReLU(0.1)

        # Dropout layer using the optimized dropout rate.
        self.dropout = nn.Dropout(best_params["dropout"])

        # Final regression output layer.
        self.classical3 = nn.Linear(best_params["hidden_units"], 1)

    def forward(self, x):

        # Apply the first classical layer and activation.
        x = self.act1(self.classical1(x))

        # Apply the quantum layer to a single sample or each sample in a batch.
        if x.ndim == 1:
            x = self.quantum(x)
        else:
            x = torch.stack([self.quantum(xi) for xi in x])

        # Apply the second classical layer and activation.
        x = self.act2(self.classical2(x))

        # Apply dropout regularization.
        x = self.dropout(x)

        # Generate the final regression output.
        x = self.classical3(x)

        # Flatten the output to one value per sample.
        return x.view(-1)


# Initialize the final hybrid model.
model = HybridNNFinal()

# Create the final AdamW optimizer using the optimized learning rate.
optimizer = torch.optim.AdamW(model.parameters(), lr=best_params["lr"])

# Define the final mean squared error criterion.
criterion = nn.MSELoss()

# Create the training DataLoader using the optimized batch size.
train_loader = DataLoader(train_dataset, batch_size=best_params["batch_size"], shuffle=True,
                          worker_init_fn=lambda worker_id: np.random.seed(SEED + worker_id))


# Train the final model for the specified number of epochs.
for epoch in range(final_epochs):

    # Set the model to training mode.
    model.train()

    # Iterate over the training batches.
    for batch_X, batch_y in train_loader:

        # Clear the previous gradients.
        optimizer.zero_grad()

        # Generate predictions for the current batch.
        outputs = model(batch_X)

        # Calculate the training loss.
        loss = criterion(outputs, batch_y)

        # Backpropagate the loss.
        loss.backward()

        # Update the model parameters.
        optimizer.step()

    # Switch the model to evaluation mode.
    model.eval()

    # Evaluate the model on the test set without gradient calculation.
    with torch.no_grad():

        # Generate predictions for the test set.
        test_preds = model(torch.tensor(X_test, dtype=torch.float32))

        # Calculate the test loss.
        test_loss = criterion(test_preds, torch.tensor(y_test, dtype=torch.float32))

    # Print training and test losses every 10 epochs.
    if (epoch+1) % 10 == 0:

        print(f"Epoch {epoch+1}/{final_epochs}, Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")

# ------------------------------
# Evaluation
# ------------------------------

# Generate scaled predictions for the test set.
y_pred_scaled = model(torch.tensor(X_test, dtype=torch.float32)).detach().numpy()

# Transform the predictions back to the original target scale.
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

# Transform the true test targets back to the original scale.
y_true = scaler_y.inverse_transform(y_test.reshape(-1, 1)).ravel()

# Calculate the mean squared error in the original target scale.
mse = mean_squared_error(y_true, y_pred)

# Calculate the mean absolute error in the original target scale.
mae = mean_absolute_error(y_true, y_pred)

# Calculate the coefficient of determination.
r2 = r2_score(y_true, y_pred)

# Display the calculated performance metrics.
print("\nPerformance Metrics:")

# Print the mean squared error.
print(f"MSE: {mse:.4f}")

# Print the mean absolute error.
print(f"MAE: {mae:.4f}")

# Print the R-squared score.
print(f"R²: {r2:.4f}")

# ------------------------------
# Prediction Plot
# ------------------------------

# Create a square figure for the true-versus-predicted plot.
fig, ax = plt.subplots(figsize=(10, 10))

# Plot predicted values against true values.
ax.plot(y_true, y_pred, 'o', color='magenta')

# Determine the minimum value across true and predicted values.
min_val = min(min(y_true), min(y_pred))

# Determine the maximum value across true and predicted values.
max_val = max(max(y_true), max(y_pred))

# Set equal limits for both axes.
ax.set_xlim(min_val, max_val)
ax.set_ylim(min_val, max_val)

# Add the ideal y = x reference line.
ax.plot([min_val, max_val], [min_val, max_val], linestyle='--', color='black', lw=2)

# Define the tick spacing.
tick_step = 0.1

# Set x-axis tick positions.
ax.set_xticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))

# Set y-axis tick positions.
ax.set_yticks(np.arange(np.floor(min_val), np.ceil(max_val) + tick_step, tick_step))

# Label the x-axis with the true band-gap values.
ax.set_xlabel('True $E_{g}$ (eV)')

# Label the y-axis with the predicted band-gap values.
ax.set_ylabel('Predicted $E_{g}$ (eV)')

# Keep the plot aspect ratio equal.
ax.set_aspect('equal', adjustable='box')

# Apply the final figure layout.
fig.tight_layout()

# Display the prediction plot.
plt.show()

# Save the prediction plot at 300 DPI.
fig.savefig('Hybrid_NN_QAOA.png', dpi=300, bbox_inches='tight')

# ======================================================
# SHAP
# ======================================================

# Define a prediction wrapper compatible with SHAP.
def model_predict(x_np):

    # Convert the NumPy input array into a PyTorch tensor.
    x_tensor = torch.tensor(x_np, dtype=torch.float32)

    # Disable gradient tracking during SHAP prediction.
    with torch.no_grad():

        # Generate model predictions.
        preds = model(x_tensor)

    # Convert predictions to a one-dimensional NumPy array.
    return preds.numpy().reshape(-1)


# Randomly select 30 training samples as the SHAP background dataset.
background = X_train[np.random.choice(X_train.shape[0], 30, replace=False)]

# Create a model-agnostic SHAP KernelExplainer.
explainer = shap.KernelExplainer(model_predict, background)

# Calculate SHAP values for the complete test set.
shap_values = explainer.shap_values(X_test, nsamples=200)

# Convert the SHAP output into a NumPy array.
shap_values = np.array(shap_values)

# Create the SHAP summary plot for all features.
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    show=False
)

# Adjust the plot layout.
plt.tight_layout()

# Save the SHAP summary plot at high resolution.
plt.savefig("SHAP_summary_all_features_quantum.png", dpi=300, bbox_inches="tight")

# Close the current figure.
plt.close()

# Create the SHAP mean absolute importance bar plot.
plt.figure(figsize=(10,6))

shap.summary_plot(
    shap_values,
    X_test,
    feature_names=feature_names,
    plot_type="bar",
    show=False
)

# Adjust the plot layout.
plt.tight_layout()

# Save the SHAP bar plot at high resolution.
plt.savefig("SHAP_bar_mean_abs_quantum.png", dpi=300, bbox_inches="tight")

# Close the current figure.
plt.close()

# Calculate the mean absolute SHAP value for each feature.
mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

# Sort feature indices from highest to lowest SHAP importance.
sorted_idx = np.argsort(mean_abs_shap)[::-1]

# Select the three most important features.
top_features = sorted_idx[:3]

# Generate a SHAP dependence plot for each top feature.
for idx in top_features:

    # Create a new figure for the current feature.
    plt.figure(figsize=(7,5))

    # Generate the SHAP dependence plot.
    shap.dependence_plot(
        idx,
        shap_values,
        X_test,
        feature_names=feature_names,
        show=False
    )

    # Adjust the plot layout.
    plt.tight_layout()

    # Save the dependence plot using the feature name.
    plt.savefig(f"SHAP_dependence_quantum_{feature_names[idx]}.png", dpi=300, bbox_inches="tight")

    # Close the current figure.
    plt.close()





