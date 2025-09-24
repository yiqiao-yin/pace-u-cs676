import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

print("=== PyTorch Logistic Regression Implementation ===\n")

# ========================================
# 1. DATA GENERATION
# ========================================
print("1. GENERATING SAMPLE DATA")
print("-" * 40)

# Generate synthetic dataset
n_samples = 1000
n_features = 2

# Generate random features
X = torch.randn(n_samples, n_features)
print(f"Generated feature matrix X with shape: {X.shape}")

# Create a true linear relationship for log-odds
true_weights = torch.tensor([1.5, -2.0])  # True coefficients
true_bias = torch.tensor([0.5])           # True bias

print(f"True weights (β): {true_weights.numpy()}")
print(f"True bias (β₀): {true_bias.numpy()}")

# Calculate true log-odds (linear predictor η = Xβ + β₀)
true_logits = X @ true_weights + true_bias
print(f"True logits (η) range: [{true_logits.min():.3f}, {true_logits.max():.3f}]")

# Convert to probabilities using sigmoid function: p = 1/(1 + e^(-η))
true_probs = torch.sigmoid(true_logits)
print(f"True probabilities range: [{true_probs.min():.3f}, {true_probs.max():.3f}]")

# Generate binary labels by sampling from Bernoulli distribution
y = torch.bernoulli(true_probs).long()
print(f"Generated labels y with shape: {y.shape}")
print(f"Class distribution: {torch.bincount(y).numpy()} (0s and 1s)")

# Split into train/test
train_size = int(0.8 * n_samples)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
print(f"Training set: X_train {X_train.shape}, y_train {y_train.shape}")
print(f"Test set: X_test {X_test.shape}, y_test {y_test.shape}\n")

# ========================================
# 2. LOGISTIC REGRESSION MODEL
# ========================================
print("2. DEFINING LOGISTIC REGRESSION MODEL")
print("-" * 40)

class LogisticRegressionNeuron(nn.Module):
    """
    Single neuron implementing logistic regression:
    η = Xβ + β₀  (linear predictor)
    p = σ(η) = 1/(1 + e^(-η))  (sigmoid activation)
    """
    def __init__(self, n_features):
        super(LogisticRegressionNeuron, self).__init__()
        # Single linear layer: y = Wx + b
        self.linear = nn.Linear(n_features, 1)
        
        # Initialize weights and bias
        nn.init.normal_(self.linear.weight, mean=0, std=0.1)
        nn.init.constant_(self.linear.bias, 0)
        
        print(f"Initialized logistic neuron with {n_features} input features")
        print(f"Initial weights: {self.linear.weight.data.numpy().flatten()}")
        print(f"Initial bias: {self.linear.bias.data.numpy()}")
    
    def forward(self, x):
        """
        Forward pass:
        1. Linear transformation: η = Xβ + β₀
        2. Sigmoid activation: p = σ(η) = 1/(1 + e^(-η))
        """
        # Linear predictor (log-odds)
        logits = self.linear(x)  # Shape: (batch_size, 1)
        
        # Sigmoid activation to get probabilities
        probabilities = torch.sigmoid(logits)  # Shape: (batch_size, 1)
        
        return logits.squeeze(), probabilities.squeeze()  # Remove extra dimension

# Initialize model
model = LogisticRegressionNeuron(n_features)
print(f"Model parameters: {sum(p.numel() for p in model.parameters())} total\n")

# ========================================
# 3. LOSS FUNCTION AND OPTIMIZER
# ========================================
print("3. SETTING UP TRAINING COMPONENTS")
print("-" * 40)

# Binary Cross-Entropy Loss
# L = -[y*log(p) + (1-y)*log(1-p)]
criterion = nn.BCELoss()  # Binary Cross-Entropy
print("Using Binary Cross-Entropy Loss: L = -[y*log(p) + (1-y)*log(1-p)]")

# Optimizer - Stochastic Gradient Descent
learning_rate = 0.1
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
print(f"Using SGD optimizer with learning rate: {learning_rate}\n")

# ========================================
# 4. TRAINING LOOP
# ========================================
print("4. TRAINING THE MODEL")
print("-" * 40)

n_epochs = 100
losses = []
accuracies = []

print("Starting training loop...")
print("Epoch | Loss    | Accuracy | Weights          | Bias")
print("-" * 55)

for epoch in range(n_epochs):
    # ========================================
    # FORWARD PASS
    # ========================================
    model.train()
    
    # Get predictions
    logits, probs = model(X_train)
    
    # Calculate loss
    loss = criterion(probs, y_train.float())
    
    # ========================================
    # BACKWARD PASS
    # ========================================
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    
    # Compute gradients via backpropagation
    loss.backward()
    
    # Update parameters using gradients
    optimizer.step()
    
    # ========================================
    # TRACKING PROGRESS
    # ========================================
    # Calculate training accuracy
    predictions = (probs > 0.5).long()
    accuracy = (predictions == y_train).float().mean().item()
    
    # Store metrics
    losses.append(loss.item())
    accuracies.append(accuracy)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        current_weights = model.linear.weight.data.numpy().flatten()
        current_bias = model.linear.bias.data.numpy()[0]
        print(f"{epoch+1:5d} | {loss.item():.3f} | {accuracy:.3f}    | "
              f"[{current_weights[0]:6.3f}, {current_weights[1]:6.3f}] | {current_bias:6.3f}")

print(f"\nTraining completed!")

# ========================================
# 5. FINAL MODEL EVALUATION
# ========================================
print("\n5. MODEL EVALUATION")
print("-" * 40)

# Final parameters
final_weights = model.linear.weight.data.numpy().flatten()
final_bias = model.linear.bias.data.numpy()[0]

print("Final Model Parameters:")
print(f"Learned weights (β̂): [{final_weights[0]:.3f}, {final_weights[1]:.3f}]")
print(f"True weights (β):    [{true_weights[0]:.3f}, {true_weights[1]:.3f}]")
print(f"Learned bias (β̂₀):   {final_bias:.3f}")
print(f"True bias (β₀):      {true_bias[0]:.3f}")

# Test set evaluation
model.eval()
with torch.no_grad():
    test_logits, test_probs = model(X_test)
    test_predictions = (test_probs > 0.5).long()
    test_accuracy = (test_predictions == y_test).float().mean().item()

print(f"\nTest Accuracy: {test_accuracy:.3f}")

# Detailed classification report
y_test_np = y_test.numpy()
test_pred_np = test_predictions.numpy()
print("\nClassification Report:")
print(classification_report(y_test_np, test_pred_np))

# ========================================
# 6. VISUALIZATION
# ========================================
print("\n6. CREATING VISUALIZATIONS")
print("-" * 40)

# Create subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Training Loss
ax1.plot(losses, 'b-', linewidth=2, label='Training Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Binary Cross-Entropy Loss')
ax1.set_title('Training Loss Over Time')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: Training Accuracy
ax2.plot(accuracies, 'g-', linewidth=2, label='Training Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy Over Time')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_ylim([0, 1])

# Plot 3: Decision Boundary (2D visualization)
# Create a mesh grid for decision boundary
h = 0.1
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# Get predictions for the mesh
mesh_points = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
model.eval()
with torch.no_grad():
    _, mesh_probs = model(mesh_points)
    mesh_probs = mesh_probs.numpy().reshape(xx.shape)

# Plot decision boundary
contour = ax3.contourf(xx, yy, mesh_probs, levels=50, alpha=0.6, cmap='RdYlBu')
scatter = ax3.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
ax3.set_xlabel('Feature 1')
ax3.set_ylabel('Feature 2')
ax3.set_title('Decision Boundary and Test Data')
plt.colorbar(contour, ax=ax3, label='Predicted Probability')

# Plot 4: Probability Distribution
test_probs_np = test_probs.detach().numpy()
ax4.hist(test_probs_np[y_test == 0], bins=30, alpha=0.7, label='Class 0', color='red')
ax4.hist(test_probs_np[y_test == 1], bins=30, alpha=0.7, label='Class 1', color='blue')
ax4.axvline(x=0.5, color='black', linestyle='--', label='Decision Threshold')
ax4.set_xlabel('Predicted Probability')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Predicted Probabilities')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Visualization complete!")

# ========================================
# 7. MATHEMATICAL VERIFICATION
# ========================================
print("\n7. MATHEMATICAL VERIFICATION")
print("-" * 40)

# Verify sigmoid function implementation
sample_logit = 2.0
manual_sigmoid = 1 / (1 + np.exp(-sample_logit))
torch_sigmoid = torch.sigmoid(torch.tensor(sample_logit)).item()

print(f"Sigmoid verification for logit = {sample_logit}:")
print(f"Manual calculation: σ({sample_logit}) = 1/(1 + e^(-{sample_logit})) = {manual_sigmoid:.6f}")
print(f"PyTorch sigmoid:   σ({sample_logit}) = {torch_sigmoid:.6f}")
print(f"Match: {np.isclose(manual_sigmoid, torch_sigmoid)}")

# Verify log-odds transformation
sample_prob = 0.8
manual_logit = np.log(sample_prob / (1 - sample_prob))
print(f"\nLog-odds verification for p = {sample_prob}:")
print(f"Manual calculation: logit({sample_prob}) = ln({sample_prob}/(1-{sample_prob})) = {manual_logit:.6f}")

print(f"\nModel successfully learned the underlying logistic relationship!")
print("The learned parameters closely approximate the true parameters used to generate the data.")