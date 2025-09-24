import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define true parameters (what we want to recover)
TRUE_BETA_0 = 3.5  # intercept
TRUE_BETA_1 = 2.8  # slope
NOISE_STD = 0.5

# Generate synthetic data
n_samples = 1000
X = torch.randn(n_samples, 1) * 2  # Input features
noise = torch.randn(n_samples, 1) * NOISE_STD
y = TRUE_BETA_0 + TRUE_BETA_1 * X + noise  # y = β₀ + β₁x + ε

print(f"True parameters: β₀ = {TRUE_BETA_0}, β₁ = {TRUE_BETA_1}")
print(f"Generated {n_samples} samples with noise std = {NOISE_STD}")

# Create DataLoader for batch processing
dataset = TensorDataset(X, y)
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define Linear Regression Model
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # 1 input, 1 output
    
    def forward(self, x):
        return self.linear(x)

# Initialize model, loss function, and optimizer
model = LinearRegression()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training parameters
num_epochs = 100
losses = []  # Store loss values for plotting

print(f"\nInitial weights: β₀ = {model.linear.bias.item():.4f}, β₁ = {model.linear.weight.item():.4f}")
print("\nStarting training...")
print("Epoch | Loss")
print("-" * 20)

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0
    num_batches = 0
    
    for batch_X, batch_y in dataloader:
        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)
        
        # Backward pass and optimization
        optimizer.zero_grad()  # Clear gradients
        loss.backward()        # Compute gradients
        optimizer.step()       # Update parameters
        
        epoch_loss += loss.item()
        num_batches += 1
    
    # Calculate average loss for this epoch
    avg_loss = epoch_loss / num_batches
    losses.append(avg_loss)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"{epoch+1:4d}  | {avg_loss:.6f}")

# Get final trained parameters
final_beta_0 = model.linear.bias.item()
final_beta_1 = model.linear.weight.item()

print(f"\nTraining completed!")
print(f"Final weights: β₀ = {final_beta_0:.4f}, β₁ = {final_beta_1:.4f}")
print(f"True weights:  β₀ = {TRUE_BETA_0:.4f}, β₁ = {TRUE_BETA_1:.4f}")
print(f"Difference:    β₀ = {abs(final_beta_0 - TRUE_BETA_0):.4f}, β₁ = {abs(final_beta_1 - TRUE_BETA_1):.4f}")

# Plot the loss curve
plt.figure(figsize=(12, 5))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(losses, 'b-', linewidth=2, label='Training Loss')
plt.title('Training Loss Over Time', fontsize=14, fontweight='bold')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.grid(True, alpha=0.3)
plt.legend()

# Data and fitted line visualization
plt.subplot(1, 2, 2)
X_np = X.numpy()
y_np = y.numpy()

# Plot original data points
plt.scatter(X_np, y_np, alpha=0.5, s=10, label='Data Points')

# Plot true line
x_range = np.linspace(X_np.min(), X_np.max(), 100)
true_line = TRUE_BETA_0 + TRUE_BETA_1 * x_range
plt.plot(x_range, true_line, 'r-', linewidth=3, label=f'True: y = {TRUE_BETA_0} + {TRUE_BETA_1}x')

# Plot fitted line
fitted_line = final_beta_0 + final_beta_1 * x_range
plt.plot(x_range, fitted_line, 'g--', linewidth=3, 
         label=f'Fitted: y = {final_beta_0:.3f} + {final_beta_1:.3f}x')

plt.title('Linear Regression: True vs Fitted', fontsize=14, fontweight='bold')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Calculate R² score for model evaluation
with torch.no_grad():
    y_pred = model(X)
    ss_res = torch.sum((y - y_pred) ** 2)
    ss_tot = torch.sum((y - torch.mean(y)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
print(f"\nModel Performance:")
print(f"R² Score: {r2_score.item():.4f}")
print(f"Final MSE Loss: {losses[-1]:.6f}")

# Verify convergence
if abs(final_beta_0 - TRUE_BETA_0) < 0.1 and abs(final_beta_1 - TRUE_BETA_1) < 0.1:
    print("✅ SUCCESS: Model weights converged close to true parameters!")
else:
    print("⚠️  WARNING: Model weights did not converge as expected. Try more epochs or different learning rate.")