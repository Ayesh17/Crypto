import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from preprocessing import load_and_preprocess_data
from model import LSTMModel
import pandas as pd

# Set seeds for reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Check if CUDA is available and set the device to GPU if possible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
file_path = '../Dataset/coin_Bitcoin_filtered.csv'
window_size = 10
X_train, X_val, X_test, y_train, y_val, y_test, scaler, df = load_and_preprocess_data(file_path, window_size)

# Convert data to PyTorch tensors and move them to the appropriate device
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val, dtype=torch.float32).to(device)
y_test = torch.tensor(y_test, dtype=torch.float32).to(device)

# Create LSTM model and move it to the appropriate device
model = LSTMModel().to(device)
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
epochs = 20
train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    y_pred = model(X_train)
    single_loss = loss_function(y_pred, y_train)
    single_loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_pred = model(X_val)
        val_loss = loss_function(val_pred, y_val)

    train_losses.append(single_loss.item())
    val_losses.append(val_loss.item())

    print(f'Epoch {epoch + 1} Train Loss: {single_loss.item()} Validation Loss: {val_loss.item()}')

# Step 5: Evaluation on the test set
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_inverse = scaler.inverse_transform(y_pred.cpu().numpy())
    y_test_inverse = scaler.inverse_transform(y_test.cpu().numpy().reshape(-1, 1))

mae = mean_absolute_error(y_test_inverse, y_pred_inverse)
mse = mean_squared_error(y_test_inverse, y_pred_inverse)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(y_test_inverse, y_pred_inverse)

print(f'Test MAE: {mae}')
print(f'Test MSE: {mse}')
print(f'Test RMSE: {rmse}')
print(f'Test MAPE: {mape}')

# Plot the predictions against the actual values
plt.figure(figsize=(10, 6))
plt.plot(df.index[len(df) - len(y_test_inverse):], y_test_inverse, label='Actual')
plt.plot(df.index[len(df) - len(y_test_inverse):], y_pred_inverse, label='Predicted')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Bitcoin Price Prediction')
plt.legend()
plt.show()

# Step 6: Forecasting future prices
future_steps = 30  # Predicting next 30 days
last_window = X_test[-1].reshape((1, window_size, 1)).to(device)
future_forecast = []

model.eval()
for _ in range(future_steps):
    with torch.no_grad():
        pred = model(last_window)
        future_forecast.append(pred.item())
        last_window = torch.cat((last_window[:, 1:, :], pred.reshape(1, 1, 1)), dim=1)

future_forecast = scaler.inverse_transform(np.array(future_forecast).reshape(-1, 1))

plt.figure(figsize=(10, 6))
plt.plot(df, label='Historical')
plt.plot(pd.date_range(start=df.index[-1], periods=future_steps, freq='D'), future_forecast, label='Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.title('Future Bitcoin Price Forecast')
plt.legend()
plt.show()

# Plot the training and validation metrics
plt.figure(figsize=(12, 8))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
