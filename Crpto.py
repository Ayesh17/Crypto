#imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
data = pd.read_csv('Dataset/coin_Ethereum.csv')

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

data
# print(data.columns)  # This will print all column names in the DataFrame


data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)
# data = data[['Date', 'Close']]
# data
print(data.head())

data = data[['Close']]
data

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device


# Assuming 'Date' is already the datetime index of your DataFrame:
plt.figure(figsize=(10, 5))  # Optional: Enhances figure size for better visibility
plt.plot(data.index, data['Close'])  # Use 'data.index' instead of 'data['Date']'
plt.title('Bitcoin Closing Prices Over Time')  # Optional: Adds a title to the plot
plt.xlabel('Date')  # Labels the x-axis
plt.ylabel('Close Price (USD)')  # Labels the y-axis
plt.show()  # Ensures that the plot is displayed when running in some environments

# Function to prepare the DataFrame with lagged features
def prepare_dataframe_for_lstm(df, n_steps, column_name='Close'):
    df_lagged = df.copy()
    for i in range(1, n_steps + 1):
        df_lagged[f'{column_name}(t-{i})'] = df[column_name].shift(i)
    df_lagged.dropna(inplace=True)  # Remove rows with NaN values resulting from shifts
    return df_lagged

# Function to create overlapping windows using the lagged DataFrame
def create_windows_with_lags(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)  # Convert rows of DataFrame into a single window
        y.append(data.iloc[i, 0])  # Assuming the target variable 'Close' is at column 0
    return np.array(X), np.array(y)

# Prepare the DataFrame with lagged features
lookback = 9  # Number of lags
shifted_df = prepare_dataframe_for_lstm(data, lookback, 'Close')

# Create windows from the lagged DataFrame
window_size = 30  # Number of rows each window will contain
X, y = create_windows_with_lags(shifted_df, window_size)

print(shifted_df.head())

shifted_df_as_np = shifted_df.to_numpy()

shifted_df_as_np

shifted_df_as_np.shape


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
shifted_df_as_np = scaler.fit_transform(shifted_df_as_np)

shifted_df_as_np


X = shifted_df_as_np[:, 1:]
y = shifted_df_as_np[:, 0]

X.shape, y.shape


# Calculate indices for train, validation, and test split
train_split = int(len(X) * 0.70)
val_split = int(len(X) * 0.85)  # This marks the end of the validation set at 85% of the data

# Split the data into train, validation, and test sets
X_train, y_train = X[:train_split], y[:train_split]
X_val, y_val = X[train_split:val_split], y[train_split:val_split]
X_test, y_test = X[val_split:], y[val_split:]

# Print the shapes of the datasets to confirm their sizes
print("Training data shape:", X_train.shape)
print("Validation data shape:", X_val.shape)
print("Test data shape:", X_test.shape)

# Reshape features for LSTM input
X_train = X_train.reshape((-1, lookback, 1))
X_val = X_val.reshape((-1, lookback, 1))  # Add this line
X_test = X_test.reshape((-1, lookback, 1))

# Reshape targets
y_train = y_train.reshape((-1, 1))
y_val = y_val.reshape((-1, 1))  # Add this line
y_test = y_test.reshape((-1, 1))

# Print shapes to verify
print("Training data shape:", X_train.shape, y_train.shape)
print("Validation data shape:", X_val.shape, y_val.shape)  # Verify this as well
print("Test data shape:", X_test.shape, y_test.shape)

X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_val = torch.tensor(X_val).float()  # Converting validation data
y_val = torch.tensor(y_val).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()

# Print shapes to verify
print("Training data shape:", X_train.shape, y_train.shape)
print("Validation data shape:", X_val.shape, y_val.shape)  # Check this as well
print("Test data shape:", X_test.shape, y_test.shape)



from torch.utils.data import Dataset

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)
test_dataset = TimeSeriesDataset(X_test, y_test)


from torch.utils.data import DataLoader

batch_size = 16

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_stacked_layers):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(1, 4, 1)
model.to(device)
model


def train_one_epoch():
    model.train(True)
    print(f'Epoch: {epoch + 1}')
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:  # print every 100 batches
            avg_loss_across_batches = running_loss / 100
            print('Batch {0}, Loss: {1:.3f}'.format(batch_index+1,
                                                    avg_loss_across_batches))
            running_loss = 0.0
    print()



def validate_one_epoch():
    model.train(False)
    running_loss = 0.0

    for batch_index, batch in enumerate(test_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(test_loader)

    print('Val Loss: {0:.3f}'.format(avg_loss_across_batches))
    print('***************************************************')
    print()


learning_rate = 0.001
num_epochs = 100
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    train_one_epoch()
    validate_one_epoch()

with torch.no_grad():
    predicted = model(X_train.to(device)).to('cpu').numpy()

plt.plot(y_train, label='Actual Close')
plt.plot(predicted, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

from copy import deepcopy as dc
train_predictions = predicted.flatten()

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = train_predictions
dummies = scaler.inverse_transform(dummies)

train_predictions = dc(dummies[:, 0])
train_predictions

dummies = np.zeros((X_train.shape[0], lookback+1))
dummies[:, 0] = y_train.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_train = dc(dummies[:, 0])
new_y_train

plt.plot(new_y_train, label='Actual Close')
plt.plot(train_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()

test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = test_predictions
dummies = scaler.inverse_transform(dummies)

test_predictions = dc(dummies[:, 0])
test_predictions


dummies = np.zeros((X_test.shape[0], lookback+1))
dummies[:, 0] = y_test.flatten()
dummies = scaler.inverse_transform(dummies)

new_y_test = dc(dummies[:, 0])
new_y_test


# Calculate and print the evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mae = mean_absolute_error(new_y_test, test_predictions)
print(f"Mean Absolute Error (MAE): {mae}")

rmse = np.sqrt(mean_squared_error(new_y_test, test_predictions))
print(f"Root Mean Square Error (RMSE): {rmse}")

# Ensure no division by zero
new_y_test, test_predictions = np.array(new_y_test), np.array(test_predictions)
mape = np.mean(np.abs((new_y_test - test_predictions) / new_y_test)) * 100 if np.all(new_y_test) else float('inf')
print(f"Mean Absolute Percentage Error (MAPE): {mape}%")


plt.plot(new_y_test, label='Actual Close')
plt.plot(test_predictions, label='Predicted Close')
plt.xlabel('Day')
plt.ylabel('Close')
plt.legend()
plt.show()
