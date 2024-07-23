import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
import random

from preprocessing import load_data, preprocess_data, split_data
from dataset_update import TimeSeriesDataset
from model import LSTM, BiLSTM
from copy import deepcopy as dc

# Set seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_one_epoch(model, train_loader, loss_function, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch_index, batch in enumerate(train_loader):
        x_batch, y_batch = batch[0].to(device), batch[1].to(device)
        output = model(x_batch)
        loss = loss_function(output, y_batch)
        running_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % 100 == 99:
            avg_loss_across_batches = running_loss / 100
            print(f'Batch {batch_index + 1}, Loss: {avg_loss_across_batches:.3f}')
            running_loss = 0.0

def validate_one_epoch(model, val_loader, loss_function, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch_index, batch in enumerate(val_loader):
            x_batch, y_batch = batch[0].to(device), batch[1].to(device)
            output = model(x_batch)
            loss = loss_function(output, y_batch)
            running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(val_loader)
    return avg_loss_across_batches

def hyperparameter_tuning(X, y, device, input_sizes, hidden_sizes, num_layers_list, dropout_rates, n_splits=5):
    best_model = None
    best_loss = float('inf')
    best_params = None

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for input_size in input_sizes:
        for hidden_size in hidden_sizes:
            for num_layers in num_layers_list:
                for dropout_rate in dropout_rates:
                    val_losses = []

                    for train_index, val_index in kf.split(X):
                        X_train_fold = X[train_index]
                        y_train_fold = y[train_index]
                        X_val_fold = X[val_index]
                        y_val_fold = y[val_index]

                        train_dataset = TimeSeriesDataset(X_train_fold, y_train_fold)
                        val_dataset = TimeSeriesDataset(X_val_fold, y_val_fold)

                        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

                        model = LSTM(input_size, hidden_size, num_layers, dropout_rate).to(device)
                        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
                        loss_function = nn.MSELoss()

                        for epoch in range(10):  # Train for fewer epochs for tuning
                            train_one_epoch(model, train_loader, loss_function, optimizer, device)
                            val_loss = validate_one_epoch(model, val_loader, loss_function, device)

                        val_losses.append(val_loss)

                    avg_val_loss = np.mean(val_losses)
                    print(f'Params: input_size={input_size}, hidden_size={hidden_size}, num_layers={num_layers}, dropout_rate={dropout_rate}, Val Loss: {avg_val_loss:.3f}')

                    if avg_val_loss < best_loss:
                        best_loss = avg_val_loss
                        best_model = model
                        best_params = (input_size, hidden_size, num_layers, dropout_rate)

    print(f'Best params: input_size={best_params[0]}, hidden_size={best_params[1]}, num_layers={best_params[2]}, dropout_rate={best_params[3]}, Val Loss: {best_loss:.3f}')
    return best_model, best_params

if __name__ == '__main__':
    # Set seeds
    set_seed()

    # Load data
    # data = load_data('../Dataset/coin_Bitcoin.csv')
    # data = load_data('../Dataset/coin_Ethereum.csv')
    data = load_data('../Dataset/coin_Bitcoin_filtered.csv')

    # Preprocess data
    lookback = 20
    window_size = 10
    X, y, scaler = preprocess_data(data, lookback, window_size)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    # Reshape data
    X_train = X_train.reshape((-1, lookback, 1))
    X_val = X_val.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_val = y_val.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # Combine train, val, and test sets for cross-validation
    X_all = np.concatenate((X_train, X_val, X_test), axis=0)
    y_all = np.concatenate((y_train, y_val, y_test), axis=0)

    # Convert to torch tensors
    X_all = torch.tensor(X_all).float()
    y_all = torch.tensor(y_all).float()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # # Define hyperparameter grids
    # input_sizes = [1]  # Based on the feature size, you might want to adjust this
    # hidden_sizes = [1, 2, 4, 8, 16]
    # num_layers_list = [1, 2, 3, 4, 5]
    # dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]
    #
    # # Hyperparameter tuning
    # best_model, best_params = hyperparameter_tuning(X_all, y_all, device, input_sizes, hidden_sizes, num_layers_list, dropout_rates)

    best_params = [ 1, 4, 1, 0.4]

    print("Best params:", best_params)

    # Train with the best parameters for the full number of epochs
    input_size, hidden_size, num_layers, dropout_rate = best_params
    model = BiLSTM(input_size, hidden_size, num_layers, dropout_rate).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.MSELoss()
    num_epochs = 100

    # Split back into train/val/test sets
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X_all.numpy(), y_all.numpy())

    # Convert to torch tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_val = torch.tensor(X_val).float()
    y_val = torch.tensor(y_val).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}')
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
        val_loss = validate_one_epoch(model, val_loader, loss_function, device)
        print(f'Validation Loss: {val_loss:.3f}')

    # Prediction and evaluation
    with torch.no_grad():
        predicted = model(X_train.to(device)).cpu().numpy().flatten()

    plt.plot(y_train, label='Actual Close')
    plt.plot(predicted, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    train_predictions = predicted.flatten()
    dummies = np.zeros((X_train.shape[0], lookback + 1))
    dummies[:, 0] = train_predictions
    dummies = scaler.inverse_transform(dummies)
    train_predictions = dc(dummies[:, 0])

    dummies = np.zeros((X_train.shape[0], lookback + 1))
    dummies[:, 0] = y_train.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_train = dc(dummies[:, 0])

    plt.plot(new_y_train, label='Actual Close')
    plt.plot(train_predictions, label='Predicted Close')
    plt.xlabel('Day')
    plt.ylabel('Close')
    plt.legend()
    plt.show()

    # Testing
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)
    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((X_test.shape[0], lookback + 1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)
    new_y_test = dc(dummies[:, 0])

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
