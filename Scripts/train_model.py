import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error

from preprocessing import load_data, preprocess_data, split_data
from dataset_update import TimeSeriesDataset
from model import LSTM
from copy import deepcopy as dc

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

            with torch.no_grad():
                output = model(x_batch)
                loss = loss_function(output, y_batch)
                running_loss += loss.item()

    avg_loss_across_batches = running_loss / len(val_loader)
    print(f'Val Loss: {avg_loss_across_batches:.3f}')
    print('***************************************************')

if __name__ == '__main__':
    # Load data
    # data = load_data('../Dataset/coin_Ethereum.csv')
    data = load_data('../Dataset/coin_Bitcoin.csv')

    # preprocess data
    lookback = 9
    window_size = 30
    X, y, scaler = preprocess_data(data, lookback, window_size)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = split_data(X, y)

    # Reshape data
    X_train = X_train.reshape((-1, lookback, 1))
    X_val = X_val.reshape((-1, lookback, 1))
    X_test = X_test.reshape((-1, lookback, 1))

    y_train = y_train.reshape((-1, 1))
    y_val = y_val.reshape((-1, 1))
    y_test = y_test.reshape((-1, 1))

    # Print shapes to verify
    print("Training data shape:", X_train.shape, y_train.shape)
    print("Validation data shape:", X_val.shape, y_val.shape)  # Check this as well
    print("Test data shape:", X_test.shape, y_test.shape)

    # Convert to torch tensors
    X_train = torch.tensor(X_train).float()
    y_train = torch.tensor(y_train).float()
    X_val = torch.tensor(X_val).float()
    y_val = torch.tensor(y_val).float()
    X_test = torch.tensor(X_test).float()
    y_test = torch.tensor(y_test).float()

    # Create datasets and data loaders
    train_dataset = TimeSeriesDataset(X_train, y_train)
    val_dataset = TimeSeriesDataset(X_val, y_val)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define model, loss function, and optimizer
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = LSTM(1, 4, 1)
    model.to(device)

    learning_rate = 0.001
    num_epochs = 100
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        print(f'Epoch: {epoch + 1}')
        train_one_epoch(model, train_loader, loss_function, optimizer, device)
        validate_one_epoch(model, val_loader, loss_function, device)

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
    dummies = np.zeros((X_train.shape[0], lookback+1))
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
    #
    # # Prediction and evaluation
    # with torch.no_grad():
    #     test_predictions = model(X_test.to(device)).cpu().numpy().flatten()
    #
    # dummies = np.zeros((X_test.shape[0], lookback + 1))
    # dummies[:, 0] = test_predictions
    # dummies = scaler.inverse_transform(dummies)
    #
    # test_predictions = dummies[:, 0]
    #
    # dummies[:, 0] = y_test.flatten()
    # dummies = scaler.inverse_transform(dummies)
    # new_y_test = dummies[:, 0]
    #
    # mae = mean_absolute_error(new_y_test, test_predictions)
    # print(f"Mean Absolute Error (MAE): {mae}")
    #
    # rmse = np.sqrt(mean_squared_error(new_y_test, test_predictions))
    # print(f"Root Mean Square Error (RMSE): {rmse}")
    #
    # mape = np.mean(np.abs((new_y_test - test_predictions) / new_y_test)) * 100
    # print(f"Mean Absolute Percentage Error (MAPE): {mape}%")
    #
    # plt.plot(new_y_test, label='Actual Close')
    # plt.plot(test_predictions, label='Predicted Close')
    # plt.xlabel('Day')
    # plt.ylabel('Close')
    # plt.legend()
    # plt.show()




  #testing
  # Prediction and evaluation
    test_predictions = model(X_test.to(device)).detach().cpu().numpy().flatten()

    dummies = np.zeros((X_test.shape[0], lookback + 1))
    dummies[:, 0] = test_predictions
    dummies = scaler.inverse_transform(dummies)

    test_predictions = dc(dummies[:, 0])

    dummies = np.zeros((X_test.shape[0], lookback + 1))
    dummies[:, 0] = y_test.flatten()
    dummies = scaler.inverse_transform(dummies)

    new_y_test = dc(dummies[:, 0])
    new_y_test

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

