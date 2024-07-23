import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    data = pd.read_csv(file_path)
    data['Date'] = pd.to_datetime(data['Date'])
    data.set_index('Date', inplace=True)
    data = data[['Close']]
    return data

def plot_data(data):
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'])
    plt.title('Bitcoin Closing Prices Over Time')
    plt.xlabel('Date')
    plt.ylabel('Close Price (USD)')
    plt.show()

def prepare_dataframe_for_lstm(df, n_steps, column_name='Close'):
    df_lagged = df.copy()
    for i in range(1, n_steps + 1):
        df_lagged[f'{column_name}(t-{i})'] = df[column_name].shift(i)
    df_lagged.dropna(inplace=True)
    return df_lagged

# def create_windows_with_lags(data, window_size):
#     X = []
#     y = []
#     for i in range(window_size, len(data)):
#         X.append(data.iloc[i-window_size:i].values)
#         y.append(data.iloc[i, 0])
#     return np.array(X), np.array(y)

def preprocess_data(data, lookback, window_size):
    shifted_df = prepare_dataframe_for_lstm(data, lookback, 'Close')
    X, y = create_windows_with_lags(shifted_df, window_size)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    shifted_df_as_np = scaler.fit_transform(shifted_df.to_numpy())

    X = shifted_df_as_np[:, 1:]
    y = shifted_df_as_np[:, 0]

    return X, y, scaler

# def split_data(X, y, train_split=0.8, val_split=0.9):
#     # Calculate the index to cut off the last 20%
#     # cutoff_idx = int(len(X) * 0.7)
#     #
#     # # Use the remaining data for splitting
#     # train_idx = int(cutoff_idx * train_split)
#     # val_idx = int(cutoff_idx * val_split)
#     train_idx = int(len(X) * train_split)
#     val_idx = int(len(X) * val_split)
#
#     X_train, y_train = X[:train_idx], y[:train_idx]
#     X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
#     X_test, y_test = X[val_idx:], y[val_idx:]
#
#     # # Print shapes to verify
#     # print("Training data shape:", X_train.shape, y_train.shape)
#     # print("Validation data shape:", X_val.shape, y_val.shape)
#     # print("Test data shape:", X_test.shape, y_test.shape)
#
#     return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Updated create_windows_with_lags function with print statements
def create_windows_with_lags(data, window_size):
    X = []
    y = []
    for i in range(window_size, len(data)):
        X.append(data.iloc[i-window_size:i].values)
        y.append(data.iloc[i, 0])
    X, y = np.array(X), np.array(y)
    print(f"Window Size: {window_size}, X shape: {X.shape}, y shape: {y.shape}")
    return X, y

# Updated split_data function with print statements
def split_data(X, y, train_split=0.8, val_split=0.9):
    train_idx = int(len(X) * train_split)
    val_idx = int(len(X) * val_split)

    X_train, y_train = X[:train_idx], y[:train_idx]
    X_val, y_val = X[train_idx:val_idx], y[train_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]

    print(f"Training data shape: {X_train.shape}, {y_train.shape}")
    print(f"Validation data shape: {X_val.shape}, {y_val.shape}")
    print(f"Test data shape: {X_test.shape}, {y_test.shape}")

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)



