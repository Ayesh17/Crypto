import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.keras.models import load_model
import keras_tuner as kt
from preprocessing import load_data, preprocess_data, split_data, inverse_transform, calculate_metrics
from model import create_lstm_model, create_bilstm_model
import matplotlib.pyplot as plt


class PrintMetrics(Callback):
    def __init__(self, validation_data, scaler):
        self.validation_data = validation_data
        self.scaler = scaler

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        y_pred = self.model.predict(X_val)
        y_val_inv = inverse_transform(self.scaler, y_val.reshape(-1, 1))
        y_pred_inv = inverse_transform(self.scaler, y_pred)

        mae, mse, rmse, mape = calculate_metrics(y_val_inv, y_pred_inv)
        print(
            f'Epoch {epoch + 1}: val_loss={logs["val_loss"]:.4f}, val_mae={mae:.4f}, val_mse={mse:.4f}, val_rmse={rmse:.4f}, val_mape={mape:.4f}')


def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(12, 6))
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


def build_lstm_model(hp):
    input_shape = (60, 1)
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return create_lstm_model(input_shape, units=units, dropout_rate=dropout_rate, optimizer=optimizer)


def hyperparameter_tuning(X_train, y_train, X_val, y_val):
    tuner = kt.Hyperband(
        build_lstm_model,
        objective='val_loss',
        max_epochs=10,
        factor=3,
        directory='my_dir',
        project_name='intro_to_kt'
    )

    stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
    tuner.search(X_train, y_train, epochs=50, validation_data=(X_val, y_val), callbacks=[stop_early])

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return best_hps


def train_and_evaluate(model_type='LSTM', window_size=60, epochs=50, batch_size=32):
    # df = load_data('../Dataset/coin_Bitcoin.csv')
    # df = load_data('../Dataset/coin_Bitcoin_filtered.csv')
    df = load_data('../Dataset/coin_Ethereum.csv')

    X, y, scaler = preprocess_data(df, window_size)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    best_hps = hyperparameter_tuning(X_train, y_train, X_val, y_val)

    if model_type == 'LSTM':
        model = create_lstm_model(
            input_shape=(window_size, 1),
            units=best_hps.get('units'),
            dropout_rate=best_hps.get('dropout_rate'),
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
        )
    else:
        model = create_bilstm_model(
            input_shape=(window_size, 1),
            units=best_hps.get('units'),
            dropout_rate=best_hps.get('dropout_rate'),
            optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
        )

    checkpoint_callback = ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min',
                                          verbose=1)
    metrics_callback = PrintMetrics(validation_data=(X_val, y_val), scaler=scaler)

    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size,
                        callbacks=[metrics_callback, checkpoint_callback])

    plot_history(history)

    best_model = load_model('best_model.keras')

    y_train_pred = best_model.predict(X_train)
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)

    y_train_inv = inverse_transform(scaler, y_train.reshape(-1, 1))
    y_train_pred_inv = inverse_transform(scaler, y_train_pred)
    y_val_inv = inverse_transform(scaler, y_val.reshape(-1, 1))
    y_val_pred_inv = inverse_transform(scaler, y_val_pred)
    y_test_inv = inverse_transform(scaler, y_test.reshape(-1, 1))
    y_test_pred_inv = inverse_transform(scaler, y_test_pred)

    plot_predictions(y_train_inv, y_train_pred_inv, 'Train Data: Actual vs Predicted')
    plot_predictions(y_val_inv, y_val_pred_inv, 'Validation Data: Actual vs Predicted')
    plot_predictions(y_test_inv, y_test_pred_inv, 'Test Data: Actual vs Predicted')

    test_mae, test_mse, test_rmse, test_mape = calculate_metrics(y_test_inv, y_test_pred_inv)
    print(f'Test MAE: {test_mae:.4f}')
    print(f'Test MSE: {test_mse:.4f}')
    print(f'Test RMSE: {test_rmse:.4f}')
    print(f'Test MAPE: {test_mape:.4f}')

    return best_model


if __name__ == "__main__":
    model_type = 'LSTM'  # or 'biLSTM'
    window_size = 60
    epochs = 100
    batch_size = 32

    best_model = train_and_evaluate(model_type=model_type, window_size=window_size, epochs=epochs,
                                    batch_size=batch_size)
