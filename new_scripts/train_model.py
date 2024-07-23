import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_absolute_error, mean_squared_error
from preprocessing import load_data, preprocess_data, split_data
from model import LSTMModel, BiLSTMModel, GRUModel, BiGRUModel, RNNModel, BiRNNModel, DecisionTreeModel, RandomForestModel, ARIMAModel
from tuner import hyperparameter_tuning
from callbacks import PrintMetrics
from utils import inverse_transform, calculate_metrics, plot_history, plot_predictions

def train_and_evaluate(model_type='LSTM', window_size=60, epochs=50, batch_size=32):
    df = load_data('../Dataset/coin_Bitcoin_filtered.csv')
    dates = df['Date'].values
    X, y, scaler = preprocess_data(df, window_size)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X, y)

    if model_type in ['LSTM', 'biLSTM', 'GRU', 'biGRU', 'RNN', 'biRNN']:
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_val = np.reshape(X_val, (X_val.shape[0], X_val.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        best_hps = hyperparameter_tuning(X_train, y_train, X_val, y_val, model_type)

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first layer is {best_hps.get('units')},
        the optimal dropout rate is {best_hps.get('dropout_rate')},
        and the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
        """)

        if model_type == 'LSTM':
            model = LSTMModel(
                input_shape=(window_size, 1),
                units=best_hps.get('units'),
                dropout_rate=best_hps.get('dropout_rate'),
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
            ).get_model()
        elif model_type == 'biLSTM':
            model = BiLSTMModel(
                input_shape=(window_size, 1),
                units=best_hps.get('units'),
                dropout_rate=best_hps.get('dropout_rate'),
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
            ).get_model()
        elif model_type == 'GRU':
            model = GRUModel(
                input_shape=(window_size, 1),
                units=best_hps.get('units'),
                dropout_rate=best_hps.get('dropout_rate'),
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
            ).get_model()
        elif model_type == 'biGRU':
            model = BiGRUModel(
                input_shape=(window_size, 1),
                units=best_hps.get('units'),
                dropout_rate=best_hps.get('dropout_rate'),
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
            ).get_model()
        elif model_type == 'RNN':
            model = RNNModel(
                input_shape=(window_size, 1),
                units=best_hps.get('units'),
                dropout_rate=best_hps.get('dropout_rate'),
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
            ).get_model()
        elif model_type == 'biRNN':
            model = BiRNNModel(
                input_shape=(window_size, 1),
                units=best_hps.get('units'),
                dropout_rate=best_hps.get('dropout_rate'),
                optimizer=tf.keras.optimizers.Adam(learning_rate=best_hps.get('learning_rate'))
            ).get_model()

        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, mode='min', verbose=1)
        metrics_callback = PrintMetrics(validation_data=(X_val, y_val), scaler=scaler)

        history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, callbacks=[metrics_callback, checkpoint_callback])

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

        plot_predictions(dates[window_size:len(y_train_inv)+window_size], y_train_inv, y_train_pred_inv, 'Train Data: Actual vs Predicted')
        plot_predictions(dates[len(y_train_inv)+window_size:len(y_train_inv)+len(y_val_inv)+window_size], y_val_inv, y_val_pred_inv, 'Validation Data: Actual vs Predicted')
        plot_predictions(dates[-len(y_test_inv):], y_test_inv, y_test_pred_inv, 'Test Data: Actual vs Predicted')

        test_mae, test_mse, test_rmse, test_mape = calculate_metrics(y_test_inv, y_test_pred_inv)
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test MSE: {test_mse:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}')
        print(f'Test MAPE: {test_mape:.4f}')

    elif model_type == 'DecisionTree':
        model = DecisionTreeModel()
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

    elif model_type == 'RandomForest':
        model = RandomForestModel()
        model.fit(X_train, y_train)
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)

    elif model_type == 'ARIMA':
        model = ARIMAModel()
        model.fit(y_train)
        y_train_pred = model.predict(start=0, end=len(y_train)-1)
        y_val_pred = model.predict(start=len(y_train), end=len(y_train) + len(y_val) - 1)
        y_test_pred = model.predict(start=len(y_train) + len(y_val), end=len(y_train) + len(y_val) + len(y_test) - 1)

    else:
        raise ValueError("Unsupported model type")

    if model_type not in ['LSTM', 'biLSTM', 'GRU', 'biGRU', 'RNN', 'biRNN']:
        y_train_inv = inverse_transform(scaler, y_train.reshape(-1, 1))
        y_val_inv = inverse_transform(scaler, y_val.reshape(-1, 1))
        y_test_inv = inverse_transform(scaler, y_test.reshape(-1, 1))
        y_train_pred_inv = inverse_transform(scaler, y_train_pred.reshape(-1, 1))
        y_val_pred_inv = inverse_transform(scaler, y_val_pred.reshape(-1, 1))
        y_test_pred_inv = inverse_transform(scaler, y_test_pred.reshape(-1, 1))

        plot_predictions(dates[window_size:len(y_train_inv)+window_size], y_train_inv, y_train_pred_inv, 'Train Data: Actual vs Predicted')
        plot_predictions(dates[len(y_train_inv)+window_size:len(y_train_inv)+len(y_val_inv)+window_size], y_val_inv, y_val_pred_inv, 'Validation Data: Actual vs Predicted')
        plot_predictions(dates[-len(y_test_inv):], y_test_inv, y_test_pred_inv, 'Test Data: Actual vs Predicted')

        test_mae, test_mse, test_rmse, test_mape = calculate_metrics(y_test_inv, y_test_pred_inv)
        print(f'Test MAE: {test_mae:.4f}')
        print(f'Test MSE: {test_mse:.4f}')
        print(f'Test RMSE: {test_rmse:.4f}')
        print(f'Test MAPE: {test_mape:.4f}')

    return model

if __name__ == "__main__":
    model_type = 'LSTM'  # or 'biLSTM', 'GRU', 'biGRU', 'RNN', 'biRNN', 'DecisionTree', 'RandomForest', 'ARIMA'
    window_size = 60
    epochs = 100
    batch_size = 32

    best_model = train_and_evaluate(model_type=model_type, window_size=window_size, epochs=epochs, batch_size=batch_size)
