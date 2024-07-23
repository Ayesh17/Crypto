import numpy as np
from tensorflow.keras.callbacks import Callback
from utils import inverse_transform, calculate_metrics

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
        print(f'Epoch {epoch + 1}: val_loss={logs["val_loss"]:.4f}, val_mae={mae:.4f}, val_mse={mse:.4f}, val_rmse={rmse:.4f}, val_mape={mape:.4f}')
