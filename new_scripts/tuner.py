import keras_tuner as kt
import tensorflow as tf
from model import LSTMModel, BiLSTMModel, GRUModel, BiGRUModel, RNNModel, BiRNNModel

def build_lstm_model(hp):
    input_shape = (60, 1)
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return LSTMModel(input_shape, units=units, dropout_rate=dropout_rate, optimizer=optimizer).get_model()

def build_gru_model(hp):
    input_shape = (60, 1)
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return GRUModel(input_shape, units=units, dropout_rate=dropout_rate, optimizer=optimizer).get_model()

def build_rnn_model(hp):
    input_shape = (60, 1)
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return RNNModel(input_shape, units=units, dropout_rate=dropout_rate, optimizer=optimizer).get_model()

def build_bilstm_model(hp):
    input_shape = (60, 1)
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return BiLSTMModel(input_shape, units=units, dropout_rate=dropout_rate, optimizer=optimizer).get_model()

def build_bigru_model(hp):
    input_shape = (60, 1)
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return BiGRUModel(input_shape, units=units, dropout_rate=dropout_rate, optimizer=optimizer).get_model()

def build_birnn_model(hp):
    input_shape = (60, 1)
    units = hp.Int('units', min_value=32, max_value=512, step=32)
    dropout_rate = hp.Float('dropout_rate', min_value=0.2, max_value=0.5, step=0.1)
    learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    return BiRNNModel(input_shape, units=units, dropout_rate=dropout_rate, optimizer=optimizer).get_model()

def hyperparameter_tuning(X_train, y_train, X_val, y_val, model_type):
    if model_type == 'LSTM':
        build_model = build_lstm_model
    elif model_type == 'GRU':
        build_model = build_gru_model
    elif model_type == 'RNN':
        build_model = build_rnn_model
    elif model_type == 'biLSTM':
        build_model = build_bilstm_model
    elif model_type == 'biGRU':
        build_model = build_bigru_model
    elif model_type == 'biRNN':
        build_model = build_birnn_model
    else:
        raise ValueError("Unsupported model type for hyperparameter tuning")

    tuner = kt.Hyperband(
        build_model,
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
