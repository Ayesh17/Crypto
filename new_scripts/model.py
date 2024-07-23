from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, SimpleRNN, Bidirectional, Dense, Dropout
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA

# Deep Learning Model Classes
class LSTMModel:
    def __init__(self, input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(LSTM(units=units, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(LSTM(units=units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def get_model(self):
        return self.model

class BiLSTMModel:
    def __init__(self, input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=units, return_sequences=True), input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Bidirectional(LSTM(units=units)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def get_model(self):
        return self.model

class GRUModel:
    def __init__(self, input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(GRU(units=units, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(GRU(units=units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def get_model(self):
        return self.model

class BiGRUModel:
    def __init__(self, input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(Bidirectional(GRU(units=units, return_sequences=True), input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Bidirectional(GRU(units=units)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def get_model(self):
        return self.model

class RNNModel:
    def __init__(self, input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(SimpleRNN(units=units, return_sequences=True, input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(SimpleRNN(units=units))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def get_model(self):
        return self.model

class BiRNNModel:
    def __init__(self, input_shape, units=50, dropout_rate=0.2, optimizer='adam'):
        self.model = Sequential()
        self.model.add(Bidirectional(SimpleRNN(units=units, return_sequences=True), input_shape=input_shape))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Bidirectional(SimpleRNN(units=units)))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(1))

        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

    def get_model(self):
        return self.model

# Basic Machine Learning Model Classes
class DecisionTreeModel:
    def __init__(self):
        self.model = DecisionTreeRegressor()

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class RandomForestModel:
    def __init__(self, n_estimators=100):
        self.model = RandomForestRegressor(n_estimators=n_estimators)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

class ARIMAModel:
    def __init__(self, order=(5, 1, 0)):
        self.order = order
        self.model = None

    def fit(self, train_data):
        self.model = ARIMA(train_data, order=self.order)
        self.model = self.model.fit()

    def predict(self, start, end):
        return self.model.predict(start=start, end=end)
