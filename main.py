# Copyright 2023-present Coinbase Global, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout
from keras.regularizers import l2
from keras.layers import LSTM

current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

df = pd.read_csv('eth_ohlcv.csv')  # replace with your own dataset
df.dropna(inplace=True)
print(len(df))
df = df.iloc[25000:]  # dependent on dataset size and testing
df['date'] = pd.to_datetime(df['timestamp'], unit='us')  # Coinbase OHLCV is provided in microseconds
df.reset_index(drop=True, inplace=True)
df['original_index'] = df.index

close_data = df[['close']]
print(close_data.shape)

train_size = int(len(close_data) * 0.75)  # reasonable default; may require customization
train_data = close_data[:train_size]
test_data = close_data[train_size:]
print(train_data.shape)
print(test_data.shape)

train_data = np.array(train_data)
test_data = np.array(test_data)

scaler = MinMaxScaler(feature_range=(0, 1))
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)
print('train_data: ', train_data_scaled.shape)
print('test_data: ', test_data_scaled.shape)

def reshape_data(X):
    return X.reshape(X.shape[0], X.shape[1], 1)

def generate_dataset(dataset, time_step):
    X, y = create_dataset(dataset, time_step)
    X = reshape_data(X)
    return X, y

def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-2*time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        b = dataset[(i+time_step):(i+2*time_step), 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

time_step = 168  # one week of hourly data
neurons = 50  # may require customization
dropout_rate = 0.2 # may require customization

X_train, y_train = generate_dataset(train_data_scaled, time_step)
X_test, y_test = generate_dataset(test_data_scaled, time_step)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

model = Sequential()
model.add(LSTM(neurons, input_shape=(None, 1), activation='tanh', kernel_regularizer=l2(0.01)))
model.add(Dropout(dropout_rate))
model.add(Dense(time_step))
model.compile(loss='mean_absolute_error', optimizer='adam')

early_stopping = EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=100, batch_size=10, verbose=1, callbacks=[early_stopping])

model.save("trained_eth_hourly_lstm.keras")
print("Model saved successfully!")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss {} / {} / {}  (Time: {})'.format(time_step,neurons,dropout_rate, current_time))
plt.legend(loc=0)
plt.show()

# Load the saved model if applicable; comment out lines 81-103 before proceeding
# from keras.models import load_model
# model = load_model('trained_eth_hourly_lstm.keras')

train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

test_loss = model.evaluate(X_test, y_test.reshape(y_test.shape[0], y_test.shape[1]))
print("Test Loss:", test_loss)

trainPredictPlot = np.empty_like(close_data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[time_step:len(train_predict)+time_step, :] = train_predict[:,-1].reshape(-1,1)

testPredictPlot = np.full_like(close_data, np.nan)
testPredictPlot[len(train_data) + time_step : len(train_data) + time_step + len(test_predict), :] = test_predict[:,-1].reshape(-1,1)

current_time = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

train_date = df['date'].iloc[time_step : time_step+len(train_predict)]
test_date = df['date'].iloc[len(train_predict) + 2*time_step + time_step : len(train_predict) + 2*time_step + time_step + len(test_predict)]

plt.figure(figsize=(12, 6))
plt.plot(df['date'], close_data, label='Original Close')
plt.plot(train_date, train_predict[:,-1], label='Training Predictions')
plt.plot(test_date, test_predict[:,-1], label='Test Predictions')
plt.xlabel('Time')
plt.ylabel('Close Value')
plt.title('Close Values vs. Predictions {} / {} / {}  (Time: {})'.format(time_step,neurons,dropout_rate, current_time))
plt.legend()
plt.show()

last_data = test_data_scaled[-time_step:]
last_data = last_data.reshape(1, time_step, 1)

predictions = model.predict(last_data)
predictions = predictions.reshape(-1, 1)
predictions = scaler.inverse_transform(predictions)

freq = 'H'
last_date = df['date'].iloc[-1]
future_dates = pd.date_range(start=last_date, periods=time_step+1, freq=freq)[1:]

plt.figure(figsize=(12, 6))
plt.plot(future_dates, predictions, label='Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Future Price Predictions {} / {} / {}  (Time: {})'.format(time_step, neurons, dropout_rate, current_time))
plt.legend()
plt.show()