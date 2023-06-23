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

df = pd.read_csv('eth_ohlcv.csv') # replace with your own dataset
df.dropna(inplace=True)
print(len(df))
df = df.iloc[25000:] # dependent on dataset size and testing
df.reset_index(drop=True, inplace=True)

close_data = df[['close']]
print(close_data.shape)

train_size = int(len(close_data) * 0.75)
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

def create_dataset(dataset, time_step):
    dataX, dataY = [], []
    for i in range(len(dataset)-2*time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        b = dataset[(i+time_step):(i+2*time_step), 0]
        dataY.append(b)
    return np.array(dataX), np.array(dataY)

time_step = 168 # one week of hourly data
neurons = 100 
dropout_rate = 0.2

X_train, y_train = create_dataset(train_data_scaled, time_step)
X_test, y_test = create_dataset(test_data_scaled, time_step)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
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

model.save("trained_eth_hourly_lstm.h5")
print("Model saved successfully!")

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss {} / {} / {}  (Time: {})'.format(time_step,neurons,dropout_rate, current_time))
plt.legend(loc=0)
plt.show()

# Load the saved model if applicable
# from keras.models import load_model
# model = load_model('trained_eth_hourly_lstm.h5')

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

plt.figure(figsize=(12, 6))
plt.plot(close_data, label='Original Close')
plt.plot(trainPredictPlot, label='Training Predictions')
plt.plot(testPredictPlot, label='Test Predictions')
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

hours = range(len(close_data), len(close_data) + time_step)

plt.figure(figsize=(12, 6))
plt.plot(hours, predictions, label='Predictions')
plt.xlabel('Hour')
plt.ylabel('Price')
plt.title('Future Price Predictions {} / {} / {}  (Time: {})'.format(time_step, neurons, dropout_rate, current_time))
plt.legend()
plt.show()
