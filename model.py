import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Data pre-processing

# Importing training data
trainingFile = "aapl.us.txt"
trainingData = pd.read_csv(trainingFile, sep=',')
training_data = trainingData.iloc[:, 1:2].values

# Visualizing training data
plt.plot(trainingData, color='red', label="Real Apple Stock Price")
plt.title("Apple Stock 1984 - 2017")
plt.xlabel("Time")
plt.ylabel("Apple Stock Price")
plt.legend()
plt.show()

# Importing testing data (real stock price in 2017)
testingFile = "aapl.us.test.txt"
testingData = pd.read_csv(testingFile, sep=',')
testing_data = testingData.iloc[:, 1:2].values

# Visualizing testing data
plt.plot(testing_data, color='red', label="Real Apple Stock Price")
plt.title("Apple Stock 2017 (Testing Data)")
plt.xlabel("Time")
plt.ylabel("Apple Stock Price")
plt.legend()
plt.show()

# Scaling the open feature
scale = MinMaxScaler(feature_range=(0, 1))
scaled_training_data = scale.fit_transform(training_data)

# Creating data structure with specific time steps and output
X_train = []
Y_train = []
for i in range(60, len(scaled_training_data)):
    X_train.append(scaled_training_data[i - 60:i, 0])
    Y_train.append(scaled_training_data[i, 0])
X_train = np.array(X_train)
Y_train = np.array(Y_train)

# Reshaping data based on the RNN
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Building the RNN

regressor = Sequential()  # RNN

# LSTM layers and Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# Output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')

# Fit RNN to the training data
regressor.fit(X_train, Y_train, epochs=100, batch_size=32)

# Predicting and visualizing results

# Predicting the price
total_data = pd.concat((trainingData["Open"], testingData["Open"]), axis=0)
inputs = total_data[len(total_data) - len(testingData) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = scale.transform(inputs)
X_test = []
for i in range(60, 278):
    X_test.append(inputs[i - 60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

predicted_price = regressor.predict(X_test)
predicted_price = scale.inverse_transform(predicted_price)

# Visualizing results
plt.plot(testing_data, color='red', label="Real Apple Stock Price")
plt.plot(predicted_price, color='blue', label="Predicted Apple Stock Price")
plt.title("Apple Stock Prediction")
plt.xlabel("Time")
plt.ylabel("Apple Stock Price")
plt.legend()
plt.show()
