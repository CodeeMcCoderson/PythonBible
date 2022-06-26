# Description: This program uses an artificial recurrent neural network called Long Short Term Memory (LSTM)
#              to predict the closing stock price of a corporation using the past 60 day stock price
import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')

# Choose your ticker symbol, data source, and date range
df = web.DataReader('ROOT', data_source='yahoo', start='2020-10-27', end='2022-06-07')

def closingPriceGraph():
    # see the size of the data
    df.shape
    #Create a dataframe of the closing cost
    plt.figure(figsize=(16,8))
    plt.title('Close Price History')
    plt.plot(df['Close'])
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.show()

def trainedGraph():
    #Create a data frame with only close column
    data = df.filter(['Close'])
    #Convert dataframe to a numpy array
    dataset = data.values
    #Get the number of rows to train the model on
    training_data_len = math.ceil( len(dataset) * .8)
    #Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)
    #Create the training dataset, scaled trainging datadet
    train_data = scaled_data[0:training_data_len , :]
    #Split into x train and y train datasets
    x_train = []
    y_train = []
    #iterate through data to build arrays
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
    #Convert x & y_train to numpy arrays
    x_train, y_train = np.array(x_train), np.array(y_train)
    #Reshape the x_train dataset
    #print(x_train.shape)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
    #Build the LSTM models, the 50, 50, 25, 1 below represent neurons
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    #Compile the models
    model.compile(optimizer='adam', loss='mean_squared_error')
    #train the models, fit another term for train
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    #Create the testing datasets, a new array containing scaled values from index
    test_data = scaled_data[training_data_len - 60: , :]
    #Create the data sets
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i-60:i, 0])
    #Convert the data to numpy arrays
    x_test = np.array(x_test)
    #Reshape dataset to 3 dimentional input_shape. Give rows(samples), colums(timestamps), features(close price)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get the models predicted price VALUES
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)
    #Get the root mean squared error or RMSE, value of 0 means that the model was perfect(predictions and y are equal)
    rmse = np.sqrt(np.mean(predictions - y_test)**2)
    #Plot the data
    train = data[:training_data_len]
    valid = data[training_data_len:]
    valid['Predictions'] = predictions
    #Visualize the data
    plt.figure(figsize=(16,8))
    plt.title('Model')
    plt.xlabel('Data', fontsize=18)
    plt.ylabel('Close Price USD ($)', fontsize=18)
    plt.plot(train['Close'])
    plt.plot(valid[['Close', 'Predictions']])
    plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
    plt.show()
    #Show the actual(left) versus predicted(right) prices
    print(valid)

    predictedPrice(scaler, model)

def predictedPrice(scaler, model):
    #Create new dataframe
    new_df = df.filter(['Close'])
    #Get last 60 day closing price values and convert to arrays
    last_60_days = new_df[-60:].values
    #Scale the data to be between 0 and 1
    last_60_days_scaled = scaler.transform(last_60_days)
    #Create an empty list
    x_test = []
    #Append last 60 days to list
    x_test.append(last_60_days_scaled)
    #Convert the x_test dataset to a NUMPY array
    x_test = np.array(x_test)
    #Reshape the data to be 3 dimentional
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    #Get predicted scaled prices
    pred_price = model.predict(x_test)
    #undo scaling
    pred_price = scaler.inverse_transform(pred_price)
    print(pred_price)

print('Starting Main')
closingPriceGraph()
trainedGraph()
print('End')
