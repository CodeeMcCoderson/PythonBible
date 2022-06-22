from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import datetime as dt
import pandas_datareader as web
import numpy as np

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

apple = web.DataReader('AAPL', 'yahoo', start, end)
data = apple['Adj Close']

# choose an amount of days and shift our price by that amount, we can see how they developed in past times and predict future times
days = 50
data['Shifted'] = data['Adj Close'].shift(-days)
data.dropna(inplace=True)
# created new column of prices shifted 50 days upward, then drop Nan values at end
# convert to NumPy arrays
X = np.array(data.drop(['Shifted'],1))
Y = np.array(data['Shifted'])
X = preprocessing.scale(X)
# we scale our x values down (normalize them) to make model more efficient
# now we split into training and testing data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2) # training on 20% testing on 80% of data)

clf = LinearRegression()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)
# cut out last 50 days and create new array which takes last 50 days of remaining to predict future values 
X = X[:-days]
X_new = X[-days:]

prediction = clf.predict(X_new)
print(prediction)
