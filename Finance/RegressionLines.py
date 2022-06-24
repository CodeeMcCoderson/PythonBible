import numpy as np
import datetime as dt
import pandas_datareader as web
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

start = dt.datetime(2016,1,1)
end = dt.datetime.now()

apple = web.DataReader('AAPL', 'yahoo', start, end)
data = apple['Adj Close']
# quantify datas to be able to use them as x values in algorithm
x = data.index.map(mdates.date2num)
# use NumPy to create linear regression line that fits share price curve
fit = np.polyfit(x, data.values, 1) # dates, prices, 1 is degree of function (linear)
fitid = np.poly1d(fit) # takes the list and makes a function for x

plt.grid()
plt.plot(data.index, data.values, 'b') # plot values and dates
plt.plot(data.index, fitid(x), 'r') # plot 1D values and dates
plt.show()

# set a different timefrome for regression line
rstart = dt.datetime(2020, 1, 1)
rend = dt.datetime(2020, 12, 1)
# create data frame fit_data
fit_data = data.reset_index()
# calculate 2 positions by querying data from df
pos1 = fit_data[fit_data.Date >= rstart].index[0] # looking for 1st position having date >= rstart
pos2 = fit_data[fit_data.Date <= rend].index[-1] # looking for last position having data <= rend

fit_data = fit_data.iloc[pos1:pos2]
# rewrite fit functions
dates = fit_data.Date.map(mdates.date2num)

fit = np.polyfit(dates, fit_data['Adj Close'], 1)
fit1d = np.poly1d(fit)

plt.grid()
plt.plot(data.index, data.values, 'b')
plt.plot(fit_data.Date, fitid(dates), 'r')
plt.show()
