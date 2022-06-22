from pandas_datareader import data as web
import datetime as dt
# define the time range you want to pul data back for
start = dt.datetime(2017,1,1)
end = dt.datetime.now()
# pull data back for specific stock, using Yahoo API, in specific date range
df = web.DataReader('AAPL', 'yahoo', start, end)
# can print to see what features you are dealing with
print(df)
# can access specific features
print(df['Close'])
# You can save this data in many different forms to further analyze and access for later
#df.to_csv('apple.csv', sep=';') # separate by ;, in case your data contains commas
#df.to_excel('apple.xlsx')
#df.to_html('apple.html')
#df.to_json('apple.json')
# You can load data back in any form as well
#df = pd.read_csv('apple.csv', sep=';')
#df = pd.read_excel('apple.xlxs')
#df = pd.read_html('apple.html')
#df = pd.read_json('apple.json')
# We can also visualize the data, which is helpful in stock analysis
# Plotting styles can be found at https://bit.ly/2OSCTdm
import matplotlib.pyplot as plt
from matplotlib import style

df['Adj Close'].plot()
style.use('ggplot')
plt.ylabel('Adjusted Close')
plt.title('APPL Closing Share Prices')
plt.show()
# Comparing Stocks
style.use('ggplot')

apple = web.DataReader('AAPL', 'yahoo', start, end)
microsoft = web.DataReader('MSFT', 'yahoo', start, end)

apple['Adj Close'].plot(label='AAPL')
microsoft['Adj Close'].plot(label='MSFT')
plt.ylabel('Adjusted Close')
plt.title('APPL vs MSFT Closing Share Prices')
plt.legend(loc='upper left')
plt.show()
# when comparing companies with drastically different share prices, employ subplots
apple = apple
tesla = web.DataReader('TSLA', 'yahoo', start, end)

plt.subplot(211)
apple['Adj Close'].plot(color='blue')
plt.ylabel('Adjusted Close')
plt.title('AAPL Closing Share Prices')

plt.subplot(212)
tesla['Adj Close'].plot(color='red')
plt.ylabel('Adj Close')
plt.title('TSLA Closing Share Prices')

plt.tight_layout()
plt.show()
# candlestick charts
from mplfinance.original_flavor import candlestick_ohlc # mpl_finance is depreciated
import matplotlib.dates as mdates
# select columns in right order needed for candlestick
apple = apple[['Open', 'High', 'Low', 'Close']]
# format dates to numbers
apple.reset_index(inplace=True) # inplace of prior iteration of data?
apple['Date'] = apple['Date'].map(mdates.date2num)
# define subplot and graph
ax = plt.subplot()
candlestick_ohlc(ax, apple.values, width=5, colordown='r', colorup='g')
ax.grid()
ax.xaxis_date()
plt.title('AAPL Candlestick')
plt.show()
# plot multiple days to clean up visualization
apple = web.DataReader('AAPL', 'yahoo', start, end)
apple_ohlc = apple['Adj Close'].resample('10D').ohlc() # 10 days grouped per candlestick
apple_ohlc.reset_index(inplace=True)
apple_ohlc['Date'] = apple_ohlc['Date'].map(mdates.date2num)
# create a second subplot to display volume
apple_volume = apple['Volume'].resample('10D').sum()
ax1 = plt.subplot2grid((6,1),(0,0), rowspan=4, colspan=1) # 2/3 of graph space (rowspan)
ax1.title.set_text('10 Day AVG Candlestick')
ax2 = plt.subplot2grid((6,1),(4,0), rowspan=2, colspan=1, sharex=ax1) # 1/3 of graph space (rowspan)
ax2.title.set_text('10 Day AVG Volume')
ax1.xaxis_date()
candlestick_ohlc(ax1, apple_ohlc.values, width=5, colorup='g', colordown='r')
ax2.fill_between(apple_volume.index.map(mdates.date2num), apple_volume.values) # fills area below graph (fill_between)
plt.tight_layout()
plt.show()
# Analysis and statistics
# 100 day moving average
# define a new column which averages last 100 closing prices, minimum periods 0 so we can build to 100
apple['100d_ma'] = apple['Adj Close'].rolling(window = 100, min_periods = 0).mean()
# drop all not a number values
apple.dropna(inplace=True) # replace this with previous data
# visualize the 100 day moving average with closing prices, and volume
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((6,1), (4,0), rowspan=2, colspan=1)

ax1.plot(apple.index, apple['Adj Close'])
ax1.plot(apple.index, apple['100d_ma'])
ax1.title.set_text('100 Day Moving AVG')
ax2.fill_between(apple.index, apple['Volume'])
ax2.title.set_text('Volume')
# tighten layout to make for cleaner look
plt.tight_layout()
plt.show()
# percent change
apple['PCT_change'] = (apple['Close'] - apple['Open']) / apple['Open']
apple['HL_PCT'] = (apple['High'] - apple['Low']) / apple['Close']

ax1 = plt.subplot2grid((8,1), (0,0), rowspan=4, colspan=1)
ax2 = plt.subplot2grid((8,1), (4,0), rowspan=4, colspan=1)

ax1.plot(apple.index, apple['PCT_change'])
ax1.title.set_text('Closing Price % Change')
ax2.plot(apple.index, apple['HL_PCT'])
ax2.title.set_text('High-Low % Change')


plt.tight_layout()
plt.show()
