import bs4 as bs
import requests
import pickle

def load_sp500_tickers():
    link = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(link)
    # response makes an HTTP request
    soup = bs.BeautifulSoup(response.text, 'lxml')
    # we choose the parser of lxml
    table = soup.find('table', {'class':'wikitable sortable'})
    # table object filters HTML and returns only the table we want
    # the find function finds the element you want (table) and sictionary with requirements of wikitable and sortable
    # Lets create an empty list and fill it with entries from the table
    tickers = []

    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text[:-1]
        tickers.append(ticker)

    # we will serialize our ticker object and save it so we don't have to scrape the web everytime
    with open('sp500tickers.pickle', 'wb') as f:
        pickle.dump(tickers, f)

    return tickers
    # we get all elements that are a table row, exclude the first on (header), get all elements from first column (0)
    # the [:-1] cuts off the line escape character

# Now lets load the financial data for all of the ticker symbols
import os # (operating system basic functions)
import datetime as dt
import pandas_datareader as web
import pandas as pd

def load_prices(reload_tickers=False):
# reload the ticker symbols if the file path already exists, if not webscrape again
    if reload_tickers:
        tickers = load_sp500_tickers()
    else:
        if os.path.exists('sp500tickers.pickle'):
            with open('sp500tickers.pickle', 'rb') as f:
                tickers = pickle.load(f)
    # create a CSV file for every ticker symbol
    if not os.path.exists('companies'):
        os.makedirs('companies')
    # pick start and end date for the data
    start = dt.datetime(2020,1,1)
    end = dt.datetime.now()
    # check to see if csv exists, if not fetch and write data to CSV
    for ticker in tickers:
        if not os.path.exists('companies/{}.csv'.format(ticker)):
            print("{} is loading...".format(ticker))
            df = web.DataReader(ticker.replace('.','-'), 'yahoo', start, end) #may have to clean some ticker symbols like BRK.B
            df.to_csv('companies/{}.csv'.format(ticker))
        else:
            print("{} already downloaded!".format(ticker))

def compile_data():
    with open('sp500tickers.pickle', 'rb') as f:
        tickers = pickle.load(f)
# we are extracting the ADJ close prices for all companies and putting them in one dataframe
    main_df = pd.DataFrame()

    print("Compiling data...")
    for ticker in tickers:
        df = pd.read_csv('companies/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)

        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Close', 'Volume'], 1, inplace=True)

        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
# iterates through all tickers and loads CSV, set index as date column since it is common index
# rename the Adj close column so can identify different company
# then check if df empty, if not outer join all columns
    main_df.to_csv('sp500_data.csv')
    print('Data compiled!')

load_prices(reload_tickers=False)
compile_data()

import matplotlib.pyplot as plt
# how to call the data back
sp500 = pd.read_csv('sp500_data.csv')
sp500['MSFT'].plot()
plt.show()

# correlation for all stocks in sp500
correlation = sp500.corr()
correlation.to_csv('sp500_correlation')
print(correlation)
# visualize the correslation
plt.matshow(correlation)
plt.show()
# the more yellow a point is, the higher the correlation
