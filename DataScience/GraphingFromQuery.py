import pyodbc
import datetime
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

con = pyodbc.connect('Driver={SQL Server};Server=solutionsprod;Database=pos;Trusted_Connection=yes;')
cursor = con.cursor()
Ssql = '''select month(ts) as Month, sum(price * qty - discount) as Sales
        from pos.dbo.tbltranssalesdtl
        where cast(ts as Date) between '2021-01-01' and '2021-12-31'
        group by month(ts)
        order by month(ts)'''
Psql = '''select  month(ts) as Month, sum(qty) as Production
        from pos.dbo.tbltransproddtl
        where cast(ts as Date) between '2021-01-01' and '2021-12-31'
        group by month(ts)
        order by month(ts)'''

graphArray = []

for row in cursor.execute(Ssql):
    startingData = str(row).replace('(', '').replace(')', '').replace('Decimal', '').replace("'", '')
    splitInfo = startingData.split(',')
    graphArrayAppend = splitInfo[0]+','+splitInfo[1]
    graphArray.append(graphArrayAppend)

graphArray2 = []

for row in cursor.execute(Psql):
    startingData2 = str(row).replace('(', '').replace(')', '').replace('Decimal', '').replace("'", '')
    splitInfo2 = startingData2.split(',')
    graphArrayAppend2 = splitInfo2[0]+','+splitInfo2[1]
    graphArray2.append(graphArrayAppend2)

month, value = np.loadtxt(graphArray,delimiter=',', unpack=True)
month2, value2 = np.loadtxt(graphArray2,delimiter=',', unpack=True)

plt.plot(month, value, 'go--', linewidth=2, markersize=12)
plt.plot(month2, value2, 'bo--', linewidth=2, markersize=12)
plt.show()
