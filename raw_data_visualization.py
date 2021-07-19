import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import sys

# Get data
print('Loading data')
dir_path = '/Users/bellanicholson/Desktop/DA Case - Real Estate Analytics'
file_path = os.path.join(dir_path, 
						 'Staten_Island_housing_market_case.xlsx')
save_path = os.path.join(dir_path, 'figures')
data = pd.read_csv("Staten_Island_housing_market_case.csv", low_memory=False)
print('File loaded.')

# Raw Data Visualization
years = data.year
median_price = data.groupby(['year']).price.median() #.to_frame()

# Raw insights plot - Price Median Data
fig, ax = plt.subplots()
ax.plot(median_price)

ax.set(xlabel='year', ylabel='median price') 
ax.grid()

fig.savefig(os.path.join(save_path, "test.png"))
plt.show()