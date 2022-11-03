import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('50_Startups.csv')

X = df.iloc[:, -2]
y = df['Profit']

from sklearn.linear_model import LinearRegression
fig = plt.figure(figsize=(12,8))
plt.scatter(X,y)
plt.xlabel('Amound Soend')
plt.ylabel('profit')