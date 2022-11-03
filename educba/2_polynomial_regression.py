import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Analisis data
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2]
y = df.iloc[:, 2]

#Membuat model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

#Preprocessing Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg_2 = PolynomialFeatures(degree=2)
poly_reg_3 = PolynomialFeatures(degree=3)
X_poly_2 = poly_reg_2.fit_transform(X)
X_poly_3 = poly_reg_3.fit_transform(X)

lin_reg_poly_2 = LinearRegression().fit(X_poly_2, y)
lin_reg_poly_3 = LinearRegression().fit(X_poly_3, y)

#Data Visualisasi
plt.scatter(X, y, color="black")
plt.plot(X, lin_reg.predict(X), color="b")
plt.plot(X, lin_reg_poly_2.predict(poly_reg_2.fit_transform(X)), color="g")
plt.plot(X, lin_reg_poly_3.predict(poly_reg_3.fit_transform(X)), color="r")
plt.show()