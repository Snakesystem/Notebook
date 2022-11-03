import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Salary_Data.csv')
X = df.iloc[:,0]
y = df.iloc[:,1]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=1/3.0, random_state=0)
X_train = X_train.values.reshape(-1,1)

from sklearn import linear_model
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

#fig = plt.figure(figsize=12,8)
plt.scatter(X_train, y_train)
plt.plot(np.arrange(0,12,0.3))
regressor.predict(np.arrange(0,12,0.3).reshape(-1,1), color='red')
plt.title('Salary vs Experience')
plt.xlabel('Years of Esperience')
plt.ylabel('Salary')
plt.show()