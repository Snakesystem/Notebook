#import necessery modules
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(1,100,100)
y = X*2

y[10:30] = np.random.rand(20)*120+100

#Needed x and y with only 1 features
X = X.reshape(-1,1)
y = y.reshape(-1,1)

def plot(clf=None, clf_name = "", color = None):
    fig = plt.figure(figsize=(15,8))
    plt.scatter(X,y, lebel="Samples")
    plt.title("Made up data with outliers")
    if clf is not None:
        y_pred = clf.predict(X)
        plt.plot(X, y_pred, label = clf_name, color=color)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
    
plot()

from sklearn.linear_model import LinearRegression
lr = LinearRegression().fit(X,y)
plt(ir, 'OLS', 'green')