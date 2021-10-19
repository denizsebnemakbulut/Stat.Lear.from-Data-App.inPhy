import numpy as np; np.random.seed(50)
import matplotlib.pyplot as plt;plt.figure(figsize=(25, 25))
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def groundTruth(x): return np.sin(2*np.pi*x)

sizes= [5,10,20,50,100,1000]

for i,size in enumerate(sizes):
    plt.title("Size: {} ".format(size))
    i += 1
    plt.subplot(4,4,i)
    
    X = np.random.random(size = size)
    
    mu, sigma = 0, 0.05
    s = np.random.normal(mu, sigma, size)
    
    y = s + groundTruth(X)
    
    X = X[:,np.newaxis]
    y = y[:,np.newaxis]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    
    
    poly_reg = PolynomialFeatures(degree = 10)
    X_poly = poly_reg.fit_transform(X)
    pol_reg = LinearRegression()
    pol_reg.fit(X_poly, y)
    yPred = pol_reg.predict(X_poly)
    
    
    plt.scatter(X, y,color='m',label='pol_reg')
    X, yPred = zip(*sorted(zip(X,yPred)))
    plt.plot(X, yPred, color='pink',label='pol_reg prediction')
    plt.legend()  
    
plt.show()