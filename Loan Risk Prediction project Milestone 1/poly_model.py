import numpy as np
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
import pickle

def Poly(  X_train , y_train , X_test , y_test ):
    poly_features = PolynomialFeatures(degree=3)

    X_train_poly = poly_features.fit_transform(X_train)
    poly_model = linear_model.LinearRegression()

    start_train = time.time()
    poly_model.fit(X_train_poly, y_train)
    stop_train = time.time()
    train_time = stop_train - start_train
    #filename = 'polynomial_model.sav'
    #pickle.dump(poly_model, open(filename, 'wb'))


    print("Train time for polynomial model: "+str(train_time))
    prediction = poly_model.predict(poly_features.fit_transform(X_test))
    print('Mean Square Error Polynomial model', metrics.mean_squared_error(y_test, prediction))
    print("R2 Score for poly model:  ",metrics.r2_score(y_test, prediction))
