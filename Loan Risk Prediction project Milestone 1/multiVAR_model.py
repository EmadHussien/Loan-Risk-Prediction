import numpy as np
from sklearn import linear_model
from sklearn import metrics
import time
import pickle

def Mul_features( X_train , y_train , X_test , y_test):
    liner_mul_features = linear_model.LinearRegression()
    start_train = time.time()
    liner_mul_features.fit(X_train, y_train)
    stop_train = time.time()
    train_time = stop_train - start_train

    # save the model to disk
    #filename = 'mul_Var_model.sav'
    #pickle.dump(liner_mul_features, open(filename, 'wb'))

    print("Train time for Multi Var model: "+str(train_time))
    prediction = liner_mul_features.predict(X_test)

    #print('Co-efficient of linear regression', liner_mul_features.coef_)
    #print('Intercept of linear regression model', liner_mul_features.intercept_)
    print('Mean Square Error Multi Var model: ', metrics.mean_squared_error(np.asarray(y_test), prediction))
    print("R2 Score for Multi var model:  ",metrics.r2_score(y_test, prediction))
    print("=========================================================================")