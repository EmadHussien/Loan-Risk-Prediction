import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
# Import Libraries
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import preProcessing
import Models


#Loading data
X,Y = preProcessing.daata_ret()

#print(X)
#print(Y)

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=0)


#Models.logistic_Reg(X_train,y_train,X_test , y_test)
#Models.Tree(X_train,y_train,X_test , y_test)
#Models.svm_Model(X_train,y_train,X_test , y_test)

