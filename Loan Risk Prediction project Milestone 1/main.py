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
import multiVAR_model
import poly_model
import new_Pre_processing

#Loading data
X,y = new_Pre_processing.daata_ret()




X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, shuffle=True)

multiVAR_model.Mul_features(X_train, y_train , X_test,  y_test)

poly_model.Poly(X_train, y_train , X_test,  y_test)


