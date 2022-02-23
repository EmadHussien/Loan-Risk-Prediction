import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
import numpy as np

from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

### Loading data

#Load data
RegData = pd.read_csv('pracReg.csv')

# fill missing values
RegData["IsBorrowerHomeowner"].fillna(RegData["IsBorrowerHomeowner"].mean(),inplace=True)
RegData["CreditScoreRangeLower"].fillna(RegData["CreditScoreRangeLower"].mean(),inplace=True)
RegData["CreditScoreRangeUpper"].fillna(RegData["CreditScoreRangeUpper"].mean(),inplace=True)
RegData["AvailableBankcardCredit"].fillna(RegData["AvailableBankcardCredit"].mean(),inplace=True)
RegData["LoanOriginalAmount"].fillna(RegData["LoanOriginalAmount"].mean(),inplace=True)
RegData["LoanRiskScore"].fillna(RegData["LoanRiskScore"].mean(),inplace=True)

X = RegData.drop(RegData.columns[[0, 1,2,3,4,5,6,7,8,9,13,14,16,17,18,19,20,21,23]], axis=1)  # df.columns is zero-based pd.Index
Y = RegData['LoanRiskScore'] #Label

#encoding
lbl = LabelEncoder()
lbl.fit(list(X['IsBorrowerHomeowner'].values))
X['IsBorrowerHomeowner']= lbl.transform(list(X['IsBorrowerHomeowner'].values))

#print(X.columns)
#print(Y)


# load the model from disk
multiVar_model = pickle.load(open('mul_Var_model.sav', 'rb'))
y_pred = multiVar_model.predict(X)
print("Y prediction " , y_pred)
print("Y actual ", list(Y))
#result = multiVar_model.score(X, Y)
#print("multi Variable model model Accuracy : " , result)
print('Mean Square Error Multi Var model: ', metrics.mean_squared_error(np.asarray(Y), y_pred))
print("R2 Score for Multi var model:  ", metrics.r2_score(Y, y_pred))

print("==============================================================")

# load the model from disk
poly_features = PolynomialFeatures(degree=3)
poly_model = pickle.load(open('polynomial_model.sav', 'rb'))
y_pred_poly = poly_model.predict(poly_features.fit_transform(X))
print("Y prediction " , y_pred_poly)
print("Y actual ", list(Y))
print('Mean Square Error polynomial model: ', metrics.mean_squared_error(np.asarray(Y), y_pred_poly))
print("R2 Score for Multi var model:  ", metrics.r2_score(Y, y_pred_poly))








