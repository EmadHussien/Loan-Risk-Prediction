from sklearn.preprocessing import LabelEncoder
import numpy as np
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

def Feature_Encoder(X,cols):
    for c in cols:
        lbl = LabelEncoder()
        lbl.fit(list(X[c].values))
        X[c] = lbl.transform(list(X[c].values))
    return X


def fill_Missing_Categories(Data, Cols):
    for c in Cols:
        #print(c)
        Data[c] = Data[c].fillna(Data[c].value_counts().index[0])
    return Data

def fill_missing_Numeric(Data,Cols):
    for c in Cols :
        Data[c].fillna(Data[c].mean(), inplace=True)
    return Data

#Load data
data = pd.read_csv('LoanRiskClassification.csv')
newData = data

cols=('CreditGrade','LoanStatus','BorrowerState','EmploymentStatus','IsBorrowerHomeowner','IncomeRange','ProsperRating (Alpha)')
newData = fill_Missing_Categories(newData, cols)
newData = Feature_Encoder(newData,cols)

#print(newData)

numric_Cols = ('BorrowerAPR','EmploymentStatusDuration','CreditScoreRangeLower', 'CreditScoreRangeUpper','RevolvingCreditBalance'
               ,'BankcardUtilization','AvailableBankcardCredit','TotalTrades','DebtToIncomeRatio','TotalProsperPaymentsBilled')
newData = fill_missing_Numeric(newData,numric_Cols)


#print(newData.isna().sum())


corr = newData.corr()
#sns.heatmap(corr, annot=True)
#plt.show()


X = newData.iloc[:,0:23] #Features
Y = newData['ProsperRating (Alpha)'] #Label


#print(X.columns)

X = X.drop(X.columns[[0, 1,2,3,7,9,10,11,12,13,15,16,17,18,19,20,21,22]], axis=1)  # df.columns is zero-based pd.Index
#print(X.columns)
#print(X.shape)

def daata_ret():
    return X,Y
