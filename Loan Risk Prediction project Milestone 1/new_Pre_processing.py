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
'''
def featureScaling(X,a,b):
    X = np.array(X)
    Normalized_X=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        Normalized_X[:,i]=((X[:,i]-min(X[:,i]))/(max(X[:,i])-min(X[:,i])))*(b-a)+a
    return Normalized_X

'''


#Load players data
data = pd.read_csv('LoanRiskScore.csv')

#Drop the rows that contain missing values
#data.dropna(how='any',inplace=True)


#Deal with missing values
print(data.isna().sum())
#print(data.info())
data["LoanRiskScore"].fillna(data["LoanRiskScore"].mean(),inplace=True)
data["BorrowerAPR"].fillna(data["BorrowerAPR"].mean(),inplace=True)
data["EmploymentStatusDuration"].fillna(data["EmploymentStatusDuration"].mean(),inplace=True)
data["CreditScoreRangeLower"].fillna(data["CreditScoreRangeLower"].mean(),inplace=True)
data["RevolvingCreditBalance"].fillna(data["RevolvingCreditBalance"].mean(),inplace=True)
data["CreditScoreRangeUpper"].fillna(data["CreditScoreRangeUpper"].mean(),inplace=True)
data["BankcardUtilization"].fillna(data["BankcardUtilization"].mean(),inplace=True)
data["AvailableBankcardCredit"].fillna(data["AvailableBankcardCredit"].mean(),inplace=True)
data["TotalTrades"].fillna(data["TotalTrades"].mean(),inplace=True)
data["DebtToIncomeRatio"].fillna(data["DebtToIncomeRatio"].mean(),inplace=True)
data["TotalProsperPaymentsBilled"].fillna(data["TotalProsperPaymentsBilled"].mean(),inplace=True)

print(data.isna().sum())




#Drop the rows that contain missing values
#data.dropna(how='any',inplace=True)
loan_data=data.iloc[:,:]


#Feature Encoding
cols=('CreditGrade','LoanStatus','BorrowerState','EmploymentStatus','IsBorrowerHomeowner','IncomeRange')
loan_data = Feature_Encoder(loan_data,cols)
X=loan_data.iloc[:,0:23] #Features
Y=loan_data['LoanRiskScore'] #Label


#loan_data = loan_data.drop(X.columns[[1,7,8]], axis=1)  # df.columns is zero-based pd.Index
#print("AFTER DROPPPING")

#print(loan_data['LoanRiskScore'])

#Feature Selection
#Get the correlation between the features
corr = loan_data.corr()
#Top 50% Correlation training features with the Value
#top_feature = corr.index[abs(corr['LoanRiskScore']>0.5)]
#Correlation plot
#plt.subplots(figsize=(12, 8))
#top_corr = loan_data[top_feature].corr()
sns.heatmap(corr, annot=True)
plt.show()

X = X.drop(X.columns[[0, 1,2,3,4,5,6,7,8,9,13,14,16,17,18,19,20,21]], axis=1)  # df.columns is zero-based pd.Index
print(X.columns)
print(X.shape)

def daata_ret():
    return X,Y
