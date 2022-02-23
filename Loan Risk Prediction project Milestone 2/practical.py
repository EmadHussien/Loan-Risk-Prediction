import pandas as pd
import preProcessing
import pickle

### Classification paractical exam :
### Loading data

#Load data
classData = pd.read_csv('prac.csv')
newData = classData

#print(classData)
cols=('CreditGrade','LoanStatus','BorrowerState','EmploymentStatus','IsBorrowerHomeowner','IncomeRange','ProsperRating (Alpha)')
newData = preProcessing.fill_Missing_Categories(newData, cols)
newData = preProcessing.Feature_Encoder(newData,cols)

numric_Cols = ('BorrowerAPR','EmploymentStatusDuration','CreditScoreRangeLower', 'CreditScoreRangeUpper','RevolvingCreditBalance'
               ,'BankcardUtilization','AvailableBankcardCredit','TotalTrades','DebtToIncomeRatio','TotalProsperPaymentsBilled')
newData = preProcessing.fill_missing_Numeric(newData,numric_Cols)

X = newData.iloc[:,0:23] #Features
Y = newData['ProsperRating (Alpha)'] #Label
X = X.drop(X.columns[[0, 1,2,3,7,9,10,11,12,13,15,16,17,18,19,20,21,22]], axis=1)  # df.columns is zero-based pd.Index


# load the model from disk
LOG_model = pickle.load(open('logesticRegression_model.sav', 'rb'))
y_pred = LOG_model.predict(X)
print("Y prediction " , y_pred)
print("Y actual ", list(Y))
result = LOG_model.score(X, Y)
print("Logistic Regerssion model Accuracy : " , result)

print("==============================================================")

# load the model from disk
Tree_model = pickle.load(open('decisionTree_model.sav', 'rb'))
y_pred = Tree_model.predict(X)
print("Y prediction " , y_pred)
print("Y actual ", list(Y))
result = Tree_model.score(X, Y)
print("Decision Tree model Accuracy : " , result)

print("==============================================================")


# load the model from disk
Svm_model = pickle.load(open('SVM_model.sav', 'rb'))
y_pred = Svm_model.predict(X)
print("Y prediction " , y_pred)
print("Y actual ", list(Y))
result = Svm_model.score(X, Y)
print("Svm model Accuracy : " , result)


##### Reg TEST TODO