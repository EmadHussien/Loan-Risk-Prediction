from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
import matplotlib.pyplot as plt
import pickle

def logistic_Reg(X_train,y_train,X_test , y_test):
    # instantiate the model (using the default parameters)
    logreg = LogisticRegression(solver='liblinear', C=10.0)
    # fit the model with data
    start_train = time.time()
    logreg.fit(X_train, y_train)
    stop_train = time.time()

    # save the model to disk
   # filename = 'logesticRegression_model.sav'
   # pickle.dump(logreg, open(filename, 'wb'))

    #testing
    start_tst = time.time()
    y_pred = logreg.predict(X_test)
    stop_tst = time.time()
    # print(y_pred)
    log_Accuracy = metrics.accuracy_score(y_test, y_pred)
    train_time = stop_train - start_train
    test_time = stop_tst - start_tst
    print("logistic Reg Info : ")
    print("Accuracy  :",log_Accuracy )
    print("train time: ", train_time)
    print("testing time: ", test_time)
    print("=====================================================================")


    # making the bar plot
    data = {'classification accuracy': log_Accuracy,
            'training time': train_time, 'testing time': test_time}
    labels = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='maroon', width=0.4)
    plt.xlabel("performance measures")
    plt.ylabel("values")
    plt.title("logistic regression")
    plt.show()
    return  None





def Tree(X_train,y_train,X_test , y_test):

    # Decision tree
    model_tree = DecisionTreeClassifier(max_depth=15)  # max_depth controls the accuracy
    start_train = time.time()
    model_tree.fit(X_train, y_train)
    stop_train = time.time()

    # save the model to disk
    #filename = 'decisionTree_model.sav'
    #pickle.dump(model_tree, open(filename, 'wb'))

    #test
    start_tst = time.time()
    y_pred_Tree = model_tree.predict(X_test)
    stop_tst = time.time()
    Tree_Accuracy = metrics.accuracy_score(y_test, y_pred_Tree)
    train_time = stop_train - start_train
    test_time = stop_tst - start_tst

    print("Decision Tree Info : ")
    print("Accuracy   :", Tree_Accuracy)
    print("train time: ", train_time)
    print("testing time: ", test_time)
    print("=====================================================================")

    # making the bar plot
    data = {'classification accuracy': Tree_Accuracy,
            'training time': train_time, 'testing time': test_time}
    labels = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='maroon', width=0.4)
    plt.xlabel("performance measures")
    plt.ylabel("values")
    plt.title(" Decision tree")
    plt.show()

    return  None

def svm_Model(X_train,y_train,X_test , y_test):
    # Create a svm Classifier
    model_svm = svm.SVC(kernel='linear')  # Linear Kernel

    # Train the model using the training sets
    start_train = time.time()
    model_svm.fit(X_train, y_train)
    stop_train = time.time()
    # save the model to disk
    #filename = 'SVM_model.sav'
    #pickle.dump(model_svm, open(filename, 'wb'))

    #test
    # Predict the response for test dataset
    start_tst = time.time()
    y_pred_svm = model_svm.predict(X_test)
    stop_tst = time.time()
    SVM_Accuracy = metrics.accuracy_score(y_test, y_pred_svm)
    train_time = stop_train - start_train
    test_time = stop_tst - start_tst


    print("SVM model Info : ")
    print("Accuracy   :", SVM_Accuracy)
    print("train time: ", train_time)
    print("testing time: ", test_time)
    print("=====================================================================")

    # making the bar plot
    data = {'classification accuracy': SVM_Accuracy,
            'training time': train_time, 'testing time': test_time}
    labels = list(data.keys())
    values = list(data.values())
    fig = plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color='maroon', width=0.4)
    plt.xlabel("performance measures")
    plt.ylabel("values")
    plt.title(" SVM ")
    plt.show()
    return  None
