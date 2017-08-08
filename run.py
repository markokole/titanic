import pandas as pd
from sklearn.metrics import accuracy_score

from feature_engineering import prepare_data
from algorithms import sgdClassifier, logReg, gauss, tree, random_forest
from utility import aboutDF, writeCsv, writeLog, stat

#features = ["Pclass", "Title", "EmbarkedClass", "SexCode", "Age", "AgeClass", "Female", "Male"]
#features = ["Pclass", "Title", "EmbarkedClass", "SexCode", "Age", "Female", "Male"]
#features = ["SexCode"]
#features = ["AgeClass", "Pclass", "Title", "EmbarkedClass", "Age", "Female", "Male"]
features = ["AgeClass", "Pclass", "Title", "EmbarkedClass", "Age", "SexCode"]

#data used for training the model
file_train_data = r"C:\marko\kaggle\titanic\csv\train.csv"
#dataframe with training data
trainDF = pd.read_csv(file_train_data, sep=",", header=0)
#trainDF = pd.read_csv(file_train_data, sep=",", header=0).drop(["Cabin"], axis=1).dropna()
X_train = prepare_data(trainDF)[features]

#trainDF["Title"] = trainDF["Name"].map(lambda s: s[s.index(",") + 2:str(s).index(".")])
#stat(trainDF)

#print X_train[:10]

y_train = trainDF[["Survived"]]

#predict with Stohastic Gradient Descent classifier
sgdPrediction = sgdClassifier(X_train, y_train, X_train)
print "===================="
accuracySGD = round(accuracy_score(y_train, sgdPrediction), 4)
print "Accuracy with Stohastic Gradient Descent: ", accuracySGD


#predict with Gaussian Naive Bayes
gaussPrediction = gauss(X_train, y_train, X_train)
print "===================="
accuracyGauss = round(accuracy_score(y_train, gaussPrediction), 4)
print "Accuracy with Gaussian Naive Bayes: ", accuracyGauss


#predict with Logistic Regression
logisticRegression = logReg(X_train, y_train, X_train)
print "===================="
accuracyLogReg = round(accuracy_score(y_train, logisticRegression), 4)
print "Accuracy with Logistic Regression: ", accuracyLogReg


#predict with Decision Tree Classifier
print "===================="
treePrediction = tree(X_train, y_train, X_train, "png", "tree.png", 5)
accuracyTree = round(accuracy_score(y_train, treePrediction), 4)
print "Accuracy with Decision Tree Classifier: ", accuracyTree

#predict with Random Forest Classifier
print "===================="
randomForestPrediction = random_forest(X_train, y_train, X_train)
accuracyRandomForest = round(accuracy_score(y_train, randomForestPrediction), 4)
print "Accuracy with Random Forest Classifier: ", accuracyRandomForest


###TEST
#path to test file
file_test_data = r"C:\marko\kaggle\titanic\csv\test.csv"
#load data into a DataFrame
testDF = pd.read_csv(file_test_data, sep=",", header=0)
#passengerId Dataframe for sewing together the id with prediction result for the output
passengerIdDF = testDF["PassengerId"]
#prepare the test data
X_test = prepare_data(testDF)[features]

#predict SGD
#predictOnTest = sgdClassifier(X_train, y_train, X_test)
#predict Naive Bayes
#predictOnTest = gauss(X_train, y_train, X_test)
#predict Logistic Regression
#predictOnTest = logReg(X_train, y_train, X_test)
#predict Decision Tree Classifier
#predictOnTest = tree(X_train, y_train, X_test, "png", "tree5.png", 5)
#predict Random Forest Classifier
predictOnTest = random_forest(X_train, y_train, X_test)

comment = "Random Forest Classifier added"

#writeLog(list(X_train.columns), accuracySGD, accuracyGauss, accuracyLogReg, accuracyTree, accuracyRandomForest, comment)

#writeCsv("csv/sgd.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/gauss.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/logistic.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/tree.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/random_forest.csv", passengerIdDF, predictOnTest)
