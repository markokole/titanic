import pandas as pd
from sklearn.metrics import accuracy_score

from feature_engineering import prepare_data
from algorithms import sgdClassifier, logReg, gauss, tree
from utility import aboutDF, writeCsv, write_log, stat, create_dataframe, feature_importance_on_survived
from ensemble_forest import voting_classifier, bagging, random_forest

accuracySGD = accuracyGauss = accuracyLogReg = accuracyTree = accuracyRandomForest = accuracyVoting = accuracyBagging = 0

#data used for training the model
file_train_data = r"C:\marko\kaggle\titanic\csv\train.csv"
#create a dataframe with train dataset - dataset is AS-IS in the source CSV file
initialDF = create_dataframe(file_train_data)

#use this one to check the importance of the features against Survived
#feature_importance_on_survived(initialDF)

#run feature engineering to prepare dataframe with features
trainDF = prepare_data(initialDF)
#Create dataframe with labels
y_trainDF = initialDF[["Survived"]]


#print out stats about this dataframe
#stat(trainDF, y_trainDF)

#trainDF features
'''
1 PassengerId
2 Pclass
3 Fare
4 Title
5 EmbarkedClass
6 SexCode
7 AgeMinusOne
8 Age
9 AgeClass
10 FamilySize
11 Female
12 Male
'''


'''
sgd_features = ["AgeClass", "Pclass", "Title", "EmbarkedClass", "SexCode", "FamilySize"]
X_train = trainDF[sgd_features]
#predict with Stohastic Gradient Descent classifier
sgdPrediction = sgdClassifier(X_train, y_trainDF, X_train)
print "===================="
accuracySGD = round(accuracy_score(y_trainDF, sgdPrediction), 4)
print "Accuracy with Stohastic Gradient Descent: ", accuracySGD


#predict with Gaussian Naive Bayes
gaussPrediction = gauss(X_train, y_trainDF, X_train)
print "===================="
accuracyGauss = round(accuracy_score(y_trainDF, gaussPrediction), 4)
print "Accuracy with Gaussian Naive Bayes: ", accuracyGauss


#predict with Logistic Regression
logisticRegression = logReg(X_train, y_trainDF, X_train)
print "===================="
accuracyLogReg = round(accuracy_score(y_trainDF, logisticRegression), 4)
print "Accuracy with Logistic Regression: ", accuracyLogReg
'''



#dtc_features = ["Age", "Pclass", "Title", "SexCode", "EmbarkedClass", "FamilySize"]
#dtc_features = ["AgeClass", "Pclass", "Title", "EmbarkedClass", "Age", "SexCode"]
#dtc_features = ["Pclass", "Title", "EmbarkedClass", "AgeMinusOne", "SexCode"]
#dtc_features = ["SexCode", "Pclass", "Fare", "Age"]
dtc_features = trainDF.columns
X_train = trainDF[dtc_features]


#predict with Decision Tree Classifier
print "===================="
treePrediction = tree(X_train, y_trainDF, X_train, "png", "tree_ageminusone.png")
accuracyTree = round(accuracy_score(y_trainDF, treePrediction), 4)
print "Accuracy with Decision Tree Classifier: ", accuracyTree


#predict with Random Forest Classifier
print "===================="
randomForestPrediction = random_forest(X_train, y_trainDF, X_train)
accuracyRandomForest = round(accuracy_score(y_trainDF, randomForestPrediction), 4)
print "Accuracy with Random Forest Classifier: ", accuracyRandomForest


#predict with Voting Classifier
print "===================="
votingPrediction = voting_classifier(X_train, y_trainDF, X_train)
accuracyVoting = round(accuracy_score(y_trainDF, votingPrediction), 4)
print "Accuracy with Voting Classifier: ", accuracyVoting


#predict with Bootstrap Aggregating
print "===================="
baggingPrediction = bagging(X_train, y_trainDF, X_train)
accuracyBagging = round(accuracy_score(y_trainDF, baggingPrediction), 4)
print "Accuracy with Bagging Classifier: ", accuracyBagging


###TEST
#path to test file
file_test_data = r"C:\marko\kaggle\titanic\csv\test.csv"
#load data into a DataFrame
testDF = pd.read_csv(file_test_data, sep=",", header=0)
#passengerId Dataframe for sewing together the id with prediction result for the output
passengerIdDF = testDF["PassengerId"]
#prepare the test data
#X_test = prepare_data(testDF)[features]


#Decision Tree Classifier
X_test = prepare_data(testDF)[dtc_features]


#predict SGD
#predictOnTest = sgdClassifier(X_train, y_trainDF, X_test)
#predict Naive Bayes
#predictOnTest = gauss(X_train, y_trainDF, X_test)
#predict Logistic Regression
#predictOnTest = logReg(X_train, y_trainDF, X_test)
#predict Decision Tree Classifier - creates a png file of tree
predictOnTest = tree(X_train, y_trainDF, X_test, "png", "tree5.png")
#predict Random Forest Classifier
#predictOnTest = random_forest(X_train, y_trainDF, X_test)
#predict Voting Classifier
predictOnTest = voting_classifier(X_train, y_trainDF, X_test)
#predict Bagging Classifier
#predictOnTest = bagging(X_train, y_trainDF, X_test)

comment = "All features, AVG for fare=NA. To Kaggle"

write_log(list(X_train.columns), accuracySGD, accuracyGauss, accuracyLogReg, accuracyTree, accuracyRandomForest, accuracyVoting, accuracyBagging, comment)

#writeCsv("csv/sgd.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/gauss.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/logistic.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/tree.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/random_forest.csv", passengerIdDF, predictOnTest)
writeCsv("csv/voting.csv", passengerIdDF, predictOnTest)
#writeCsv("csv/bagging.csv", passengerIdDF, predictOnTest)

