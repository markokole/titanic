####################
#### Algorithms ####

##Stohastic Gradient Descent Classification
def sgdClassifier(X, y, Xpred):
    from sklearn.linear_model import SGDClassifier
    sgd_clf = SGDClassifier(random_state=42)
    sgd_clf.fit(X, y.values.ravel())
    # pred = sgd_clf.predict(X_test) #test on the training part of the data
    pred = sgd_clf.predict(Xpred)
    #print sgd_clf.score(X, y)

    return pred
#END sgdClassifier

##Logistic Regression
def logReg(X, y, Xpred):
    from sklearn.linear_model import LogisticRegression
    log_reg = LogisticRegression(random_state=42, C=32)
    log_reg.fit(X, y.values.ravel())
    # pred = sgd_clf.predict(X_test) #test on the training part of the data
    pred = log_reg.predict(Xpred)
    #print sgd_clf.score(X, y)

    return pred
#END sgdClassifier

##Gaussian Naive Bayes
def gauss(X, y, Xpred):
    from sklearn.naive_bayes import GaussianNB
    gauss_clf = GaussianNB()
    gauss_clf.fit(X, y.values.ravel())
    pred = gauss_clf.predict(Xpred)

    return pred
#END gauss

##Decision Tree Classifier
#Regularization - restrict the Decision Tree's freedom during training - by using hyperparameters
def tree(X, y, Xpred, type, file_name, sample_split):
    from sklearn.tree import DecisionTreeClassifier
    from utility import generate_graphics
    #tree_clf = DecisionTreeClassifier(min_samples_split=sample_split)
    #tree_clf = DecisionTreeClassifier(min_samples_leaf=10, min_samples_split=30, max_depth=4, criterion="gini")
    tree_clf = DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=15, presort=True)
    '''
    #is "samples" on the graphic tree
    print "min_samples_split", tree_clf.min_samples_split
    #guarantees minimum number of samples in a leaf
    print "min_samples_leaf", tree_clf.min_samples_leaf
    #controls the size of the tree to prevent overfitting
    print "max_depth", tree_clf.max_depth
    #
    print "class_weight", tree_clf.class_weight

    print "criterion", tree_clf.criterion
    '''

    tree_clf.fit(X, y)
    #print X.columns
    #print tree_clf.feature_importances_

    pred = tree_clf.predict(Xpred)
    feature_names = list(X.columns)
    target_names = ["0","1"]
    generate_graphics(type, file_name, tree_clf, feature_names, target_names)

    return pred

def random_forest(X, y, Xpred):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=500, n_jobs=-1)
    clf.fit(X, y.values.ravel())
    pred = clf.predict(Xpred)

    return pred

#### Algorithms ####
####################
