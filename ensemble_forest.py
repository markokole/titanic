#Aggregating predictions of each classifier and predict the class that gets the most votes
#Works also if each classifier is weak learner (does slightly better than random guessing) ->
# the ensemble can still be strong learner if there is sufficient number of weak learners and they are diverse
def voting_classifier(X, y, Xpred):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC

    #Soft Voting/Majority Rule for unfitted estimators
    from sklearn.ensemble import VotingClassifier

    #random numbers are always the same
    import numpy as np
    np.random.seed(15)

    #divide the dataset on train and test
    the_cut = 700
    X_train, X_test = X[:the_cut], X[the_cut:]
    y_train, y_test = y[:the_cut], y[the_cut:]

    log_clf = LogisticRegression()
    rnd_clf = RandomForestClassifier()
    svm_clf = SVC(probability=True)

    #rather using soft voting - delivers higher accuracy score
    #svm_clf = SVC()
    #voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)], voting='hard')

    #Soft voting - gives more weight to highly confident votes
    voting_clf = VotingClassifier( estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)]
                                 , voting='soft'
                                 , weights=[1,3,1]
                                 )
    #Hard voting: 0.8377, Soft voting: 0.8639

    voting_clf.fit(X_train, y_train.values.ravel())

    pred = voting_clf.predict(Xpred)

    '''
    #print out each classifier's accuracy
    from sklearn.metrics import accuracy_score
    print "Voting:", voting_clf.voting
    for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
        clf.fit(X_train, y_trainDF.values.ravel())
        y_pred = clf.predict(X_test)
        print(clf.__class__.__name__, round(accuracy_score(y_test, y_pred), 4))
    '''

    return pred

#Bagging: use same training algorithms for every predictor, train them on different random subsets
#Bagging - Bootstrap aggregating
#Sampling is performed with replacement
#The ensemble predicts on new instance by aggregating the predictions of predictors
def bagging(X, y, Xpred):
    from sklearn.ensemble import BaggingClassifier
    from sklearn.tree import DecisionTreeClassifier

    #If True -> Bagging else Pasting
    bootstrap_value = True

    #automatically performs soft voting if possible (possible = base estimator can estimate class probabilities)
    bag_clf = BaggingClassifier(#base estimator to fit on random subsets of the dataset
                                #tried with hyperparameters from the DecisionTreeClassifiers - no improvement
                                DecisionTreeClassifier()
                                #number of predictors in ensemble
                                , n_estimators=500
                                #by raising max samples, accuracy on training set was rising
                                , max_samples=300
                                #whether samples are drawn with replacement
                                #if true -> 63% of the training instances are sampled on average for each estimator.
                                #37% are not sampled -> out-of-bag instances - Not same for all predictors!
                                # these instances can be used for evaluation of estimator
                                , bootstrap=bootstrap_value
                                , oob_score=True
                                #number of jobs to run parallel - -1 -> number of cores
                                , n_jobs=1
                               )

    bag_clf.fit(X, y.values.ravel())
    #print "oob score:", bag_clf.oob_score_

    pred = bag_clf.predict(Xpred)

    return pred

#ensemble of Decision Trees
def random_forest(X, y, Xpred):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(
                                  n_estimators=100
                                , n_jobs=-1
                                , max_leaf_nodes=16
                                , max_depth=5
                                )
    clf.fit(X, y.values.ravel())
    pred = clf.predict(Xpred)

    return pred
