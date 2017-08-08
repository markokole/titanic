def aboutDF(df):
   #describe the dataset and get familiar with it.
    #891 instances
    #Age feature has some unavailable values (NaN)
    #Gender (Sex) is in text form male/female
    #Pclass is either 1, 2 or 3
    #The label (Survived) is either 0 or 1

    print df.describe()
    print "------------"
    print df.info()
    print "------------"
    print df.head(5)

def stat(df):
    #count each value for feature Age

    ##Correlation matrix & Mean
    #X_train["Survived"] = trainDF[["Survived"]]
    corr_matrix = df.corr()
    print corr_matrix
    print df[["Age", "Title", "Survived"]].groupby(["Title"]).mean()
    #X_train = X_train.drop(["Survived"], axis=1)
    ##Correlation matrix



#path: path to where the file is written
#id: passengerId - type DataFrame
#pred: prediction - type DataFrame
def writeCsv(path, id, pred):
    #zip the DataFrames into a List
    res = zip(id, pred)
    # write the result to a csv file
    import csv
    with open(path, "w") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow(["PassengerId", "Survived"])
        for line in res:
            writer.writerow(line)
    print "Output has been written to ", path
#END writeCsv

def writeLog(cols, sgd, gnb, lr, dtc, rfc, comment):
    import csv, time
    columns = str(cols).replace("[", "").replace("]", "").replace("'", "")
    t = time.strftime("%Y/%m/%d %H:%M:%S")
    with open("log\log.log", "a") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([t, columns, sgd, gnb, lr, dtc, rfc, comment])


def generate_graphics(type, file_name, clf, feature_names, target_names):
    from sklearn import tree
    import os
    import pydotplus

    #Add Graphviz to the path
    os.environ["PATH"] += os.pathsep + r"C:\Python27\Lib\site-packages\graphviz2.38\bin"

    dot_data = tree.export_graphviz(clf, out_file=None,
                         feature_names=feature_names,
                         class_names=target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)

    file_name = r"graphic\\" + file_name

    if type == 'png':
        graph.write_png(file_name)
    elif type == 'pdf':
        graph.write_pdf(file_name)
    else:
        print "Unknown file type!"
        exit(0)

