import pandas as pd

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

#feature importance in relationship to label Survived
def feature_importance_on_survived(df):
    df["Sex10"] = df["Sex"].map(lambda x: 1 if x == 'male' else 0)
    corr_matrix = df.corr()
    print corr_matrix["Survived"].sort_values(ascending=False)
    df.drop("Sex10", axis=1)


#print out some statistics about the features and label
def stat(df, label_df):

    print "Available features:"
    i = 0
    for c in df.columns:
        i += 1
        print i, c
    # print 3 instances
    print df.head(3)

    #Add survived label for easier correlation matrix and grouping
    df["Survived"] = label_df

    ##Correlation matrix & Mean
    #X_train["Survived"] = trainDF[["Survived"]]
    corr_matrix = df.corr()
    print corr_matrix
    print df[["AgeClass", "Title", "Survived"]].groupby(["Title"]).mean()
    #X_train = X_train.drop(["Survived"], axis=1)
    ##Correlation matrix

    df.drop(["Survived"], axis=1)


def create_dataframe(csv_file):
    # data used for training the model
    #file_train_data = csv_file
    # dataframe with training data
    df = pd.read_csv(csv_file, sep=",", header=0)
    # trainDF = pd.read_csv(file_train_data, sep=",", header=0).drop(["Cabin"], axis=1).dropna()

    return df


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

def write_log(cols, sgd, gnb, lr, dtc, rfc, vc, bc, comment):
    import csv, time
    columns = str(cols).replace("[", "").replace("]", "").replace("'", "")
    t = time.strftime("%Y/%m/%d %H:%M:%S")
    with open("log\log.log", "a") as output:
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([t, columns, "sgd: "+str(sgd), "gnb: "+str(gnb), "lr: "+str(lr), "dtc: "+str(dtc), "rfc: "+str(rfc), "vc: "+str(vc), "bc: "+str(bc), comment])


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
