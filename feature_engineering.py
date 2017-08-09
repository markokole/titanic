import numpy as np

#method takes an age and assigns it to a class
def ageClass(n):
    if (n <= 16):
        ageC = 0
    elif (n <= 30):
        ageC = 1
    elif (n <= 48):
        ageC = 2
    elif (n <= 99):
        ageC = 3
    else:
        ageC = 4
    return ageC

#attempt to classify fare
#attempt has been dropped - fare is too complex.
def fareClass(n):
    if (n >= 500):
        fareC = 0
    elif (n <= 200):
        fareC = 1
    elif (n <= 100):
        fareC = 2
    elif (n <= 50):
        fareC = 3
    else:
        fareC = 4
    return fareC

def titleClass(n):
    titleC = ""
    s = str(n)
    title =s[s.index(",") + 2:str(s).index(".")]

    '''
    #gender divided
    if (title in ['Mr', 'Master', 'Rev', 'Don', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']):
        titleC = 0
    elif (title in ['Mrs', 'Miss', 'Mme', 'Ms', 'Lady', 'Mlle', 'the Countess']):
        titleC = 1
    '''

    '''
    #class divided
    if (title in ['Mr', 'Master', 'Mrs', 'Miss', 'Mme', 'Ms', 'Mlle']):
        titleC = 0
    elif (title in ['Rev', 'Don', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer', 'Lady', 'the Countess']):
        titleC = 1
    else:
        titleC = 2
    return titleC
    '''

    # class divided - detailed
    #young boys
    if (title in ['Master']):
        titleC = 0
    #young females - prime of life,
    elif (title in ['Mrs', 'Miss', 'Mme', 'Ms', 'Mlle']):
        titleC = 1
    #older females - high status symbol
    elif (title in ['Lady', 'the Countess']):
        titleC = 2
    #younger men - from biological perspective: worth least
    elif (title in ['Mr']):
        titleC = 3
    #older men - wouldnt board out of honor, too old to survive
    elif (title in ['Rev', 'Don', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']):
        titleC = 4
    else:
        titleC = 5
    return titleC


    '''
    if (title == 'Mr'):
        titleC = 0
    elif (title == 'Mrs'):
        titleC = 1
    elif (title == 'Miss'):
        titleC = 2
    elif (title == 'Master'):
        titleC = 3
    elif (title == 'Rev'):
        titleC = 4
    elif (title == 'Don'):
        titleC = 5
    elif (title == 'Dr'):
        titleC = 6
    elif (title == 'Major'):
        titleC = 7
    elif (title == 'Mme'):
        titleC = 8
    elif (title == 'Ms'):
        titleC = 9
    elif (title == 'Lady'):
        titleC = 10
    elif (title == 'Sir'):
        titleC = 11
    elif (title == 'Mlle'):
        titleC = 12
    elif (title == 'Col'):
        titleC = 13
    elif (title == 'Capt'):
        titleC = 14
    elif (title == 'the Countess'):
        titleC = 15
    elif (title == 'Jonkheer'):
        titleC = 16
    else:
        titleC = 99
    return titleC
    Accuracy with Stohastic Gradient Descent:  0.772166105499
====================
Accuracy with Gaussian Naive Bayes:  0.785634118967

    '''

def embarkedClass(a):
    if (a == 'C'):
        return 0
    elif (a == 'Q'):
        return 1
    elif (a == 'S'):
        return 2
    else:
        return 2

#calculate average age for sex
def calculate_avg_age_sex(trainDF):
    #maleAvgAge = round(trainDF.where(trainDF["Sex"] == 'male')["Age"].mean(), 2)
    #femaleAvgAge = round(trainDF.where(trainDF["Sex"] == 'female')["Age"].mean(), 2)
    maleAvgAge = round(trainDF.where(trainDF["SexCode"] == 1)["Age"].mean(), 2)
    femaleAvgAge = round(trainDF.where(trainDF["SexCode"] == 0)["Age"].mean(), 2)

    return maleAvgAge, femaleAvgAge

#
def ageManipulation(df, maleAvgAge, femaleAvgAge):

    df["Age2"] = np.where((df["Age"].isnull() & (df["SexCode"] == 1)), maleAvgAge, df["Age"])
    df["Age2"] = np.where((df["Age2"].isnull() & (df["SexCode"] == 0)), femaleAvgAge, df["Age2"])
    #print df[["Sex", "Age", "Age2"]][:20]
    df = df.drop("Age", axis=1)
    df.rename(columns={'Age2': 'Age'}, inplace=True)

    return df

#
def prepare_data(initialDF):

    ##TRAIN SET
    #Pick the features:
    #Pclass - ticket class - 1 = 1st, 2 = 2nd, 3 = 3rd
    prepare_df = initialDF[["PassengerId", "Pclass", "Age", "Sex", "Name", "Embarked", "SibSp", "Parch", "Fare"]]

    prepare_df.is_copy = False

    #title
    #parse title from name feature
    prepare_df["Title"] = prepare_df["Name"].map(lambda x: titleClass(x))
    prepare_df = prepare_df.drop("Name", axis=1)
    #print prepare_df["Title"]

    #feature ActualAge is added to tell which instances have value for
    #  feature Age (1) and which do not (0)
    #prepare_df["ActualAge"] = np.where(x_trainDF["Age"].isnull(), 0, 1)
    #print "First 10 instances with feature ActualAge:\n", x_trainDF.head(10)

    #Embarked class
    prepare_df["EmbarkedClass"] = prepare_df["Embarked"].map(lambda x: embarkedClass(x))
    prepare_df = prepare_df.drop("Embarked", axis=1)


    ###ENCODE
    #encode gender from text to numerical values
    #female -> 0, male -> 1
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    sex_cat_train = encoder.fit_transform(prepare_df["Sex"])
    prepare_df["SexCode"] = sex_cat_train
    #print "First 10 instances with new feature SexCode:\n", prepare_df[:10]

    #drop the Sex feature with male/female values
    prepare_df = prepare_df.drop("Sex", axis=1)

    #Age
    prepare_df["AgeMinusOne"]  = initialDF["Age"].fillna(-1)
    maleAvgAge, femaleAvgAge = calculate_avg_age_sex(prepare_df)
    prepare_df = ageManipulation(prepare_df, maleAvgAge, femaleAvgAge)

    ###REPLACE NaN
    # Replace missing values with median
    # In this case it is only feature Age that has NaN values.
    from sklearn.preprocessing import Imputer
    imputer = Imputer(strategy="median")

    # replace the missing values by the learned medians from imputer
    # transformation results a plain Numpy array - conversion to Pandas DataFrame is needed
    # column names after this transformation are erased and numbers from 0 on are used. Thats the reason
    # why columns parameter is added - to keep the column names
    #imputer.fit(prepare_df)
    #prepare_df = pd.DataFrame(imputer.transform(x_trainDF), columns= prepare_df.columns)
    # print "First 10 instances with NaN for Age feature being 28:\n", prepare_df[:10]

    # Put age in a class by using anonymous function lambda
    prepare_df["AgeClass"] = prepare_df["Age"].map(lambda x: ageClass(x))
    #print x_trainDF[:20]

    #Family Size feature
    prepare_df["FamilySize"] = prepare_df["SibSp"] + prepare_df["Parch"]
    prepare_df = prepare_df.drop(["SibSp", "Parch"], axis=1)
    #x_trainDF["HasFamily"] = x_trainDF["HasFamily"].map(lambda x: 0 if x < 1 else 1)

    ##one-hot encoding for Sex. The code makes a feature out of each gender
    #using this one in SGD gives 0,675, using only Sex with 0/1 values gives 0,63 (at random_state=42)
    ###ONE-HOT ENCODE
    from sklearn.preprocessing import OneHotEncoder
    encoderOHE = OneHotEncoder()
    sex_cat_1hot_train = encoderOHE.fit_transform(sex_cat_train.reshape(-1, 1))
    sex_one_hot_train = np.array(sex_cat_1hot_train.toarray())

    #we add new columns to the DataFrame - Female and Male
    prepare_df["Female"] = sex_one_hot_train[:,0]
    prepare_df["Male"] = sex_one_hot_train[:,1]

    #x_trainDF["FamilySize"] = x_trainDF[""]

    #print "First 10 instances with each gender having own feature:\n", x_trainDF[:10]

    #Fare
    #prepare_df["Fare"] = prepare_df["Fare"].fillna(-1)
    med_fare = prepare_df["Fare"].median()
    prepare_df["Fare"] = prepare_df["Fare"].fillna(prepare_df["Fare"].median())
    #print med_fare

    return prepare_df
    #END prepare_data
