import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class feature_engineering():
    # method takes an age and assigns it to a class
    def age_class(self, age):
        if (age <= 16):
            age_class = 0
        elif (age <= 30):
            age_class = 1
        elif (age <= 48):
            age_class = 2
        elif (age <= 99):
            age_class = 3
        else:
            age_class = 4
        return age_class

    # title is classified by social class - advanced version
    def title_social_class_classified_advanced(self, name_string):
        def parse_title(name_str): # parse title from name string
            return name_str[name_str.index(",") + 2:str(name_str).index(".")]

        title = parse_title(name_str=name_string)
        if title in ['Master']: # young boys
            title_class = 0
        elif title in ['Mrs', 'Miss', 'Mme', 'Ms', 'Mlle']: # young females - prime of life
            title_class = 1
        elif title in ['Lady', 'the Countess']: # older females - high status symbol
            title_class = 2
        elif title in ['Mr']: # younger men - from biological perspective: worth least
            title_class = 3
        elif title in ['Rev', 'Don', 'Dr', 'Major', 'Sir', 'Col', 'Capt', 'Jonkheer']: # wouldnt board out of honor, too old to survive
            title_class = 4
        else:
            title_class = 5
        return title_class

    def get_title_from_name(self, df):
        df["Title"] = df["Name"].map(lambda x: self.title_social_class_classified_advanced(name_string=x))
        df = df.drop(columns="Name", axis=1)
        return df

    def embarked_class(self, df):
        def classify_embark_point(embarked):
            if embarked == 'C':
                return 0
            elif embarked == 'Q':
                return 1
            elif embarked == 'S':
                return 2
            else:
                return 2  # if embarked is unknown, use class of 'S' since most passengers embarked there

        df["EmbarkedClass"] = df["Embarked"].map(lambda x: classify_embark_point(embarked=x))
        df = df.drop(columns="Embarked", axis=1)
        return df

    # calculate average age for each sex class
    def calculate_avg_age_sex(self, df):
        male_avg_age = round(df.where(df["SexCode"] == 1)["Age"].mean(), 2)
        female_avg_age = round(df.where(df["SexCode"] == 0)["Age"].mean(), 2)
        return male_avg_age, female_avg_age

    # add average age based on gender for missing values
    def age_manipulation(self, df):
        male_avg_age, female_avg_age = self.calculate_avg_age_sex(df) # calculate average age for each gender
        df["Age_temp"] = np.where((df["Age"].isnull() & (df["SexCode"] == 1)), male_avg_age, df["Age"])
        df["Age_temp"] = np.where((df["Age_temp"].isnull() & (df["SexCode"] == 0)), female_avg_age, df["Age_temp"])
        df = df.drop("Age", axis=1)
        df.rename(columns={'Age_temp': 'Age'}, inplace=True)
        return df

    # encode sex from text to numerical values -- female -> 0, male -> 1
    def encode_sex(self, df):
        encoder = LabelEncoder()
        sex_class = encoder.fit_transform(df["Sex"])
        df["SexCode"] = sex_class
        df = df.drop("Sex", axis=1) # drop the Sex feature with male/female values
        return df

    # family size - features SibSp and Parch are added together
    def family_size(self, df):
        df["FamilySize"] = df["SibSp"] + df["Parch"]
        df = df.drop(columns=["SibSp", "Parch"], axis=1)
        return df

    # load data in dataframe
    def load_data(self, file_name):
        initial_df = pd.read_csv(file_name)
        return initial_df

    # prepare dataframe with inputs, features, dimensions
    def prepare_data(self, df):
        # Pclass - ticket class - 1 = 1st, 2 = 2nd, 3 = 3rd
        prepare_df = df[["PassengerId", "Survived", "Pclass", "Age", "Sex", "Name", "Embarked", "SibSp", "Parch", "Fare"]]
        prepare_df.is_copy = False

        prepare_df = self.get_title_from_name(df=prepare_df) # parse title from name feature
        prepare_df = self.embarked_class(df=prepare_df) # embarked class
        prepare_df = self.encode_sex(df=prepare_df) # gender encoding
        prepare_df = self.age_manipulation(df=prepare_df) # filling missing age values
        prepare_df = self.family_size(df=prepare_df) #family size
        return prepare_df

    def save_to_csv(self, df, save_file):
        df.to_csv(path_or_buf=save_file, index=False)

path = "data/" # path to the folder where input file is and where the output file will be written
open_file = "train.csv"
save_file = path + "train_prepared.csv"

fe = feature_engineering()
df = fe.load_data(file_name=path + open_file)
prepared_df = fe.prepare_data(df)
fe.save_to_csv(df=prepared_df, save_file= save_file)