
from csv_reader import csv_reader
from sklearn.model_selection import train_test_split
import pandas as pd
import sklearn.preprocessing as skl_pre

def maxnorm(s):
    return s/s.max()

def meannorm(s):
    return s/s.mean()

def get_testset():
    df = csv_reader("data/test.csv")
    df = feature_prep(df)
    minmax = skl_pre.MinMaxScaler()
    x_minmax = minmax.fit_transform(df)
    df = pd.DataFrame(x_minmax, columns=df.columns)
    df=df.apply(meannorm, axis=0)
    return df



def getdata_minmax():
    
    df = csv_reader("data/train.csv")
    label = df.get('Lead')
    df = df.drop('Lead', axis=1)
    df = feature_prep(df)
    
    minmax = skl_pre.MinMaxScaler()
    x_minmax = minmax.fit_transform(df)
    df = pd.DataFrame(x_minmax, columns=df.columns)

    
    x_train, x_test, y_train, y_test = train_test_split(df, label,train_size=0.75, random_state = 0)
    classes = ["Male","Female"]
    y_train, y_test = y_train.apply(classes.index), y_test.apply(classes.index)
    
    return x_train, x_test, y_train, y_test

def feature_prep(df):
    
    #Add some new features by combining others

    df["Perc NMA"] = df["Number of male actors"]/(df["Number of male actors"] + df["Number of female actors"])
    df["Perc NFA"] = df["Number of female actors"]/(df["Number of male actors"] + df["Number of female actors"])
    
    df["AL/ACL"] = df["Age Lead"]/df["Age Co-Lead"]
    df["AL/AF"] = (df["Age Lead"]/df["Mean Age Female"])
    df["ACL/AM"] = (df["Age Co-Lead"]/df["Mean Age Male"])
    df["ACL/AF"] = df["Age Co-Lead"]/df["Mean Age Female"]
    df["AM/AF"] = df["Mean Age Male"]/df["Mean Age Female"]
    

    df["Words per actor"] = df["Total words"]/(df["Number of male actors"] + df["Number of female actors"])
    df["Words per male"] = df["Number words male"]/df["Number of male actors"]
    df["Words per female"] = df["Number words female"]/df["Number of female actors"]
    df["Number of words co-lead"] = df["Number of words lead"] - df["Difference in words lead and co-lead"]
    
    df["Perc WF"] = (df["Number words female"]/df["Total words"])
    df["Perc WM"] = (df["Number words male"]/df["Total words"])
    df["Perc WCL"] = (df["Number of words co-lead"]/df["Total words"])
    
    df["WpM/WCL"] = df["Words per male"]/df["Number of words co-lead"]
    df["WpF/WCL"] = df["Words per female"]/df["Number of words co-lead"]
    
    df["WpM/WpA"] = df["Words per male"]/df["Words per actor"]
    df["WpF/WpA"] = df["Words per female"]/df["Words per actor"]

#Remove low performing features
    df = df.drop(['Number of words lead', 'Difference in words lead and co-lead', 'Mean Age Female', 'Mean Age Male', 
                  'Year', 'Gross', 'Age Co-Lead', 'Age Lead', 'Number of male actors', 'Number words male', 'Number words female', 
                  'Number of female actors','Number of words co-lead','Total words', 'Words per male', 'Words per actor', 'Words per female'], axis=1)
    return df

def getdata_minmax_mean():
    df = csv_reader("data/train.csv")
    label = df.get('Lead')
    df = df.drop('Lead', axis=1)

    df = feature_prep(df)
    
    minmax = skl_pre.MinMaxScaler()
    x_minmax = minmax.fit_transform(df)
    df = pd.DataFrame(x_minmax, columns=df.columns)
    df=df.apply(meannorm, axis=0)

    x_train, x_test, y_train, y_test = train_test_split(df, label,train_size=0.75, random_state = 0)
    classes = ["Male","Female"]
    # Converts "Male" and "female" calss labels to 0 and 1.
    y_train, y_test = y_train.apply(classes.index), y_test.apply(classes.index) 
   
    return x_train, x_test, y_train, y_test

def getdata_max():
    
    df = csv_reader("data/train.csv")
    label = df.get('Lead')
    df = df.drop('Lead', axis=1)
    df = feature_prep(df)
    
    df=df.apply(maxnorm, axis=0)
 
    x_train, x_test, y_train, y_test = train_test_split(df, label,train_size=0.75, random_state = 0)
    classes = ["Male","Female"]
    y_train, y_test = y_train.apply(classes.index), y_test.apply(classes.index)
    
    return x_train, x_test, y_train, y_test

def getdata_mean():
    df = csv_reader("data/train.csv")
    label = df.get('Lead')
    df = df.drop('Lead', axis=1)
    df = feature_prep(df)

    df=df.apply(meannorm, axis=0)
    x_train, x_test, y_train, y_test = train_test_split(df, label,train_size=0.75, random_state = 0)
    classes = ["Male","Female"]
    y_train, y_test = y_train.apply(classes.index), y_test.apply(classes.index)
    x_val = csv_reader('data/test.csv')
    
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":
    print(getdata_minmax_mean)
   
    