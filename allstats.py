import matplotlib.pyplot as plt
from csv_reader import csv_reader
import sklearn.preprocessing as skl_pre
import pandas as pd
from dataprep import meannorm


df = csv_reader("data/train.csv")
label = df.get('Lead')
df = df.drop('Lead', axis=1)


scaler = skl_pre.MinMaxScaler()
x_minmax = scaler.fit_transform(df)
df = pd.DataFrame(x_minmax, columns=df.columns)


#df=df.apply(meannorm, axis=0)
df['Lead'] = label

class_means = pd.DataFrame(index=['Male', 'Female'])
#Adds the mean of a feature for the male/female lead class to a new df under same column name
for column in df:
    if column == 'Lead':
        pass
    else:
        class_means.at['Male', column] = df.loc[df['Lead'] == 'Male', column].mean()      
        class_means.at['Female', column] = df.loc[df['Lead'] == 'Female', column].mean()  

perc_diff = abs(class_means.loc['Male', :]/class_means.loc['Female', :] -1) #Now a panda series

perc_diff.index = perc_diff.index.str.wrap(10)  # Wraps the index (keys) of the Panda Series (to be readable on bar chart later)

class_means.columns = class_means.columns.str.wrap(10)

perc_diff.plot.bar()
class_means.T.plot.bar()
plt.show()
