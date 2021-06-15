try:
    from IPython import get_ipython
    get_ipython().magic('clear')
    get_ipython().magic('reset -f')
except:
    pass

#To ignore the warnings ####################Setting working directory############################
import os
os.chdir("E:\Model Deployment\prediction_Expense")

#Importing all the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
import sklearn


import warnings as wg
wg.filterwarnings("ignore")

#Read the dataset file of train data
train=pd.read_csv("train.csv")
#Read the dataset file of test data
test=pd.read_csv("test.csv")

train.head()

test.head()

train.tail()

test.tail()

train.shape

test.shape

train.describe()

test.describe()

#Now we will check if our dataset contains null or missings values
train.isnull().sum()

#Now we will check if our dataset contains null or missings values
test.isnull().sum()

#we can also use .corr to determine the correlation between the variables
train.corr()

#we can also use .corr to determine the correlation between the variables
test.corr()

train.dtypes

test.dtypes

def print_unique_col_values(train):
       for column in train:
            if train[column].dtypes=='object':
                print(f'{column}: {train[column].unique()}')

print_unique_col_values(train)

def print_unique_col_values(test):
       for column in test:
            if test[column].dtypes=='object':
                print(f'{column}: {test[column].unique()}')

print_unique_col_values(test)

yes_no_columns = ['Aged','Married','TotalDependents','MobileService','CyberProtection','HardwareSupport','TechnicalAssistance',
                'FilmSubscription','CustomerAttrition' ]
for col in yes_no_columns:
    train[col].replace({'Yes': 1,'No': 0},inplace=True)

train['sex'].replace({'Female':1,'Male':0},inplace=True)
train.sex.unique()

train.dtypes

train1 = pd.get_dummies(data=train, columns=['4GService','SettlementProcess'])
train1.columns

train1.dtypes

yes_no_columns = ['Aged','Married','TotalDependents','MobileService','CyberProtection','HardwareSupport','TechnicalAssistance',
                'FilmSubscription']
for col in yes_no_columns:
    test[col].replace({'Yes': 1,'No': 0},inplace=True)

test['sex'].replace({'Female':1,'Male':0},inplace=True)
test.sex.unique()

test.dtypes

test1 = pd.get_dummies(data=test, columns=['4GService','SettlementProcess'])
test1.columns

test1.dtypes

mean_value=train1['GrandPayment'].mean()
train1['GrandPayment'].fillna(value=mean_value,inplace=True)

train1.isnull().sum()

train1.head()

train1.drop('ID',inplace=True,axis=1)

mean_value=test1['GrandPayment'].mean()
test1['GrandPayment'].fillna(value=mean_value,inplace=True)

test1.isnull().sum()

test1.head()

#Perform Feature Scaling and One Hot Encoding on the train dataset
from sklearn.preprocessing import StandardScaler

#Perform Feature Scaling on 'QuarterPayment', 'ServiceSpan', 'GrandPayment' in order to bring them on same scale
standardScaler = StandardScaler()
columns_for_ft_scaling = ['QuarterlyPayment', 'ServiceSpan', 'GrandPayment']

#Apply the feature scaling operation on dataset using fit_transform() method
train1[columns_for_ft_scaling] = standardScaler.fit_transform(train1[columns_for_ft_scaling])

train1.head()

#Perform Feature Scaling and One Hot Encoding on the test dataset
from sklearn.preprocessing import StandardScaler

#Perform Feature Scaling on 'QuarterPayment', 'ServiceSpan', 'GrandPayment' in order to bring them on same scale
standardScaler = StandardScaler()
columns_for_ft_scaling = ['QuarterlyPayment', 'ServiceSpan', 'GrandPayment']

#Apply the feature scaling operation on dataset using fit_transform() method
test1[columns_for_ft_scaling] = standardScaler.fit_transform(test1[columns_for_ft_scaling])

test1.head()

test1.drop('ID',inplace=True,axis=1)

#check for any missing or null values in the train data
train1.isnull().values.any()

#check for any missing or null values in the test data
test1.isnull().values.any()

#Number of count of the customer stay with company or leave the company
train1['CustomerAttrition'].value_counts()

#Visualize this count 
sns.countplot(train1['CustomerAttrition'])

#Show the number of employees that left and stayed by age
import matplotlib.pyplot as plt
fig_dims = (12, 4)
fig, ax = plt.subplots(figsize=fig_dims)

#ax = axis
sns.countplot(x='Aged', hue='CustomerAttrition', data = train1, palette="colorblind", ax = ax,  edgecolor=sns.color_palette("dark", n_colors = 1));

#Visualize the correlation
plt.figure(figsize=(14,14))  #14in by 14in
sns.heatmap(train1.corr(), annot=True, fmt='.0%')

#Split the data into independent 'X' and dependent 'Y' variables
X = train1.iloc[:, 1:train1.shape[1]].values 
Y = train1.iloc[:, 0].values

# Split the dataset into 75% Training set and 25% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

Y_train=train1['CustomerAttrition'].copy()

X_train=train1.drop('CustomerAttrition', axis='columns' , inplace=True)

#Split the data into independent 'X' and dependent 'Y' variables
X = test1.iloc[:, 1:test1.shape[1]].values 
Y = test1.iloc[:, 0].values

# Split the dataset into 75% Training set and 25% Testing set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25, random_state = 0)

#Use Random Forest Classification algorithm
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rf.fit(X_train ,Y_train)

rf.score(X_train, Y_train)

final_pred = rf.predict(X_test)

y_pred = final_pred

pred=pd.DataFrame(y_pred)
sub_df=pd.read_csv(r'C:\Users\Pooja\Downloads\Sample Submission.csv')

submission = pd.concat([sub_df['ID'],pred],axis=1)
submission.columns=['ID',"CustomerAttrition"]

#submission.to_csv('SAMPLESUB5.csv')

# pickling the Model
import pickle
file = open('Predict.pkl', 'wb')
pickle.dump(rf, file)