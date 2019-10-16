import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as seabornInstance 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder 
from sklearn import metrics
import csv

#read in data from csv files
df = pd.read_csv('tcd ml 2019-20 income prediction training (with labels).csv')
df_test = pd.read_csv('tcd ml 2019-20 income prediction test (without labels).csv')
df_test = df_test.drop('Income', axis=1) # Remove income as headings are different

total = df.append(df_test)

index = len(total) - len(df_test)

#Replace empty values and remove unused columns
df = total.replace(np.NaN, 0)
X = df.drop(['Instance', 'Profession', 'Hair Color'], axis=1)

#One hot encoding to handle categorical data
Xohe = X.select_dtypes(include=['object'])
Xohe = pd.get_dummies(Xohe, columns=['University Degree', 'Country', 'Gender'])
X = X.drop(['University Degree', 'Country', 'Gender'], axis=1)

#concatenate numerical and categorical data
data = pd.concat([X,Xohe], axis=1)

training_data = data[:index]
test_data = data[index:]
trainy = training_data['Income in EUR'].astype('int')
trainx = training_data.drop(['Income in EUR'], axis=1)
test = test_data.drop(['Income in EUR'], axis=1)

#train model
X_train, X_test, y_train, y_test = train_test_split(trainx, trainy, test_size=0.3, random_state=0)

regressor = LinearRegression()  
regressor.fit(X_train, y_train) #training the algorithm
y_pred = regressor.predict(X_test)

y_testpred = regressor.predict(test)
print(len(y_testpred))

subf = pd.read_csv('tcd ml 2019-20 income prediction submission file.csv')
i = 0
while (i < len(y_testpred)):
    subf.Income[i] = y_testpred[i]
    i += 1
subf.to_csv('out3.csv', index=False, sep=',')

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))