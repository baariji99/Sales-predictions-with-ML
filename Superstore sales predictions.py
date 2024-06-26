import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn import metrics
# loading the data from csv file to Pandas DataFrame
super_mart_data = pd.read_csv('/Users/arijit/Downloads/sales_prediction.csv')

# first 5 rows of the dataframe
super_mart_data.head()
# number of data points & number of features
super_mart_data.shape
# getting some information about thye dataset
super_mart_data.info()
# checking for missing values
super_mart_data.isnull().sum()
# mean value of "Item_Weight" column
super_mart_data['Item_Weight'].mean()
# filling the missing values in "Item_weight column" with "Mean" value
super_mart_data['Item_Weight'].fillna(super_mart_data['Item_Weight'].mean(), inplace=True)
# mode of "Outlet_Size" column
super_mart_data['Outlet_Size'].mode()
# filling the missing values in "Outlet_Size" column with Mode
mode_of_Outlet_size = super_mart_data.pivot_table(values='Outlet_Size', columns='Outlet_Type', aggfunc=(lambda x: x.mode()[0]))
print(mode_of_Outlet_size)
# checking for missing values
super_mart_data.isnull().sum()
#Data Analysis
super_mart_data.describe()
sns.set()
# Item_Weight distribution
plt.figure(figsize=(6,6))
sns.distplot(super_mart_data['Item_Weight'])
plt.show()
# Item Visibility distribution
plt.figure(figsize=(6,6))
sns.distplot(super_mart_data['Item_Visibility'])
plt.show()
# Item MRP distribution
plt.figure(figsize=(6,6))
sns.distplot(super_mart_data['Item_MRP'])
plt.show()
# Item_Outlet_Sales distribution
plt.figure(figsize=(6,6))
sns.distplot(super_mart_data['Item_Outlet_Sales'])
plt.show()
# Outlet_Establishment_Year column
plt.figure(figsize=(6,6))
sns.countplot(x='Outlet_Establishment_Year', data=super_mart_data)
plt.show()
# Item_Fat_Content column
plt.figure(figsize=(6,6))
sns.countplot(x='Item_Fat_Content', data=super_mart_data)
plt.show()
# Item_Type column
plt.figure(figsize=(30,6))
sns.countplot(x='Item_Type', data=super_mart_data)
plt.show()
super_mart_data['Item_Fat_Content'].value_counts()
super_mart_data.replace({'Item_Fat_Content': {'low fat':'Low Fat','LF':'Low Fat', 'reg':'Regular'}}, inplace=True)
encoder = LabelEncoder()
super_mart_data['Item_Identifier'] = encoder.fit_transform(super_mart_data['Item_Identifier'])

super_mart_data['Item_Fat_Content'] = encoder.fit_transform(super_mart_data['Item_Fat_Content'])

super_mart_data['Item_Type'] = encoder.fit_transform(super_mart_data['Item_Type'])

super_mart_data['Outlet_Identifier'] = encoder.fit_transform(super_mart_data['Outlet_Identifier'])
super_mart_data['Outlet_Location_Type'] = encoder.fit_transform(super_mart_data['Outlet_Location_Type'])
super_mart_data['Outlet_Type'] = encoder.fit_transform(super_mart_data['Outlet_Type'])


super_mart_data['Outlet_Size'] = encoder.fit_transform(super_mart_data['Outlet_Size'])
X = super_mart_data.drop(columns='Item_Outlet_Sales', axis=1)
Y = super_mart_data['Item_Outlet_Sales']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
print(X.shape, X_train.shape, X_test.shape)
regressor = XGBRegressor()
regressor.fit(X_train, Y_train)
# prediction on training data
training_data_prediction = regressor.predict(X_train)
# R squared Value
r2_train = metrics.r2_score(Y_train, training_data_prediction)
# prediction on test data
test_data_prediction = regressor.predict(X_test)
# R squared Value
r2_test = metrics.r2_score(Y_test, test_data_prediction)
print('R Squared value = ', r2_test)


