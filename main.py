import  pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

#Loading data
data = pd.read_csv("housing.csv")
print(data)
print(data.info())

#Null_Values
data.dropna(inplace=True)
print(data.info())

#Spliting data
X = data.drop(['median_house_value'],axis=1)
y = data['median_house_value']
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.2)
train_data = X_train.join(y_train)
print(train_data)

#preprocessing
print(train_data.ocean_proximity.value_counts())
print(pd.get_dummies(train_data.ocean_proximity))
train_data =train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'],axis =1)
print(train_data)
train_data['total_rooms'] = np.log(train_data['total_rooms'] + 1)
train_data[ 'total_bedrooms'] = np.log(train_data['total_bedrooms'] + 1)
train_data['population'] = np.log(train_data['population'] + 1)
train_data['households'] = np.log(train_data['households'] + 1)
plt.figure(figsize=(15,8))
sns.heatmap(train_data.corr(),annot=True,cmap="YlGnBu")
#plt.show()


#adding parameters
train_data[ 'bedroom_ratio'] = train_data['total_bedrooms' ] / train_data[ "total_rooms"]
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']

from sklearn.linear_model import LinearRegression
x_train, y_train = train_data.drop([ 'median_house_value'], axis=1), train_data['median_house_value']
reg = LinearRegression()
reg.fit(x_train,y_train)

#linearregressionmodel
test_data = X_test.join(y_test)
test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1)
test_data['total_bedrooms'] = np.log(test_data['total_bedrooms' ] + 1)
test_data['population'] = np.log(test_data[ 'population'] + 1)
test_data['households'] = np.log(test_data[ 'households'] + 1)
test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)) .drop(['ocean_proximity'], axis=1)
test_data['bedroom_ratio'] = test_data["total_bedrooms" ] / test_data[ "total_rooms"]
test_data['household_rooms'] = test_data[ 'total_rooms'] / test_data[ 'households']
x_test, y_test= test_data.drop([ 'median_house_value'], axis=1), test_data['median_house_value']
z = reg.score(x_test,y_test)
print("Linear Regression Model Score :",z)




