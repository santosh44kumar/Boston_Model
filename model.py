import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston

import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

boston = load_boston()


bostData = pd.DataFrame(boston.data, columns=boston.feature_names)
bostData.head()

# Adding the target variable to the dataframe

bostData["MEDV"] = boston.target
bostData.head()

bostData.describe()

#Feature selection using Wrapper method

x_FCheck = bostData.drop("MEDV",axis=1)
y_FCheck = bostData[["MEDV"]]

x = bostData[["CRIM","ZN","CHAS","NOX","RM","DIS","PTRATIO","B","LSTAT"]]
y = bostData[["MEDV"]]

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.33,random_state=42)

# define the model

LinearRegModel = LinearRegression()

# fit the model
LinearRegModel.fit(x_train,y_train)

# Saving model to disk
import pickle
pickle.dump(LinearRegModel, open('model.pkl','wb'))


# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[0.00632,18,0,0.538,6.575,4.0900,15.3,396.90,4.98]]))

#print(model.coef_)

#y_predict = model.predict(x_test)

#print(model.score(x_test,y_test))

# Accuracy metrix

#R2 & MSE

#from sklearn.metrics import mean_squared_error,r2_score
#print("R2 value:" , r2_score(y_test,y_predict))
#print("MSE vlaue:" , mean_squared_error(y_test,y_predict))