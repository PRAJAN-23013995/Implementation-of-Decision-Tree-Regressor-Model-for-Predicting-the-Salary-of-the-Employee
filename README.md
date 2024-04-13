# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard libraries.

2.Upload the dataset and check for any null values using .isnull() function.

3.Import LabelEncoder and encode the dataset.

4.Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5.Predict the values of arrays.

6.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7.Pedict the values of array.

8.Apply to new unknown values. 

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: PRAJAN P
RegisterNumber: 212223240121
*/

import pandas as pd
data=pd.read_csv("Salary.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)


from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])



```

## Output:

![318635862-77ea74b5-eaa6-40ab-becc-66d0ef92ffe3](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/7c004e33-9234-49da-a096-4e369a05df3c)


![318635934-0807da6e-e3b2-49de-a80d-a2023012cee3](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/5c5645e6-d814-4f0b-aa28-c0c2479f7b7d)


![318636024-12d97fa7-beba-4475-b9bc-ef82b774231f](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/38714279-17aa-4abe-a762-74d66f047252)


![318636172-281974a1-6486-4472-9298-638f993f3f97](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/ab7af8cf-0006-48bc-ac68-017abfa92525)


![318636284-739c68f1-c0b0-46ac-aaf0-4c15851f3452](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/2cb7ad3a-817d-450e-90e7-628f99c117d1)

![318636477-60389f88-f036-4269-ac7e-9626f8bdcb74](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/d947b775-a9ab-41e4-8be7-b439aeb3c1f8)

![318636578-fa80681b-aa21-42cc-9ca3-bf25aba48a69](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/78ac948c-dc27-4a82-81f1-0f777e7871aa)

![318636693-a1d6a34a-c8c1-47b6-b8ff-2942692cdf3b](https://github.com/PRAJAN-23013995/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/150313345/6d45c9fa-fb44-44da-b92d-bc80724bfdd9)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
