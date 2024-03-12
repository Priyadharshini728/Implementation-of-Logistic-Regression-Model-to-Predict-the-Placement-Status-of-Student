# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.

## Program:



```python

/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: HIRUTHIK SUDHAKAR
RegisterNumber: 212223240054
*/

import pandas as pd
data=pd.read_csv("C:/Users/admin/Downloads/Midhun/Placement_Data.csv")
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:



### TOP 5 ELEMENTS
![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/9f6fe04b-767a-4fe1-a08a-d10ecf965f12)



![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/958449ad-7710-4cdf-a5ca-1e7513d5e765)

### DATA DUPLICATE


![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/dedc174a-504b-407e-9c78-f0dc56a4c25b)


### PRINT DATA

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/95283ec5-e5ff-4783-9368-50d85ce85052)

### DATA_STATUS

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/e222167f-0780-4199-9588-9a660bf0ae7c)


### Y_PREDICTION ARRAY

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/2a10248e-b708-43d7-84e1-278c34cfcdeb)

### CONFUSION ARRAY
![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/a5cdf063-69ec-424d-b683-0e3f4641a8bd)

### ACCURACY VALUE

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/22d97e80-aa70-4bb8-be92-87e92a88d7cc)

### CLASSFICATION REPORT


![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/8ecc1721-e62d-435c-9df1-97670e572e0d)
### PREDICTION
![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/394cb915-b7e1-4023-8151-6d4a13d8b71f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
