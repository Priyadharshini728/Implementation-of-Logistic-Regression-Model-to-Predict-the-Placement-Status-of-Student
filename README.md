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

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/04b0fcce-c444-42d7-b055-a7e6ac77678d)

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/e9963d89-9a92-4575-b01e-677a85537cf1)

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/e21f9740-4b51-489a-a471-f59892e74f69)



### DATA DUPLICATE

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/210eead6-4770-4e23-b794-b1a5f9428e0d)


### PRINT DATA

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/2ad5eecf-18b0-47ba-84e0-9fc00443541e)


### DATA_STATUS

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/3a6e8cae-f77d-414e-b674-020cd6f87fae)


### Y_PREDICTION ARRAY

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/ff5c9ca0-aa45-4b9f-a727-8ff5bb079ba9)


### CONFUSION ARRAY
![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/1e6c4ab6-c90b-4278-be39-1f7721487b21)

### ACCURACY VALUE

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/e70452c6-a4ee-4428-be43-bfd1d677f5fa)

### CLASSFICATION REPORT

![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/70144eaf-0396-4780-ac47-a34e16f85ad8)

### PREDICTION
![image](https://github.com/HIRU-VIRU/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/145972122/791654fd-4387-4d63-a415-cbaa1564a6a1)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
