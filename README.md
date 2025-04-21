# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import pandas module and import the required data seT.
2.Find the null values and count them.
3.Count number of left values.
4.From sklearn import LabelEncoder to convert string values to numerical values.
5.From sklearn.model_selection import train_test_split.
6.Assign the train dataset and test dataset.
7.From sklearn.tree import DecisionTreeClassifier.
8.Use criteria as entropy.
9.From sklearn import metrics.
10.Find the accuracy of our model and predict the require values.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:   VARSHA SARATHY
RegisterNumber:  212223040233
*/

import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x=data[["satisfaction_level","last_evaluation","number_project", "average_montly_hours",
"time_spend_company", "Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn. tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt. predict(x_test)
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260, 6,0,1,2]])

```

## Output:

![image](https://github.com/user-attachments/assets/8a2bd529-6682-4e8b-a0b8-fc36fde003b0)

![image](https://github.com/user-attachments/assets/7a6e6c69-62e3-4221-83c0-16458316d2be)

![image](https://github.com/user-attachments/assets/a18787cb-5eb1-444d-8848-9a1c36cdf6c1)

![image](https://github.com/user-attachments/assets/47f27211-22a7-4876-80d9-3fcbca41a5b5)

![image](https://github.com/user-attachments/assets/5265fa18-64d2-4fa4-95c1-101ed572a44d)

![image](https://github.com/user-attachments/assets/5f237d33-2cb5-4f07-b5a8-f44c114f11b4)

![image](https://github.com/user-attachments/assets/ea711f09-392e-4607-9369-25f0bd08b62b)

![image](https://github.com/user-attachments/assets/a2f2d034-5e5b-4375-a207-7de47d28895e)


## Result:

Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
