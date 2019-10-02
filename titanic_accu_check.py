#!/usr/bin/env python
# coding: utf-8

# In[130]:


import numpy as np
import pandas as pd

#LOADING THE DATASET

tr = pd.read_csv("train.csv")
#display(tr.head(5))


# In[131]:


#PREPROCESSING THE DATASETS-DROPPING UNWANTED FEATURES 

cols = ['PassengerId','Name','Ticket','Cabin']
tr = tr.drop(cols, axis=1)

display(tr.info())


# In[132]:


#SPLITTING COLUMNS OF P_CLASS, SEX, EMBARKED IN TRAINING DATASET 


dummies = []
cols2 = ['Pclass','Sex','Embarked']
for i in cols2:
    dummies.append(pd.get_dummies(tr[i]))

titanic_dummies = pd.concat(dummies, axis=1)
tr = pd.concat((tr,titanic_dummies),axis=1)

tr = tr.drop(['Pclass','Sex','Embarked'],axis=1)
display(tr.head(5))


# In[133]:


#FILLED THE MISSING DATA VALUES USING INTERPOLATE FUNCTION

tr['Age'] = tr['Age'].interpolate()


display(tr.info())


# In[134]:


#SPLITTING INPUT VALUES AND OUTPUT 

x = tr.values
y = tr['Survived'].values
x = np.delete(x, 0, axis=1)

#print(x)


# In[135]:


#SPLITTING THE DATASET INTO TRAINING AND TESTING
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3) # 70% training and 30% test


# In[136]:


#MODEL TRAINING AND PREDICTING

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=4)

#TRAINING THE MODEL USING DATASET
model.fit(X_train, y_train)

#PREDICTING OUTPUT
predicted= model.predict(X_test) 
print(predicted)


# In[137]:


#ACCURACY CALCULATION

from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, predicted))


# In[138]:


#EVALUATING THE MODEL PERFORMANCE

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, predicted))
print(classification_report(y_test, predicted))

