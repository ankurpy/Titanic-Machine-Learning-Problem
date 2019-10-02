#!/usr/bin/env python
# coding: utf-8

# In[133]:


import numpy as np
import pandas as pd


#LOADING THE DATSETS

tr = pd.read_csv("train.csv")
ts = pd.read_csv("test.csv")

display(tr.head())
display(ts.head())


# In[124]:


#PREPROCESSING THE DATASETS-DROPPING UNWANTED FEATURES 

cols = ['PassengerId','Name','Ticket','Cabin']
tr = tr.drop(cols, axis=1)

cols1 = ['PassengerId','Name','Ticket','Cabin']
ts = ts.drop(cols, axis=1)

#display(tr.head(5))
#display(ts.head(5))

#display(tr.shape)
#display(ts.shape)


ts = ts.dropna()

display(tr.info())
display(ts.info())


# In[125]:


#SPLITTING COLUMNS OF P_CLASS, SEX, EMBARKED IN TRAINING DATASET 

dummies = []
cols2 = ['Pclass','Sex','Embarked']
for i in cols2:
    dummies.append(pd.get_dummies(tr[i]))

titanic_dummies = pd.concat(dummies, axis=1)
tr = pd.concat((tr,titanic_dummies),axis=1)

tr = tr.drop(['Pclass','Sex','Embarked'],axis=1)
display(tr.head(5))



# In[126]:


#SPLITTING COLUMNS OF P_CLASS, SEX, EMBARKED IN TEST DATASET 

dummies = []
cols3 = ['Pclass','Sex','Embarked']
for i in cols3:
    dummies.append(pd.get_dummies(ts[i]))

titanic_dummies1 = pd.concat(dummies, axis=1)
ts = pd.concat((ts,titanic_dummies1),axis=1)

ts = ts.drop(['Pclass','Sex','Embarked'],axis=1)
display(ts.head(5))


# In[127]:


#FILLED THE MISSING DATA VALUES USING INTERPOLATE FUNCTION

tr['Age'] = tr['Age'].interpolate()


display(tr.info())
display(ts.info())


# In[128]:


#SPLITTING INPUT VALUES AND OUTPUT 

x = tr.values 
y = tr['Survived'].values
x = np.delete(x, 0, axis=1)

x1 = ts.values
#print(y)

print(x)
print(x1)


# In[129]:


#MODEL TRAINING AND PREDICTING

from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors=1)

#TRAINING THE MODEL USING DATASET
model.fit(x, y)

#PREDICTING OUTPUT
predicted= model.predict(x1) 
print(predicted)

