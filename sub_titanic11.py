#!/usr/bin/env python
# coding: utf-8

# In[244]:


import numpy as np
import pandas as pd


# In[245]:


data = pd.read_csv("train.csv")
display(data.head())
display(data.info())


# In[246]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[247]:


sur = data[data['Survived']==1].size
de = data[data['Survived']==0].size
ps=pd.DataFrame([sur,de])
ps.plot(kind='pie', subplots=True,labels=['survived','dead'], startangle=180, explode=[0,0.04], autopct='%1.1f%%',colors=['g','r'])
plt.legend()
plt.show()


# In[248]:


a31=pd.Series(data['Age'])
for j, i in enumerate(a31.values):
    if i<10:
        a31[j]='child'
    elif i>=10 and i<20:
        a31[j]='teenager'
    elif i>=20 and i<50:
        a31[j]='adult'
    else:
        a31[j]='senior citizen'
print(a31.value_counts())


# In[249]:


def bar_chart(feature):
    survived = data[data['Survived']==1][feature].value_counts()
    dead = data[data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', figsize=(10,5))


# In[250]:


bar_chart('Sex')
print("Survived :\n",data[data['Survived']==1]['Sex'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Sex'].value_counts())


# In[251]:


bar_chart('Pclass')
print("Survived :\n",data[data['Survived']==1]['Pclass'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Pclass'].value_counts())


# In[252]:


bar_chart('SibSp')
print("Survived :\n",data[data['Survived']==1]['SibSp'].value_counts())
print("Dead:\n",data[data['Survived']==0]['SibSp'].value_counts())


# In[253]:


bar_chart('Embarked')
print("Survived :\n",data[data['Survived']==1]['Embarked'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Embarked'].value_counts())


# In[254]:


bar_chart('Parch')
print("Survived :\n",data[data['Survived']==1]['Parch'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Parch'].value_counts())


# In[255]:


figure = plt.figure(figsize=(25, 7))
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], 
         stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();


# In[256]:


tr = pd.read_csv("train.csv")
ts = pd.read_csv("test.csv")


# In[257]:


#print(tr.head())
#print(ts.head())
#print(tr.isnull().sum())
#print(ts.isnull().sum())


# In[258]:


cols = ['PassengerId','Name','Ticket','Cabin']
tr = tr.drop(cols, axis=1)

cols1 = ['PassengerId','Name','Ticket','Cabin']
ts = ts.drop(cols, axis=1)


# In[259]:


#tr.info()
#ts.info()


# In[260]:


#SPLITTING COLUMNS OF P_CLASS, SEX, EMBARKED IN TRAINING DATASET 

dummies = []
cols2 = ['Pclass','Sex','Embarked']
for i in cols2:
    dummies.append(pd.get_dummies(tr[i]))

titanic_dummies = pd.concat(dummies, axis=1)
tr = pd.concat((tr,titanic_dummies),axis=1)

tr = tr.drop(['Pclass','Sex','Embarked'],axis=1)
display(tr.head(5))


# In[261]:


#SPLITTING COLUMNS OF P_CLASS, SEX, EMBARKED IN TEST DATASET 

dummies = []
cols3 = ['Pclass','Sex','Embarked']
for i in cols3:
    dummies.append(pd.get_dummies(ts[i]))

titanic_dummies1 = pd.concat(dummies, axis=1)
ts = pd.concat((ts,titanic_dummies1),axis=1)

ts = ts.drop(['Pclass','Sex','Embarked'],axis=1)
display(ts.head(5))


# In[262]:


#FILLED THE MISSING DATA VALUES USING INTERPOLATE FUNCTION

tr['Age'] = tr['Age'].interpolate()
ts['Age'] = ts['Age'].interpolate()
ts['Fare'] = ts['Fare'].interpolate()


# In[263]:


#SPLITTING INPUT VALUES AND OUTPUT 

x = tr.values 
y = tr['Survived'].values
x = np.delete(x, 0, axis=1)

x1 = ts.values
#print(y)
#print(x)
#print(x1)


# In[264]:


print(tr.shape)
print(y.shape)
print(ts.shape)


# In[265]:


from sklearn import preprocessing
x = preprocessing.normalize(x)
x1 = preprocessing.normalize(x1)


# In[266]:


ac = pd.read_csv("gender_submission.csv")
pid = ['PassengerId']
ac = ac.drop(pid, axis=1)
print(ac.shape)


# In[267]:


#MODEL TRAINING AND PREDICTING

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
k = range(1,11)
l=[]
for i in k:
    

    model = KNeighborsClassifier(n_neighbors=i)

    #TRAINING THE MODEL USING DATASET
    model.fit(x, y)

    #PREDICTING OUTPUT
    predicted= model.predict(x1) 
    #print(predicted)
    
    
    print("Accuracy for k = ",i,"is",metrics.accuracy_score(ac, predicted))
    print(confusion_matrix(ac, predicted))
    #print(classification_report(ac, predicted))
    print('\n')
    l.insert(i,metrics.accuracy_score(ac, predicted))


# In[268]:


plt.plot(k,l)
plt.xlabel('k neighbours')
plt.ylabel('accuracy')
plt.plot()


# In[271]:


'''
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

svc = SVC()
svc.fit(x, y)
y_pred = svc.predict(x1)
acc_svc = round(accuracy_score(ac, y_pred) * 100, 2)
print(acc_svc)
'''


# In[272]:


'''
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x, y)
y_pred = gaussian.predict(x1)
acc_gaussian = round(accuracy_score(y_pred, ac) * 100, 2)
print(acc_gaussian)
'''


# In[ ]:




