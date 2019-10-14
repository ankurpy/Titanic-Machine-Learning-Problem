#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


tr = pd.read_csv("train.csv")
ts = pd.read_csv("test.csv")
display(tr.head())
display(ts.head())


# In[3]:


print(tr.info())
print(ts.info())
#display(tr.isnull().sum())
#display(ts.isnull().sum())


# In[7]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set()


# In[8]:


sur=tr[tr['Survived']==1].size
de=tr[tr['Survived']==0].size
ps=pd.DataFrame([sur,de])
ps.plot(kind='pie', subplots=True,labels=['survived','dead'], startangle=180, explode=[0,0.04], autopct='%1.1f%%',colors=['g','r'])
plt.legend()
plt.show()


# In[9]:


tr['Age'].hist(bins=50)


# In[10]:


def bar_chart(feature):
    survived = tr[tr['Survived']==1][feature].value_counts()
    dead = tr[tr['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', figsize=(10,5))


# In[11]:


bar_chart('Sex')
print("Survived :\n",tr[tr['Survived']==1]['Sex'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['Sex'].value_counts())


# In[12]:


bar_chart('Pclass')
print("Survived :\n",tr[tr['Survived']==1]['Pclass'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['Pclass'].value_counts())


# In[13]:


bar_chart('SibSp')
print("Survived :\n",tr[tr['Survived']==1]['SibSp'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['SibSp'].value_counts())


# In[14]:


bar_chart('Embarked')
print("Survived :\n",tr[tr['Survived']==1]['Embarked'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['Embarked'].value_counts())


# In[16]:


a31=pd.Series(tr['Age'])
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


# In[17]:


cols = ['PassengerId','Name','Ticket','Cabin']
tr = tr.drop(cols, axis=1)

cols1 = ['PassengerId','Name','Ticket','Cabin']
ts = ts.drop(cols, axis=1)

#ts = ts.dropna()

display(tr.shape)
display(ts.shape)
#display(tr.head())
#display(ts.head())


# In[18]:


a=tr['Sex']
s=pd.Series(a)
#print(s)
for j, i in enumerate(s.values):
    if i=='male':
        s[j]=0
    else:
        s[j]=1
        
#print(s)    
tr['Sex']=s
display(tr.head())


# In[19]:


b=ts['Sex']
t=pd.Series(b)
#print(s)
for j, i in enumerate(t.values):
    if i=='male':
        t[j]=0
    else:
        t[j]=1
        
#print(s)    
ts['Sex']=t
display(ts.head())


# In[20]:


b1=tr['Embarked']
t1=pd.Series(b1)
for j, i in enumerate(t1.values):
    if i=='Q':
        t1[j]=0
    elif i=='S': 
        t1[j]=1
        
        
            
    else:
        t1[j]=2
        
#print(s)    
tr['Embarked']=t1
display(tr.head())


# In[21]:


b12=ts['Embarked']
t12=pd.Series(b12)
for j, i in enumerate(t12.values):
    if i=='Q':
        t12[j]=0
    elif i=='S': 
        t12[j]=1
        
        
            
    else:
        t12[j]=2
        
#print(s)    
ts['Embarked']=t12
display(ts.head())


# In[22]:


#FILLED THE MISSING DATA VALUES USING INTERPOLATE FUNCTION

tr['Age'] = tr['Age'].interpolate()
ts['Age'] = ts['Age'].interpolate()
ts['Fare'] = ts['Fare'].interpolate()


display(tr.info())
display(ts.info())


# In[23]:


#SPLITTING INPUT VALUES AND OUTPUT 

x = tr.values 
y = tr['Survived'].values
x = np.delete(x, 0, axis=1)

x1 = ts.values
#print(y)

print(x)
print(x1)


# In[24]:


from sklearn import preprocessing
x = preprocessing.normalize(x)
x1 = preprocessing.normalize(x1)


# In[25]:


ac = pd.read_csv("gender_submission.csv")
pid = ['PassengerId']
ac = ac.drop(pid, axis=1)
print(ac.shape)


# In[31]:


#MODEL TRAINING AND PREDICTING

from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
k = range(1,7)
l=[]
for i in k:
    

    model = KNeighborsClassifier(n_neighbors=i)

    #TRAINING THE MODEL USING DATASET
    model.fit(x, y)

    #PREDICTING OUTPUT
    predicted= model.predict(x1) 
    #print(predicted)
    
    
    print("Accuracy for k = ",i,"is",metrics.accuracy_score(ac, predicted))
    #print(confusion_matrix(ac, predicted))
    #print(classification_report(ac, predicted))
    print('\n')
    l.insert(i,metrics.accuracy_score(ac, predicted))


# In[32]:


a5 = predicted[predicted==1].size
a6 = predicted[predicted==0].size


dm = pd.DataFrame([a5,a6])
dm.plot(kind='pie', subplots=True,labels=['survived','dead'], startangle=180, explode=[0,0.04], autopct='%1.1f%%',colors=['g','r'])
plt.legend()
plt.show()


# In[33]:


plt.plot(k,l)
plt.xlabel('k neighbours')
plt.ylabel('accuracy')
plt.plot()



