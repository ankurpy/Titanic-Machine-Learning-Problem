# Titanic problem analysis

```python
import numpy as np
import pandas as pd
```


```python
tr = pd.read_csv("train.csv")
ts = pd.read_csv("test.csv")
display(tr.head())
display(ts.head())
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>Braund, Mr. Owen Harris</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>A/5 21171</td>
      <td>7.2500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>Cumings, Mrs. John Bradley (Florence Briggs Th...</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>PC 17599</td>
      <td>71.2833</td>
      <td>C85</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>3</td>
      <td>Heikkinen, Miss. Laina</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>STON/O2. 3101282</td>
      <td>7.9250</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>Futrelle, Mrs. Jacques Heath (Lily May Peel)</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>113803</td>
      <td>53.1000</td>
      <td>C123</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>0</td>
      <td>3</td>
      <td>Allen, Mr. William Henry</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>373450</td>
      <td>8.0500</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PassengerId</th>
      <th>Pclass</th>
      <th>Name</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Ticket</th>
      <th>Fare</th>
      <th>Cabin</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>892</td>
      <td>3</td>
      <td>Kelly, Mr. James</td>
      <td>male</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>330911</td>
      <td>7.8292</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>893</td>
      <td>3</td>
      <td>Wilkes, Mrs. James (Ellen Needs)</td>
      <td>female</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>363272</td>
      <td>7.0000</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>894</td>
      <td>2</td>
      <td>Myles, Mr. Thomas Francis</td>
      <td>male</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>240276</td>
      <td>9.6875</td>
      <td>NaN</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>895</td>
      <td>3</td>
      <td>Wirz, Mr. Albert</td>
      <td>male</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>315154</td>
      <td>8.6625</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>896</td>
      <td>3</td>
      <td>Hirvonen, Mrs. Alexander (Helga E Lindqvist)</td>
      <td>female</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>3101298</td>
      <td>12.2875</td>
      <td>NaN</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



```python
print(tr.info())
print(ts.info())
#display(tr.isnull().sum())
#display(ts.isnull().sum())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 12 columns):
    PassengerId    891 non-null int64
    Survived       891 non-null int64
    Pclass         891 non-null int64
    Name           891 non-null object
    Sex            891 non-null object
    Age            714 non-null float64
    SibSp          891 non-null int64
    Parch          891 non-null int64
    Ticket         891 non-null object
    Fare           891 non-null float64
    Cabin          204 non-null object
    Embarked       889 non-null object
    dtypes: float64(2), int64(5), object(5)
    memory usage: 83.6+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 11 columns):
    PassengerId    418 non-null int64
    Pclass         418 non-null int64
    Name           418 non-null object
    Sex            418 non-null object
    Age            332 non-null float64
    SibSp          418 non-null int64
    Parch          418 non-null int64
    Ticket         418 non-null object
    Fare           417 non-null float64
    Cabin          91 non-null object
    Embarked       418 non-null object
    dtypes: float64(2), int64(4), object(5)
    memory usage: 36.0+ KB
    None
    


```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```


```python
sur=tr[tr['Survived']==1].size
de=tr[tr['Survived']==0].size
ps=pd.DataFrame([sur,de])
ps.plot(kind='pie', subplots=True,labels=['survived','dead'], startangle=180, explode=[0,0.04], autopct='%1.1f%%',colors=['g','r'])
plt.legend()
plt.show()
```


![png](output_4_0.png)



```python
tr['Age'].hist(bins=50)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x267a8b15c50>




![png](output_5_1.png)



```python
def bar_chart(feature):
    survived = tr[tr['Survived']==1][feature].value_counts()
    dead = tr[tr['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', figsize=(10,5))
```


```python
bar_chart('Sex')
print("Survived :\n",tr[tr['Survived']==1]['Sex'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['Sex'].value_counts())
```

    Survived :
     female    233
    male      109
    Name: Sex, dtype: int64
    Dead:
     male      468
    female     81
    Name: Sex, dtype: int64
    


![png](output_7_1.png)



```python
bar_chart('Pclass')
print("Survived :\n",tr[tr['Survived']==1]['Pclass'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['Pclass'].value_counts())
```

    Survived :
     1    136
    3    119
    2     87
    Name: Pclass, dtype: int64
    Dead:
     3    372
    2     97
    1     80
    Name: Pclass, dtype: int64
    


![png](output_8_1.png)



```python
bar_chart('SibSp')
print("Survived :\n",tr[tr['Survived']==1]['SibSp'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['SibSp'].value_counts())
```

    Survived :
     0    210
    1    112
    2     13
    3      4
    4      3
    Name: SibSp, dtype: int64
    Dead:
     0    398
    1     97
    4     15
    2     15
    3     12
    8      7
    5      5
    Name: SibSp, dtype: int64
    


![png](output_9_1.png)



```python
bar_chart('Embarked')
print("Survived :\n",tr[tr['Survived']==1]['Embarked'].value_counts())
print("Dead:\n",tr[tr['Survived']==0]['Embarked'].value_counts())
```

    Survived :
     S    217
    C     93
    Q     30
    Name: Embarked, dtype: int64
    Dead:
     S    427
    C     75
    Q     47
    Name: Embarked, dtype: int64
    


![png](output_10_1.png)



```python
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
```

    adult             476
    senior citizen    251
    teenager          102
    child              62
    Name: Age, dtype: int64
    


```python
cols = ['PassengerId','Name','Ticket','Cabin']
tr = tr.drop(cols, axis=1)

cols1 = ['PassengerId','Name','Ticket','Cabin']
ts = ts.drop(cols, axis=1)

#ts = ts.dropna()

display(tr.shape)
display(ts.shape)
#display(tr.head())
#display(ts.head())
```


    (891, 8)



    (418, 7)



```python
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
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>S</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>Q</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>S</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>S</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Survived</th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
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
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Pclass</th>
      <th>Sex</th>
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>Embarked</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3</td>
      <td>0</td>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3</td>
      <td>1</td>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0</td>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3</td>
      <td>0</td>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3</td>
      <td>1</td>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
#FILLED THE MISSING DATA VALUES USING INTERPOLATE FUNCTION

tr['Age'] = tr['Age'].interpolate()
ts['Age'] = ts['Age'].interpolate()
ts['Fare'] = ts['Fare'].interpolate()


display(tr.info())
display(ts.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 8 columns):
    Survived    891 non-null int64
    Pclass      891 non-null int64
    Sex         891 non-null object
    Age         891 non-null float64
    SibSp       891 non-null int64
    Parch       891 non-null int64
    Fare        891 non-null float64
    Embarked    891 non-null object
    dtypes: float64(2), int64(4), object(2)
    memory usage: 55.8+ KB
    


    None


    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 418 entries, 0 to 417
    Data columns (total 7 columns):
    Pclass      418 non-null int64
    Sex         418 non-null object
    Age         418 non-null float64
    SibSp       418 non-null int64
    Parch       418 non-null int64
    Fare        418 non-null float64
    Embarked    418 non-null object
    dtypes: float64(2), int64(3), object(2)
    memory usage: 22.9+ KB
    


    None



```python
#SPLITTING INPUT VALUES AND OUTPUT 

x = tr.values 
y = tr['Survived'].values
x = np.delete(x, 0, axis=1)

x1 = ts.values
#print(y)

print(x)
print(x1)
```

    [[3 0 22.0 ... 0 7.25 1]
     [1 1 38.0 ... 0 71.2833 2]
     [3 1 26.0 ... 0 7.925 1]
     ...
     [3 1 22.5 ... 2 23.45 1]
     [1 0 26.0 ... 0 30.0 2]
     [3 0 32.0 ... 0 7.75 0]]
    [[3 0 34.5 ... 0 7.8292 0]
     [3 1 47.0 ... 0 7.0 1]
     [2 0 62.0 ... 0 9.6875 0]
     ...
     [3 0 38.5 ... 0 7.25 1]
     [3 0 38.5 ... 0 8.05 1]
     [3 0 38.5 ... 1 22.3583 2]]
    


```python
from sklearn import preprocessing
x = preprocessing.normalize(x)
x1 = preprocessing.normalize(x1)
```


```python
ac = pd.read_csv("gender_submission.csv")
pid = ['PassengerId']
ac = ac.drop(pid, axis=1)
print(ac.shape)
```

    (418, 1)
    


```python
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
```

    Accuracy for k =  1 is 0.6889952153110048
    
    
    Accuracy for k =  2 is 0.6985645933014354
    
    
    Accuracy for k =  3 is 0.7464114832535885
    
    
    Accuracy for k =  4 is 0.7368421052631579
    
    
    Accuracy for k =  5 is 0.7464114832535885
    
    
    Accuracy for k =  6 is 0.7368421052631579
    
    
    


```python
a5 = predicted[predicted==1].size
a6 = predicted[predicted==0].size


dm = pd.DataFrame([a5,a6])
dm.plot(kind='pie', subplots=True,labels=['survived','dead'], startangle=180, explode=[0,0.04], autopct='%1.1f%%',colors=['g','r'])
plt.legend()
plt.show()
```


![png](output_22_0.png)



```python
plt.plot(k,l)
plt.xlabel('k neighbours')
plt.ylabel('accuracy')
plt.plot()
```




    []




![png](output_23_1.png)



