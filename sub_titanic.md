
# Titanic survival analysis
```python
import numpy as np
import pandas as pd
```


```python
data = pd.read_csv("train.csv")
display(data.head())
display(data.info())
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


## Data visualization
```python
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```


```python
sur = data[data['Survived']==1].size
de = data[data['Survived']==0].size
ps=pd.DataFrame([sur,de])
ps.plot(kind='pie', subplots=True,labels=['survived','dead'], startangle=180, explode=[0,0.04], autopct='%1.1f%%',colors=['g','r'])
plt.legend()
plt.show()
```
#### we can see that approx 40% got survived.


![png](output_3_0.png)



```python
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
```
##### there were more no of adults followed by senior citizens onboard.
    adult             476
    senior citizen    251
    teenager          102
    child              62
    Name: Age, dtype: int64
    


```python
def bar_chart(feature):
    survived = data[data['Survived']==1][feature].value_counts()
    dead = data[data['Survived']==0][feature].value_counts()
    df = pd.DataFrame([survived,dead])
    df.index = ['Survived','Dead']
    df.plot(kind='bar', figsize=(10,5))
```


```python
bar_chart('Sex')
print("Survived :\n",data[data['Survived']==1]['Sex'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Sex'].value_counts())
```
##### we can see that females are more likely survived.
    Survived :
     female    233
    male      109
    Name: Sex, dtype: int64
    Dead:
     male      468
    female     81
    Name: Sex, dtype: int64
    


![png](output_6_1.png)



```python
bar_chart('Pclass')
print("Survived :\n",data[data['Survived']==1]['Pclass'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Pclass'].value_counts())
```
##### passengers of class 3 were less likely survived.
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
    


![png](output_7_1.png)



```python
bar_chart('SibSp')
print("Survived :\n",data[data['Survived']==1]['SibSp'].value_counts())
print("Dead:\n",data[data['Survived']==0]['SibSp'].value_counts())
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
    


![png](output_8_1.png)



```python
bar_chart('Embarked')
print("Survived :\n",data[data['Survived']==1]['Embarked'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Embarked'].value_counts())
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
    


![png](output_9_1.png)



```python
bar_chart('Parch')
print("Survived :\n",data[data['Survived']==1]['Parch'].value_counts())
print("Dead:\n",data[data['Survived']==0]['Parch'].value_counts())
```

    Survived :
     0    233
    1     65
    2     40
    3      3
    5      1
    Name: Parch, dtype: int64
    Dead:
     0    445
    1     53
    2     40
    5      4
    4      4
    3      2
    6      1
    Name: Parch, dtype: int64
    


![png](output_10_1.png)



```python
figure = plt.figure(figsize=(25, 7))
plt.hist([data[data['Survived'] == 1]['Fare'], data[data['Survived'] == 0]['Fare']], 
         stacked=True, color = ['g','r'],
         bins = 50, label = ['Survived','Dead'])
plt.xlabel('Fare')
plt.ylabel('Number of passengers')
plt.legend();
```
##### Here, we can see that passengers paid high fare were more likely survived.


![png](output_11_0.png)



```python
tr = pd.read_csv("train.csv")
ts = pd.read_csv("test.csv")
```


```python
#print(tr.head())
#print(ts.head())
#print(tr.isnull().sum())
#print(ts.isnull().sum())
```

##### Droppped unwanted features.
```python
cols = ['PassengerId','Name','Ticket','Cabin']
tr = tr.drop(cols, axis=1)

cols1 = ['PassengerId','Name','Ticket','Cabin']
ts = ts.drop(cols, axis=1)

```


```python
#tr.info()
#ts.info()
```


```python
#SPLITTING COLUMNS OF P_CLASS, SEX, EMBARKED IN TRAINING DATASET 

dummies = []
cols2 = ['Pclass','Sex','Embarked']
for i in cols2:
    dummies.append(pd.get_dummies(tr[i]))

titanic_dummies = pd.concat(dummies, axis=1)
tr = pd.concat((tr,titanic_dummies),axis=1)

tr = tr.drop(['Pclass','Sex','Embarked'],axis=1)
display(tr.head(5))
```
##### Used get_dummies to classify pclass, sex, and embarked features.

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
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



```python
#SPLITTING COLUMNS OF P_CLASS, SEX, EMBARKED IN TEST DATASET 

dummies = []
cols3 = ['Pclass','Sex','Embarked']
for i in cols3:
    dummies.append(pd.get_dummies(ts[i]))

titanic_dummies1 = pd.concat(dummies, axis=1)
ts = pd.concat((ts,titanic_dummies1),axis=1)

ts = ts.drop(['Pclass','Sex','Embarked'],axis=1)
display(ts.head(5))
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
      <th>Age</th>
      <th>SibSp</th>
      <th>Parch</th>
      <th>Fare</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>female</th>
      <th>male</th>
      <th>C</th>
      <th>Q</th>
      <th>S</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>34.5</td>
      <td>0</td>
      <td>0</td>
      <td>7.8292</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>47.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.0000</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>62.0</td>
      <td>0</td>
      <td>0</td>
      <td>9.6875</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>27.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.6625</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>22.0</td>
      <td>1</td>
      <td>1</td>
      <td>12.2875</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
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
```
##### used interpolate function to fill the missing values in age.

```python
#SPLITTING INPUT VALUES AND OUTPUT 

x = tr.values 
y = tr['Survived'].values
x = np.delete(x, 0, axis=1)

x1 = ts.values
#print(y)
#print(x)
#print(x1)
```


```python
print(tr.shape)
print(y.shape)
print(ts.shape)
```

    (891, 13)
    (891,)
    (418, 12)
    

## Modelling
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
```

    Accuracy for k =  1 is 0.7248803827751196
    [[190  76]
     [ 39 113]]
    
    
    Accuracy for k =  2 is 0.7464114832535885
    [[240  26]
     [ 80  72]]
    
    
    Accuracy for k =  3 is 0.7727272727272727
    [[217  49]
     [ 46 106]]
    
    
    Accuracy for k =  4 is 0.7535885167464115
    [[230  36]
     [ 67  85]]
    
    
    Accuracy for k =  5 is 0.7727272727272727
    [[218  48]
     [ 47 105]]
    
    
    Accuracy for k =  6 is 0.7918660287081339
    [[237  29]
     [ 58  94]]
    
    
    Accuracy for k =  7 is 0.784688995215311
    [[217  49]
     [ 41 111]]
    
    
    Accuracy for k =  8 is 0.7799043062200957
    [[229  37]
     [ 55  97]]
    
    
    Accuracy for k =  9 is 0.7990430622009569
    [[216  50]
     [ 34 118]]
    
    
    Accuracy for k =  10 is 0.7822966507177034
    [[224  42]
     [ 49 103]]
    
    
    


```python
plt.plot(k,l)
plt.xlabel('k neighbours')
plt.ylabel('accuracy')
plt.plot()
```




    []




![png](output_24_1.png)



```python
'''
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

svc = SVC()
svc.fit(x, y)
y_pred = svc.predict(x1)
acc_svc = round(accuracy_score(ac, y_pred) * 100, 2)
print(acc_svc)
'''
```

    65.07
    


```python
'''
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gaussian = GaussianNB()
gaussian.fit(x, y)
y_pred = gaussian.predict(x1)
acc_gaussian = round(accuracy_score(y_pred, ac) * 100, 2)
print(acc_gaussian)
'''
```

    80.14
    


```python

```
