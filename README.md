# Titanic survival analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```
## Loading datasets
```python
tr = pd.read_csv("train.csv")
ts = pd.read_csv("test.csv")
ac = pd.read_csv("gender_submission.csv")
```
```python
tr.info()
```
```python
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
``` 
We can see that values are missing in Age, Cabin and Embarked.
```python
ts.info()
```
```python
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
```
In test data, values are missing in Age, Cabin and Fare.

## Data visualization
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
print("Survived :\n",train[train['Survived']==1]['Sex'].value_counts())
print("Dead:\n",train[train['Survived']==0]['Sex'].value_counts())
```

    Survived :
     female    233
    male      109
    Name: Sex, dtype: int64
    Dead:
     male      468
    female     81
    Name: Sex, dtype: int64
![png](sdjvnswjvnjn.png)
