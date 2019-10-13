# Titanic survival analysis

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.set()
```
```python
tr = pd.read_csv("train.csv")
ts = pd.read_csv("test.csv")
ac = pd.read_csv("gender_submission.csv")
```
```python
tr.info()
```
