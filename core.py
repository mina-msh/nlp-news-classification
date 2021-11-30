from __future__ import unicode_literals
import nltk as nk
import numpy as np
import autocorrect as ac
from autocorrect import spell
from __future__ import unicode_literals
import hazm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = "NLP datatset title.csv"

# reading and ploting data
df = pd.read_csv(dataset , encoding='utf-8' )
print(df.head())
sns.countplot(x=df["News path"])
plt.show()

# x is input data
# y is output data
X = df["Title"]
Y = df["News path"]

# adding int index for labels from 0 to 3
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
Y = le.fit_transform(Y)
print(Y)
sns.countplot(Y)
plt.show()
# segmentation to 80% test and 20% train
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
