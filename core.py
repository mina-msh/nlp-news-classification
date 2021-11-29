from __future__ import unicode_literals
import nltk as nk
import pandas as pd
import numpy as np
import autocorrect as ac
from autocorrect import spell
from __future__ import unicode_literals
from hazm import *
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

dataset = "NLP datatset title.csv"

df = pd.read_csv(dataset , encoding='utf-8' )
print(df.head())

sns.countplot(x=df["news path"])
plt.show()
# normalizer = Normalizer()
# normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند')