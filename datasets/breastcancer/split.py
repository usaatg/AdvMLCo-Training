import os
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

f = open('breast-cancer.data')
df = pd.read_csv(f)
df_cat = df.select_dtypes(include=[object])
le = LabelEncoder()
df2 = df.apply(le.fit_transform)

total_count = df2.shape[0]
unlabeled_count = int(total_count * 0.6)
labeled_count = int(total_count * 0.4)

pos_count = int(labeled_count / 2)
neg_count = labeled_count - pos_count
pos_sample = df2.loc[df2['class']==1].sample(pos_count)
neg_sample = df2.loc[df2['class']==0].sample(neg_count)
labeled_sample = pd.concat([pos_sample, neg_sample])
df2 = df2[~df2.index.isin(labeled_sample.index)]

labeled_sample.to_csv('breastcancer-labeled2.csv', index=False, index_label=False)
df2.to_csv('breastcancer-unlabeled2.csv', index=False, index_label=False)

# unlabeled_sample = df2.sample(unlabeled_count)
# df2 = df2[~df2.index.isin(unlabeled_sample.index)]
# unlabeled_sample.to_csv('breastcancer-unlabeled.csv', index=False, index_label=False)
# df2.to_csv('breastcancer-labeled.csv', index=False, index_label=False)


