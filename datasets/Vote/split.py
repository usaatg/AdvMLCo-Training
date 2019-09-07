from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import pandas as pd

f = open('house-votes-84.data')
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

labeled_sample.to_csv('vote-labeled.csv', index=False, index_label=False)
df2.to_csv('vote-unlabeled.csv', index=False, index_label=False)
