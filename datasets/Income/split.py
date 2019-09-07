from sklearn.preprocessing import LabelEncoder
import pandas as pd

f = open('adult.data')
df = pd.read_csv(f)
df_cat = df.select_dtypes(include=[object])
le = LabelEncoder()
df['workclass'] = le.fit_transform(df['workclass'])
df['education'] = le.fit_transform(df['education'])
df['maritalstatus'] = le.fit_transform(df['maritalstatus'])
df['occupation'] = le.fit_transform(df['occupation'])
df['relationship'] = le.fit_transform(df['relationship'])
df['race'] = le.fit_transform(df['race'])
df['sex'] = le.fit_transform(df['sex'])
df['nativecountry'] = le.fit_transform(df['nativecountry'])
df['class'] = le.fit_transform(df['class'])
classcol = df[['class']]
del df['class']
df.insert(0, 'class', classcol)

total_count = df.shape[0] * 0.2
unlabeled_count = int(total_count * 0.6)
labeled_count = int(total_count * 0.4)

pos_count = int(labeled_count / 2)
neg_count = labeled_count - pos_count
pos_sample = df.loc[df['class']==1].sample(pos_count)
neg_sample = df.loc[df['class']==0].sample(neg_count)
labeled_sample = pd.concat([pos_sample, neg_sample])
df = df[~df.index.isin(labeled_sample.index)]
unlabeled_sample = df.sample(unlabeled_count)

labeled_sample.to_csv('adult-labeled.csv', index=False, index_label=False)
unlabeled_sample.to_csv('adult-unlabeled.csv', index=False, index_label=False)
