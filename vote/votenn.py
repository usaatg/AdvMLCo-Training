import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import OneHotEncoder
from keras.metrics import categorical_accuracy
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

base_path = Path(__file__).parent
path = (base_path / "../datasets/Vote/vote-labeled.csv").resolve()


def label_one_hot_encode(labels):
    label_reshape = labels.reshape(-1, 1)
    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(label_reshape)
    return enc.transform(label_reshape).toarray()


np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
f = open(path)
df = pd.read_csv(f)

total_count = df.shape[0]
train_count = int(total_count)
test_count = total_count - train_count

total_runs = 1
all_stats = np.empty((total_runs, 2))
sample = 0

kfold = StratifiedKFold(n_splits=10, shuffle=True)

# while sample < total_runs:
#     try:
training_data = df.values
training_labels = training_data[:, 0]
training_data = np.delete(training_data, 0, 1)
training_one_hot_encoded_labels = label_one_hot_encode(training_labels)

cvscores = list()
for train, test in kfold.split(training_data, training_labels):
    print('training count: %s' % train_count)
    print('test count: %s' % test_count)

    nn = Sequential()
    nn.add(Dense(9, activation='relu', input_shape=training_data[0].shape))
    nn.add(Dense(units=2, activation='softmax'))
    nn.compile(optimizer='adadelta',
                  loss='categorical_crossentropy',
                  metrics=['accuracy', categorical_accuracy])
    nn.fit(training_data[train], training_one_hot_encoded_labels[train], validation_split=0.2, epochs=100, batch_size=10)

    labels_encoded = label_one_hot_encode(training_labels[test])
    val_score = nn.evaluate(training_data[test], labels_encoded, verbose=0)[2]
    cvscores.append(val_score)

avg_val_score = np.mean(cvscores)
print('val acc: %s' % np.mean(cvscores))
run_stats = np.array([train_count,
                      avg_val_score])
all_stats[sample] = run_stats
sample += 1
    # except:
    #     print('Failed run')
mean_stats = np.mean(all_stats, axis=0)
stddev = all_stats[:, 1:].flatten().std()
mean_stats = np.insert(mean_stats, 2, stddev)
print(mean_stats)