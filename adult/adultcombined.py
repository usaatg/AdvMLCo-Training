import numpy as np
import pandas as pd
from cotrain import Cotrain
from sklearn.svm import SVC
from clustering import SemiSupervisedKMeans
from sklearn.model_selection import StratifiedKFold
from pathlib import Path

base_path = Path(__file__).parent
labeled_path = (base_path / "../datasets/Income/adult-labeled.csv").resolve()
unlabeled_path = (base_path / "../datasets/Income/adult-unlabeled.csv").resolve()

f = open(labeled_path)
u = open(unlabeled_path)
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
df = pd.read_csv(f)
udf = pd.read_csv(u)

total_count = df.shape[0]
train_count = int(total_count)

total_runs = 1
all_stats = np.empty((total_runs, 9))
sample = 0

kfold = StratifiedKFold(n_splits=10, shuffle=True)

avg_nn_pred_confidence = list()
avg_svm_pred_confidence = list()

# while sample < total_runs:
#     try:
training_data = df.values
training_labels = training_data[:, 0]
training_data = np.delete(training_data, 0, 1)
unlabeled_data = udf.values
unlabeled_labels = unlabeled_data[:, 0]
unlabeled_data = np.delete(unlabeled_data, 0, 1)

print('cotraining training count: %s' % train_count)
print('cotraining unlabeled count: %s' % unlabeled_data.shape[0])
print()

cotrain_model = Cotrain()
nn_args = {'hidden_nodes': 8, 'epochs': 50, 'batch_size': 100}
svm_args = {'kernel': 'rbf', 'c': 100, 'gamma': 0.0001}
cotrain_model.initialize(training_data, training_labels, nn_args, svm_args)
cotrain_model.fit(unlabeled_data, 0.65)

# Label prediction accuracy setup
unlabeled_truth = np.insert(unlabeled_data, 0, unlabeled_labels, axis=1)
unlabeled_truth_set = set([tuple(x) for x in unlabeled_truth])

# Cotraining label prediction accuracy
avg_nn_pred_confidence.append(cotrain_model.avg_nn_pred_confidence)
avg_svm_pred_confidence.append(cotrain_model.avg_svm_pred_confidence)
cotrain_unlabeled_predictions = cotrain_model.get_unlabeled_predictions()
cotrain_unlabeled_predictions_set = set([tuple(x) for x in cotrain_unlabeled_predictions])
cotrain_correct_matches = np.array([x for x in cotrain_unlabeled_predictions_set & unlabeled_truth_set])
cotrain_label_acc = len(cotrain_correct_matches) / len(cotrain_unlabeled_predictions)
print('cotraining labeling accuracy: %s' % cotrain_label_acc)
print()

cotrain_unpredicted_data = cotrain_model.get_unpredicted_data()
kmeans = SemiSupervisedKMeans(num_clusters=2)
kmeans.initialize(training_data, training_labels)
kmeans.fit(unlabeled_data, 4500)

# Clustering prediction accuracy
cluster_unlabeled_predictions = kmeans.get_unlabeled_predictions()
cluster_unlabeled_predictions_set = set([tuple(x) for x in cluster_unlabeled_predictions])
cluster_correct_matches = np.array([x for x in cluster_unlabeled_predictions_set & unlabeled_truth_set])
cluster_label_acc = len(cluster_correct_matches) / len(cluster_unlabeled_predictions)
print('clustering labeling accuracy: %s' % cluster_label_acc)
print()

labeled_data = np.insert(training_data, 0, training_labels, axis=1)
remaining_unlabeled_predictions = kmeans.predict(cotrain_unpredicted_data, 4500)
complete_training_data = np.vstack([labeled_data, cotrain_unlabeled_predictions, remaining_unlabeled_predictions])

# Remaining data labeling accuracy
remaining_unlabeled_predictions_set = set([tuple(x) for x in remaining_unlabeled_predictions])
remaining_preds_correct_matches = np.array([x for x in remaining_unlabeled_predictions_set & unlabeled_truth_set])
remaining_preds_acc = len(remaining_preds_correct_matches) / len(remaining_unlabeled_predictions)
print('clustering remaining data labeling accuracy: %s' % remaining_preds_acc)
print()

# Combined SVM performance
all_predicted_data = cotrain_unlabeled_predictions
all_predicted_labels = all_predicted_data[:, 0]
all_predicted_data = all_predicted_data[:, 1:]
cvscores = list()
for train, test in kfold.split(training_data, training_labels):
    clf = SVC(gamma=0.0001, C=100, kernel='rbf')
    kfold_labeled_data = training_data[train]
    kfold_labeled_labels = training_labels[train]
    kfold_test_data = training_data[test]
    kfold_test_labels = training_labels[test]
    kfold_train = np.vstack([kfold_labeled_data, all_predicted_data])
    kfold_train_labels = np.concatenate([kfold_labeled_labels, all_predicted_labels])
    clf.fit(kfold_train, kfold_train_labels)
    score = clf.score(kfold_test_data, kfold_test_labels)
    cvscores.append(score)

avg_score = np.mean(cvscores)
cv_stddev = np.std(cvscores)
print('combined cv scores: %s' % cvscores)
print("combined Accuracy: %0.2f (+/- %0.2f)" % (avg_score, cv_stddev * 2))
print()

total_unlabeled_count = len(unlabeled_data)
cotrain_labels_given = len(cotrain_unlabeled_predictions)
cluster_labels_given = len(cluster_unlabeled_predictions)
remaining_pred_count = len(cotrain_unpredicted_data)
remaining_pred_labels_given = len(remaining_unlabeled_predictions)
run_stats = np.array([total_unlabeled_count,
                      cotrain_labels_given, cotrain_label_acc,
                      cluster_labels_given, cluster_label_acc,
                      remaining_pred_count, remaining_pred_labels_given, remaining_preds_acc,
                      avg_score])
all_stats[sample] = run_stats
sample += 1
    # except:
    #     print('Failed run')
mean_stats = np.mean(all_stats, axis=0)
stddev = all_stats[:, 8:].flatten().std()
mean_stats = np.insert(mean_stats, 9, stddev)
print(mean_stats)
print('avg nn confidence: %s' % np.mean(avg_nn_pred_confidence))
print('avg svm confidence: %s' % np.mean(avg_svm_pred_confidence))
