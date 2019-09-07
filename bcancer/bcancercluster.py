import numpy as np
import pandas as pd
from clustering import SemiSupervisedKMeans
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)
f = open('datasets/breastcancer/breastcancer-labeled2.csv')
u = open('datasets/breastcancer/breastcancer-unlabeled2.csv')
df = pd.read_csv(f)
udf = pd.read_csv(u)

total_count = df.shape[0]
train_count = int(total_count)
test_count = total_count - train_count

total_runs = 50
all_stats = np.empty((total_runs, 4))
sample = 0

kfold = StratifiedKFold(n_splits=10, shuffle=True)

while sample < total_runs:
    try:
        training_data = df.values
        training_labels = training_data[:, 0]
        training_data = np.delete(training_data, 0, 1)
        unlabeled_data = udf.values
        unlabeled_labels = unlabeled_data[:, 0]
        unlabeled_data = np.delete(unlabeled_data, 0, 1)

        kmeans = SemiSupervisedKMeans(num_clusters=2)
        kmeans.initialize(training_data, training_labels)
        kmeans.fit(unlabeled_data, 7)

        # Label prediction accuracy setup
        unlabeled_truth = np.insert(unlabeled_data, 0, unlabeled_labels, axis=1)
        unlabeled_truth_set = set([tuple(x) for x in unlabeled_truth])

        # Clustering prediction accuracy
        cluster_unlabeled_predictions = kmeans.get_unlabeled_predictions()
        cluster_unlabeled_predictions_set = set([tuple(x) for x in cluster_unlabeled_predictions])
        cluster_correct_matches = np.array([x for x in cluster_unlabeled_predictions_set & unlabeled_truth_set])
        cluster_label_acc = len(cluster_correct_matches) / len(cluster_unlabeled_predictions)
        print('clustering labeling accuracy: %s' % cluster_label_acc)
        print()

        # Combined SVM performance
        all_predicted_data = cluster_unlabeled_predictions
        all_predicted_labels = all_predicted_data[:, 0]
        all_predicted_data = all_predicted_data[:, 1:]
        cvscores = list()
        for train, test in kfold.split(training_data, training_labels):
            clf = SVC(gamma='auto', C=1, kernel='rbf')
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
        cluster_labels_given = len(cluster_unlabeled_predictions)
        run_stats = np.array([total_unlabeled_count,
                              cluster_labels_given, cluster_label_acc,
                              avg_score])
        all_stats[sample] = run_stats
        sample += 1
    except:
        print('Failed run')
mean_stats = np.mean(all_stats, axis=0)
stddev = all_stats[:, 3:].flatten().std()
mean_stats = np.insert(mean_stats, 4, stddev)
np.savetxt('bcancercluster.csv', mean_stats, fmt='%.3e', delimiter=',')
print(mean_stats)
