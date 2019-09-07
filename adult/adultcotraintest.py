import numpy as np
import pandas as pd
from cotrain import Cotrain
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from pathlib import Path

base_path = Path(__file__).parent
labeled_path = (base_path / "../datasets/Income/adult-labeled.csv").resolve()
unlabeled_path = (base_path / "../datasets/Income/adult-unlabeled.csv").resolve()

f = open(labeled_path)
u = open(unlabeled_path)
df = pd.read_csv(f)
udf = pd.read_csv(u)

total_count = df.shape[0]
train_count = int(total_count)

training_data = df.values
training_labels = training_data[:, 0]
training_data = np.delete(training_data, 0, 1)
unlabeled_data = udf.values
unlabeled_labels = unlabeled_data[:, 0]
unlabeled_data = np.delete(unlabeled_data, 0, 1)

print('training count: %s' % train_count)
print('unlabeled count: %s' % unlabeled_data.shape[0])

cotrain_model = Cotrain()
nn_args = {'hidden_nodes': 8, 'epochs': 25, 'batch_size': 100}
svm_args = {'kernel': 'rbf', 'c': 100, 'gamma': 0.0001}
cotrain_model.initialize(training_data, training_labels, nn_args, svm_args)
cotrain_model.fit(unlabeled_data, 0.6)

# Label prediction accuracy setup
unlabeled_truth = np.insert(unlabeled_data, 0, unlabeled_labels, axis=1)
unlabeled_truth_set = set([tuple(x) for x in unlabeled_truth])

# Cotraining label prediction accuracy
cotrain_unlabeled_predictions = cotrain_model.get_unlabeled_predictions()
cotrain_unlabeled_predictions_set = set([tuple(x) for x in cotrain_unlabeled_predictions])
cotrain_correct_matches = np.array([x for x in cotrain_unlabeled_predictions_set & unlabeled_truth_set])
cotrain_label_acc = len(cotrain_correct_matches) / len(cotrain_unlabeled_predictions)
print('cotraining labeling accuracy: %s' % cotrain_label_acc)
print()

# SVM performance
new_labeled_data = cotrain_model.get_full_labeled_data()
labels = new_labeled_data[:, 0]
new_labeled_data = new_labeled_data[:, 1:]

X_train, X_test, y_train, y_test = train_test_split(
    new_labeled_data, labels, test_size=0.5, random_state=0)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4, 'auto'],
                     'C': [1, 10, 100, 1000]}]

scores = ['accuracy']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

    print("Detailed classification report:")
    print()
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, clf.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()



