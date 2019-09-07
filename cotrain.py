import numpy as np
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder


class Cotrain:
    def __init__(self):
        self.nn = None
        self.svm = None
        self.labeled_data = None
        self.labels = None
        self.enc = None
        self.num_labeled = list()
        self.unpredicted = None
        self.predicted = None
        self.avg_nn_pred_confidence = 0
        self.avg_svm_pred_confidence = 0

    def initialize(self, labeled_data, labels, nn_args, svm_args):
        self.labeled_data = labeled_data
        self.labels = labels
        num_unique_labels = len(np.unique(labels))
        num_hidden_nodes = nn_args['hidden_nodes']
        epochs = nn_args['epochs']
        batch_size = nn_args['batch_size']
        kernel = svm_args['kernel']
        c = svm_args['c']
        gamma = svm_args['gamma']

        nn_label_one_hot_encode = self.label_one_hot_encode(labels)
        shape = labeled_data[0].shape
        self.nn = Sequential()
        self.nn.add(Dense(num_hidden_nodes, activation='relu', input_shape=shape))
        self.nn.add(Dense(units=num_unique_labels, activation='softmax'))
        self.nn.compile(optimizer='adadelta',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.nn.fit(labeled_data, nn_label_one_hot_encode, epochs=epochs, batch_size=batch_size)

        print('----- FITTING COTRAINING SVM -----')
        self.svm = SVC(kernel=kernel, C=c, gamma=gamma, probability=True)
        self.svm.fit(labeled_data, labels)
        # scores = cross_val_score(self.svm, labeled_data, labels, cv=10)
        # print('svm cv scores: %s' % scores)
        # print("svm Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    def fit(self, unlabeled_data, conf_threshold):
        unlabeled_copy = np.copy(unlabeled_data)
        converged = False
        svm_most_confident_preds = list()
        nn_most_confident_preds = list()
        while not converged:
            nn_preds = self.nn.predict_proba(unlabeled_copy)
            nn_rows_and_preds_above_thresh = np.argwhere(nn_preds >= conf_threshold)

            svm_preds = self.svm.predict_proba(unlabeled_copy)
            svm_rows_and_preds_above_thresh = np.argwhere(svm_preds >= conf_threshold)

            svm_most_confident_preds += np.amax(svm_preds, axis=1).tolist()
            nn_most_confident_preds += np.amax(nn_preds, axis=1).tolist()

            nn_set = set([tuple(x) for x in nn_rows_and_preds_above_thresh])
            svm_set = set([tuple(x) for x in svm_rows_and_preds_above_thresh])
            matching_rows = np.array([x for x in nn_set & svm_set])
            converged = len(matching_rows) == 0
            if not converged:
                matching_rows = matching_rows[matching_rows[:, 0].argsort()]

                confident_row_indices = matching_rows[:, 0].flatten()
                confident_rows = unlabeled_copy[confident_row_indices]
                confident_labels = matching_rows[:, 1].flatten()

                # Add confident rows to labeled data
                self.labeled_data = np.vstack([self.labeled_data, confident_rows])
                self.labels = np.concatenate((self.labels, confident_labels))
                confident_one_hot_labels = self.label_one_hot_encode(confident_labels)
                converged = len(confident_labels) == 0

                # Delete unconfident rows from unlabeled data
                unlabeled_copy = np.delete(unlabeled_copy, confident_row_indices, axis=0)

                self.num_labeled.append(confident_rows.shape[0])
                confident_rows_with_labels = np.insert(confident_rows, 0, confident_labels, axis=1)
                self.add_to_predicted(confident_rows_with_labels)

                # Refit models
                self.nn.fit(confident_rows, confident_one_hot_labels, epochs=50, batch_size=10)
                print('----- REFITTING COTRAINING SVM -----')
                print('# Confident predictions: %s' % confident_rows.shape[0])
                print('# Unlabeled remaining: %s' % unlabeled_copy.shape[0])
                self.svm.fit(self.labeled_data, self.labels)
        self.unpredicted = unlabeled_copy
        self.avg_nn_pred_confidence = np.mean(nn_most_confident_preds)
        self.avg_svm_pred_confidence = np.mean(svm_most_confident_preds)

        print('total unlabeled: %s ' % len(unlabeled_data))
        print('total labels given: %s ' % len(self.predicted) if self.predicted is not None else 0)
        print('avg nn confidence: %s' % self.avg_nn_pred_confidence)
        print('avg svm confidence: %s' % self.avg_svm_pred_confidence)

    def label_one_hot_encode(self, labels):
        label_reshape = labels.reshape(-1, 1)
        if self.enc is None:
            self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(label_reshape)
        return self.enc.transform(label_reshape).toarray()

    def get_full_labeled_data(self):
        return np.insert(self.labeled_data, 0, self.labels, axis=1)  # Add label to unlabeled point

    def get_unpredicted_data(self):
        return self.unpredicted

    def get_unlabeled_predictions(self):
        return self.predicted

    def add_to_predicted(self, predicted_rows):
        if self.predicted is None:
            self.predicted = predicted_rows
        else:
            self.predicted = np.vstack([self.predicted, predicted_rows])

    def get_avg_nn_pred_confidence(self):
        return self.avg_nn_pred_confidence

    def get_avg_svm_pred_confidence(self):
        return self.avg_svm_pred_confidence
