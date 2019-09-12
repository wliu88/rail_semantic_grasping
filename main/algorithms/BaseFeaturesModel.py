import numpy as np
import pickle

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from pylmnn import LargeMarginNearestNeighbor as LMNN


class BaseFeaturesModel:

    def __init__(self, classifier="KNN"):
        self.classifier = classifier
        assert classifier in ["KNN", "LR", "LMNN"]

    def run_experiments(self, data, experiments, save_filename):
        """
        This method runs all experiments provided and reports the results
        results is in format (description, ground truth labels, prediction scores), where prediction scores is a
        multi-dimensional array where the first dimension represents the object instance, the second the grasps, the
        third the score for each class label

        :param experiments:
        :return:
        """
        results = []

        for experiment in experiments:
            description, train_objs, test_objs = experiment
            # print("\nRun experiment {}".format(description))

            # normalize features
            scalar = preprocessing.StandardScaler()

            # vectorize training and testing data
            for split in ["train", "test"]:

                features = []
                histograms = []
                descriptors = []
                labels = []

                if split == "train":
                    objs = train_objs
                elif split == "test":
                    objs = test_objs

                for obj in objs:
                    for id in obj:
                        label = data[id][0]
                        extracted_base_features = data[id][2]
                        # extracted_base_features = (features, histograms, descriptor)
                        features.append(extracted_base_features[0])
                        histograms.append(extracted_base_features[1])
                        descriptors.append(extracted_base_features[2])
                        labels.append(label)

                features = np.array(features)
                histograms = np.array(histograms)
                descriptors = np.array(descriptors)

                # preprocess features
                if split == "train":
                    scalar.fit(features)
                features = scalar.transform(features)

                if split == "train":
                    # there may be nan values in histograms
                    X_train = np.nan_to_num(np.concatenate([features, histograms, descriptors], axis=1))
                    Y_train = np.array(labels)
                elif split == "test":
                    # there may be nan values in histograms
                    X_test = np.nan_to_num(np.concatenate([features, histograms, descriptors], axis=1))
                    Y_test = np.array(labels)

            if len(np.unique(Y_train)) <= 1:
                print("Skip this task because there is only one class")
                continue

            train_stats = {}
            for label in np.unique(Y_train):
                train_stats[label] = np.sum(Y_train == label)
            test_stats = {}
            for label in np.unique(Y_test):
                test_stats[label] = np.sum(Y_test == label)
            # print("train stats:", train_stats)
            # print("test stats:", test_stats)

            lmnn = None
            if self.classifier == "KNN":
                model = KNeighborsClassifier(n_neighbors=5)
            elif self.classifier == "LR":
                # model = LogisticRegressionCV(cv=10, tol=0.0001, class_weight='balanced', random_state=42,
                #                                   multi_class='ovr', verbose=False)
                model = LogisticRegression()
            elif self.classifier == "LMNN":
                k_train, k_test, n_components, max_iter = 3, 3, X_train.shape[1], 180
                lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter, n_components=n_components)
                lmnn.fit(X_train, Y_train)
                X_train = lmnn.transform(X_train)
                X_test = lmnn.transform(X_test)

                model = KNeighborsClassifier(n_neighbors=k_test)

            model.fit(X_train, Y_train)
            Y_probs = model.predict_proba(X_test)

            # reshape
            Y_probs = Y_probs.reshape([len(test_objs), -1, len(np.unique(Y_train))])
            Y_test = Y_test.reshape([len(test_objs), -1])
            result = (description, Y_test.tolist(), Y_probs.tolist())
            results.append(result)

        # Important: This will only save the model of the last experiment
        if save_filename:
            with open(save_filename, "wb") as fh:
                pickle.dump([self.classifier, model, scalar, lmnn], fh)

        return results







