import numpy as np

import main.DataSpecification as DataSpecification


class Batcher:

    def __init__(self, train_features, train_labels, test_features, test_labels, batch_size=20, shuffle=True):

        # each feature is a tuple of (task, object_class, state, parts, grasp)
        self.train_features = train_features
        self.train_labels = train_labels
        self.test_features = test_features
        self.test_labels = test_labels

        # vocabs
        self.affordance_to_idx = {"<PAD>": 0}
        self.material_to_idx = {"<PAD>": 0}
        self.task_to_idx = {}
        self.object_to_idx = {}
        self.state_to_idx = {}
        self.label_to_idx = {}

        self.build_vocabs()
        self.vectorize_data()
        # when the code is executed to here, all features and labels are in numpy array

    def build_vocabs(self):
        for aff in DataSpecification.AFFORDANCES:
            self.affordance_to_idx[aff] = len(self.affordance_to_idx)
        for mat in DataSpecification.MATERIALS:
            self.material_to_idx[mat] = len(self.material_to_idx)

        # misspell in data
        self.material_to_idx["platic"] = self.material_to_idx["plastic"]

        for task in DataSpecification.TASKS:
            self.task_to_idx[task] = len(self.task_to_idx)
        for obj in DataSpecification.OBJECTS:
            self.object_to_idx[obj] = len(self.object_to_idx)

        states = set()
        for obj in DataSpecification.STATES:
            states.update(DataSpecification.STATES[obj])
        for state in states:
            self.state_to_idx[state] = len(self.state_to_idx)

        for l in np.sort(np.unique(self.train_labels)):
            c = len(self.label_to_idx)
            self.label_to_idx[l] = c

    def vectorize_data(self):

        # first get the maximum number of parts
        max_num_parts = 0
        for feature in self.train_features + self.test_features:
            parts = feature[4]
            if len(parts) > max_num_parts:
                max_num_parts = len(parts)

        # vectorize features of each data point
        train_features = None
        test_features = None
        for features in [self.train_features, self.test_features]:
            vectorize_features = []

            for task, object_class, state, grasp, parts in features:
                parts_vec = []
                for aff, mat in parts:
                    parts_vec += [self.affordance_to_idx[aff], self.material_to_idx[mat]]
                # pad
                if len(parts) < max_num_parts:
                    parts_vec += [0, 0] * (max_num_parts - len(parts))

                feature_vec = [self.task_to_idx[task], self.object_to_idx[object_class], self.state_to_idx[state],
                               self.affordance_to_idx[grasp[0]], self.material_to_idx[grasp[1]]] + parts_vec

                vectorize_features.append(feature_vec)

            if features == self.train_features:
                train_features = np.array(vectorize_features)
            elif features == self.test_features:
                test_features = np.array(vectorize_features)

        self.train_features = train_features
        self.test_features = test_features

        # vectorize labels and convert labels (e.g., -1, 0, 1) to label indices (e.g., 1, 2, 3)
        train_labels = np.array(self.train_labels)
        test_labels = np.array(self.test_labels)

        for label in self.label_to_idx:
            train_labels[self.train_labels == label] = self.label_to_idx[label]
            test_labels[self.test_labels == label] = self.label_to_idx[label]

        self.train_labels = train_labels
        self.test_labels = test_labels

        print(self.train_features.shape)
        print(self.train_labels.shape)
        print(self.test_features.shape)
        print(self.test_labels.shape)

    def shuffle(self):
        pass

    def batch(self):
        pass

