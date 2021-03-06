import numpy as np
from sklearn.utils import shuffle

import torch

import main.DataSpecification as DataSpecification


class Batcher:

    def __init__(self, description, num_train_objects, num_test_objects,
                 train_features, train_labels, test_features, test_labels, batch_size=20, do_shuffle=True):

        self.description = description
        self.num_train_objects = num_train_objects
        self.num_test_objects = num_test_objects

        # each feature is a tuple of ((task, object_class, state, parts, grasp), base_features)
        self.raw_train_features = train_features
        self.raw_test_features = test_features
        self.raw_train_labels = train_labels
        self.raw_test_labels = test_labels

        self.train_semantic_features = None
        self.train_base_features = None
        self.train_labels = None
        self.test_semantic_features = None
        self.test_base_features = None
        self.test_labels = None

        # vocabs
        self.affordance_to_idx = {"<PAD>": 0}
        self.material_to_idx = {"<PAD>": 0}
        self.task_to_idx = {}
        self.object_to_idx = {}
        self.state_to_idx = {}
        self.label_to_idx = {}

        self.base_features_dim = None
        self.max_num_parts = 0

        # class weights for unbalanced data
        self.class_weights = []

        self.build_vocabs()
        self.vectorize_data()

        self.do_shuffle = do_shuffle
        self.batch_size = batch_size

        # batch count
        self.train_batch_count = 0
        self.test_batch_count = 0

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

        for l in np.sort(np.unique(self.raw_train_labels)):
            c = len(self.label_to_idx)
            self.label_to_idx[l] = c
            self.class_weights.append(1.0 / np.sum(self.raw_train_labels == l) * len(self.raw_train_labels))
        print("Class weights: {}".format(self.class_weights))

    def vectorize_data(self):

        # first get the maximum number of parts
        self.max_num_parts = 0
        for features in self.raw_train_features + self.raw_test_features:
            semantic_features = features[0]
            parts = semantic_features[4]
            if len(parts) > self.max_num_parts:
                self.max_num_parts = len(parts)

        # vectorize features of each data point
        for features in [self.raw_train_features, self.raw_test_features]:
            vectorize_semantic_features = []
            vectorize_base_features = []

            for semantic_features, base_features in features:
                if self.base_features_dim is None:
                    self.base_features_dim = len(base_features)

                task, object_class, state, grasp, parts = semantic_features

                parts_vec = []
                for aff, mat in parts:
                    parts_vec += [self.affordance_to_idx[aff], self.material_to_idx[mat]]
                # pad
                if len(parts) < self.max_num_parts:
                    parts_vec += [0, 0] * (self.max_num_parts - len(parts))

                feature_vec = [self.task_to_idx[task], self.object_to_idx[object_class], self.state_to_idx[state],
                               self.affordance_to_idx[grasp[0]], self.material_to_idx[grasp[1]]] + parts_vec

                vectorize_semantic_features.append(feature_vec)
                vectorize_base_features.append(base_features)

            if features == self.raw_train_features:
                self.train_semantic_features = np.array(vectorize_semantic_features)
                self.train_base_features = np.array(vectorize_base_features)
            elif features == self.raw_test_features:
                self.test_semantic_features = np.array(vectorize_semantic_features)
                self.test_base_features = np.array(vectorize_base_features)

        # vectorize labels and convert labels (e.g., -1, 0, 1) to label indices (e.g., 1, 2, 3)
        self.train_labels = np.array(self.raw_train_labels)
        self.test_labels = np.array(self.raw_test_labels)
        for label in self.label_to_idx:
            self.train_labels[self.raw_train_labels == label] = self.label_to_idx[label]
            self.test_labels[self.raw_test_labels == label] = self.label_to_idx[label]

    def batch_one_object(self, features):
        vectorize_semantic_features = []
        vectorize_base_features = []

        for semantic_features, base_features in features:
            task, object_class, state, grasp, parts = semantic_features

            parts_vec = []
            for aff, mat in parts:
                parts_vec += [self.affordance_to_idx[aff], self.material_to_idx[mat]]
            # pad
            if len(parts) < self.max_num_parts:
                parts_vec += [0, 0] * (self.max_num_parts - len(parts))

            feature_vec = [self.task_to_idx[task], self.object_to_idx[object_class], self.state_to_idx[state],
                           self.affordance_to_idx[grasp[0]], self.material_to_idx[grasp[1]]] + parts_vec

            vectorize_semantic_features.append(feature_vec)
            vectorize_base_features.append(base_features)

        all_semantic_features = np.array(vectorize_semantic_features)
        all_base_features = np.array(vectorize_base_features)

        return all_semantic_features, all_base_features

    def shuffle_data(self):
        shuffle(self.train_semantic_features, self.train_base_features, self.train_labels)

    def reset(self):
        self.train_batch_count = 0
        self.test_batch_count = 0
        if self.do_shuffle:
            self.shuffle_data()

    def get_train_batch(self):
        while self.train_batch_count < len(self.train_labels):
            if self.train_batch_count + self.batch_size <= len(self.train_labels):
                batch_semantic_features = self.train_semantic_features[self.train_batch_count:self.train_batch_count + self.batch_size]
                batch_base_features = self.train_base_features[self.train_batch_count:self.train_batch_count + self.batch_size]
                batch_labels = self.train_labels[self.train_batch_count:self.train_batch_count+self.batch_size]
            else:
                batch_semantic_features = self.train_semantic_features[self.train_batch_count:]
                batch_base_features = self.train_base_features[self.train_batch_count:]
                batch_labels = self.train_labels[self.train_batch_count:]

            self.train_batch_count += self.batch_size

            yield torch.LongTensor(batch_semantic_features), torch.FloatTensor(batch_base_features), \
                  torch.LongTensor(batch_labels)

    def get_test_batch(self):
        while self.test_batch_count < len(self.test_labels):
            if self.test_batch_count + self.batch_size <= len(self.test_labels):
                batch_semantic_features = self.test_semantic_features[self.test_batch_count:self.test_batch_count + self.batch_size]
                batch_base_features = self.test_base_features[self.test_batch_count:self.test_batch_count + self.batch_size]
                batch_labels = self.test_labels[self.test_batch_count:self.test_batch_count + self.batch_size]
            else:
                batch_semantic_features = self.test_semantic_features[self.test_batch_count:]
                batch_base_features = self.test_base_features[self.test_batch_count:]
                batch_labels = self.test_labels[self.test_batch_count:]

            self.test_batch_count += self.batch_size

            yield torch.LongTensor(batch_semantic_features), torch.FloatTensor(batch_base_features), \
                  torch.LongTensor(batch_labels)

    def get_affordance_vocab_size(self):
        return len(self.affordance_to_idx)

    def get_material_vocab_size(self):
        return len(self.material_to_idx)

    def get_task_vocab_size(self):
        return len(self.task_to_idx)

    def get_object_vocab_size(self):
        return len(self.object_to_idx)

    def get_state_vocab_size(self):
        return len(self.state_to_idx)

    def get_label_dim(self):
        return len(self.label_to_idx)

    def get_base_features_dim(self):
        return self.base_features_dim

    def get_class_weights(self):
        return torch.FloatTensor(self.class_weights)

    def get_raw_labels(self):
        return np.copy(self.raw_train_labels), np.copy(self.raw_test_labels)




