import numpy as np
from sklearn.utils import shuffle

import torch

import main.DataSpecification as DataSpecification


class Batcher:

    def __init__(self, train_features, train_labels, test_features, test_labels, batch_size=20, do_shuffle=True):

        # before vectorization, each feature is a tuple of (task, object_class, state, parts, grasp)
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

        # class weights for unbalanced data
        self.class_weights = []

        # self.build_vocabs()
        self.vectorize_data()
        # when the code is executed to here, all features and labels are in numpy array

        self.do_shuffle = do_shuffle
        self.batch_size = batch_size

        # batch count
        self.train_batch_count = 0
        self.test_batch_count = 0

    # def build_vocabs(self):
    #     for aff in DataSpecification.AFFORDANCES:
    #         self.affordance_to_idx[aff] = len(self.affordance_to_idx)
    #     for mat in DataSpecification.MATERIALS:
    #         self.material_to_idx[mat] = len(self.material_to_idx)
    #
    #     # misspell in data
    #     self.material_to_idx["platic"] = self.material_to_idx["plastic"]
    #
    #     for task in DataSpecification.TASKS:
    #         self.task_to_idx[task] = len(self.task_to_idx)
    #     for obj in DataSpecification.OBJECTS:
    #         self.object_to_idx[obj] = len(self.object_to_idx)
    #
    #     states = set()
    #     for obj in DataSpecification.STATES:
    #         states.update(DataSpecification.STATES[obj])
    #     for state in states:
    #         self.state_to_idx[state] = len(self.state_to_idx)
    #
    #     for l in np.sort(np.unique(self.train_labels)):
    #         c = len(self.label_to_idx)
    #         self.label_to_idx[l] = c
    #         self.class_weights.append(1.0 / np.sum(self.train_labels == l) * len(self.train_labels))
    #     print("Class weights: {}".format(self.class_weights))

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

        train_object_combos = set()
        train_grasp_combos = set()
        train_task_combos = set()
        train_state_combos = set()

        test_object_combos = set()
        test_grasp_combos = set()
        test_task_combos = set()
        test_state_combos = set()

        for features in [self.train_features, self.test_features]:
            vectorize_features = []

            for task, object_class, state, grasp, parts in features:
                parts_vec = []
                for aff, mat in parts:
                    parts_vec += [aff, mat]

                object_vec = tuple([object_class, state] + parts_vec)
                task_vec = task
                grasp_vec = grasp

                if features == self.train_features:
                    train_object_combos.add(object_vec)
                    train_grasp_combos.add(grasp_vec)
                    train_task_combos.add(task_vec)
                    train_state_combos.add(state)
                elif features == self.test_features:
                    test_object_combos.add(object_vec)
                    test_grasp_combos.add(grasp_vec)
                    test_task_combos.add(task_vec)
                    test_state_combos.add(state)

        new_object_ratio = len(test_object_combos - train_object_combos) * 1.0 / len(test_object_combos)
        new_task_ratio = len(test_task_combos - train_task_combos) * 1.0 / len(test_task_combos)
        new_grasp_ratio = len(test_grasp_combos - train_grasp_combos) * 1.0 / len(test_grasp_combos)
        new_state_ratio = len(test_state_combos - train_state_combos) * 1.0 / len(test_state_combos)

        print("New object in test:", test_object_combos - train_object_combos)
        print(new_object_ratio)
        print("New task in test:", test_task_combos - train_task_combos)
        print(new_task_ratio)
        print("New grasp in test:", test_task_combos - train_task_combos)
        print(new_grasp_ratio)
        print("New state in test:", test_state_combos - train_state_combos)
        print(new_state_ratio)

        #     if features == self.train_features:
        #         train_features = np.array(vectorize_features)
        #     elif features == self.test_features:
        #         test_features = np.array(vectorize_features)
        #
        # self.train_features = train_features
        # self.test_features = test_features
        #
        # # vectorize labels and convert labels (e.g., -1, 0, 1) to label indices (e.g., 1, 2, 3)
        # train_labels = np.array(self.train_labels)
        # test_labels = np.array(self.test_labels)
        #
        # for label in self.label_to_idx:
        #     train_labels[self.train_labels == label] = self.label_to_idx[label]
        #     test_labels[self.test_labels == label] = self.label_to_idx[label]
        #
        # self.train_labels = train_labels
        # self.test_labels = test_labels

    # def shuffle_data(self):
    #     shuffle(self.train_features, self.train_labels)
    #
    # def reset(self):
    #     self.train_batch_count = 0
    #     self.test_batch_count = 0
    #     if self.do_shuffle:
    #         self.shuffle_data()
    #
    # def get_train_batch(self):
    #     while self.train_batch_count < len(self.train_labels):
    #         if self.train_batch_count + self.batch_size <= len(self.train_labels):
    #             batch_features = self.train_features[self.train_batch_count:self.train_batch_count+self.batch_size]
    #             batch_labels = self.train_labels[self.train_batch_count:self.train_batch_count+self.batch_size]
    #         else:
    #             batch_features = self.train_features[self.train_batch_count:]
    #             batch_labels = self.train_labels[self.train_batch_count:]
    #
    #         self.train_batch_count += self.batch_size
    #
    #         yield torch.LongTensor(batch_features), torch.LongTensor(batch_labels)
    #     # else:
    #     #     return None, None
    #
    # def get_test_batch(self):
    #     while self.test_batch_count < len(self.test_labels):
    #         if self.test_batch_count + self.batch_size <= len(self.test_labels):
    #             batch_features = self.test_features[self.test_batch_count:self.test_batch_count + self.batch_size]
    #             batch_labels = self.test_labels[self.test_batch_count:self.test_batch_count + self.batch_size]
    #         else:
    #             batch_features = self.test_features[self.test_batch_count:]
    #             batch_labels = self.test_labels[self.test_batch_count:]
    #
    #         self.test_batch_count += self.batch_size
    #
    #         yield torch.LongTensor(batch_features), torch.LongTensor(batch_labels)
    #     # else:
    #     #     yield None, None
    #
    # def get_affordance_vocab_size(self):
    #     return len(self.affordance_to_idx)
    #
    # def get_material_vocab_size(self):
    #     return len(self.material_to_idx)
    #
    # def get_task_vocab_size(self):
    #     return len(self.task_to_idx)
    #
    # def get_object_vocab_size(self):
    #     return len(self.object_to_idx)
    #
    # def get_state_vocab_size(self):
    #     return len(self.state_to_idx)
    #
    # def get_label_dim(self):
    #     return len(self.label_to_idx)
    #
    # def get_class_weights(self):
    #     return torch.FloatTensor(self.class_weights)




