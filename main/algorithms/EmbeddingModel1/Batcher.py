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
        self.object_to_idx = {"<PAD>": 0}
        self.task_to_idx = {"<PAD>": 0}
        self.grasp_to_idx = {"<PAD>": 0}
        self.label_to_idx = {}

        # class weights for unbalanced data
        self.class_weights = []

        self.build_vocabs()
        self.vectorize_data()
        # when the code is executed to here, all features and labels are in numpy array

        self.do_shuffle = do_shuffle
        self.batch_size = batch_size

        # batch count
        self.train_batch_count = 0
        self.test_batch_count = 0

    def build_vocabs(self):

        train_feats = {"objects": set(), "tasks": set(), "grasps": set()}
        test_feats = {"objects": set(), "tasks": set(), "grasps": set()}

        for features in [self.train_features, self.test_features]:

            for task, object_class, state, grasp, parts in features:
                parts_vec = []
                for aff, mat in parts:
                    # misspell in data
                    if mat == "platic":
                        mat = "plastic"
                    parts_vec += [aff, mat]

                object_descript = tuple([object_class, state] + parts_vec)

                if object_descript not in self.object_to_idx:
                    self.object_to_idx[object_descript] = len(self.object_to_idx)

                if task not in self.task_to_idx:
                    self.task_to_idx[task] = len(self.task_to_idx)

                if grasp not in self.grasp_to_idx:
                    self.grasp_to_idx[grasp] = len(self.grasp_to_idx)

                if features == self.train_features:
                    train_feats["objects"].add(object_descript)
                    train_feats["tasks"].add(task)
                    train_feats["grasps"].add(grasp)
                elif features == self.test_features:
                    test_feats["objects"].add(object_descript)
                    test_feats["tasks"].add(task)
                    test_feats["grasps"].add(grasp)

        # find out-of-vocabulary
        new_objects_in_test = test_feats["objects"] - train_feats["objects"]
        new_tasks_in_test = test_feats["tasks"] - train_feats["tasks"]
        new_grasps_in_test = test_feats["grasps"] - train_feats["grasps"]
        print("{} new object in test: {}".format(len(new_objects_in_test)*1.0/len(test_feats["objects"]),
                                                 new_objects_in_test))
        print("{} new tasks in test: {}".format(len(new_tasks_in_test) * 1.0 / len(test_feats["tasks"]),
                                                 new_tasks_in_test))
        print("{} new grasps in test: {}".format(len(new_grasps_in_test) * 1.0 / len(test_feats["grasps"]),
                                                 new_grasps_in_test))

        for l in np.sort(np.unique(self.train_labels)):
            if l == 1:
                c = 1
            else:
                c = -1
            self.label_to_idx[l] = c
            self.class_weights.append(1.0 / np.sum(self.train_labels == l) * len(self.train_labels))
        print("Class weights: {}".format(self.class_weights))

    def vectorize_data(self):

        # vectorize features of each data point
        train_features = None
        test_features = None
        for features in [self.train_features, self.test_features]:
            vectorize_features = []

            for task, object_class, state, grasp, parts in features:
                parts_vec = []
                for aff, mat in parts:
                    # misspell in data
                    if mat == "platic":
                        mat = "plastic"
                    parts_vec += [aff, mat]

                object_descript = tuple([object_class, state] + parts_vec)

                vectorize_features.append([self.object_to_idx[object_descript], self.task_to_idx[task], self.grasp_to_idx[grasp]])

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

    def shuffle_data(self):
        shuffle(self.train_features, self.train_labels)

    def reset(self):
        self.train_batch_count = 0
        self.test_batch_count = 0
        if self.do_shuffle:
            self.shuffle_data()

    def get_train_batch(self):
        while self.train_batch_count < len(self.train_labels):
            if self.train_batch_count + self.batch_size <= len(self.train_labels):
                batch_features = self.train_features[self.train_batch_count:self.train_batch_count+self.batch_size]
                batch_labels = self.train_labels[self.train_batch_count:self.train_batch_count+self.batch_size]
            else:
                batch_features = self.train_features[self.train_batch_count:]
                batch_labels = self.train_labels[self.train_batch_count:]

            self.train_batch_count += self.batch_size

            yield torch.LongTensor(batch_features), torch.FloatTensor(batch_labels)
        # else:
        #     return None, None

    def get_test_batch(self):
        while self.test_batch_count < len(self.test_labels):
            if self.test_batch_count + self.batch_size <= len(self.test_labels):
                batch_features = self.test_features[self.test_batch_count:self.test_batch_count + self.batch_size]
                batch_labels = self.test_labels[self.test_batch_count:self.test_batch_count + self.batch_size]
            else:
                batch_features = self.test_features[self.test_batch_count:]
                batch_labels = self.test_labels[self.test_batch_count:]

            self.test_batch_count += self.batch_size

            yield torch.LongTensor(batch_features), torch.FloatTensor(batch_labels)
        # else:
        #     yield None, None

    def get_task_vocab_size(self):
        return len(self.task_to_idx)

    def get_object_vocab_size(self):
        return len(self.object_to_idx)

    def get_grasp_vocab_size(self):
        return len(self.grasp_to_idx)

    def get_label_dim(self):
        return len(self.label_to_idx)

    def get_class_weights(self):
        return torch.FloatTensor(self.class_weights)




