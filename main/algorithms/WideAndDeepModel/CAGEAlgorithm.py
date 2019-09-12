import numpy as np
from collections import defaultdict
import time
import pickle

from sklearn.utils import shuffle
from sklearn import preprocessing

import torch
import torch.optim as optim

from main.algorithms.WideAndDeepModel.CAGEModel import CAGEModel
from main.algorithms.WideAndDeepModel.Batcher import Batcher
from main.Metrics import compute_aps
from main.algorithms.Logger import Logger


class CAGEAlgorithm:
    """
    This class implements the CAGE algorithm
    """

    def __init__(self, number_of_epochs=150):
        self.model = None
        self.number_of_epochs = number_of_epochs

        self.logger = Logger()

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
            print("\nRun experiment {}".format(description))

            # normalize features
            scalar = preprocessing.StandardScaler()

            # organize training and testing data
            for split in ["train", "test"]:

                if split == "train":
                    objs = train_objs
                elif split == "test":
                    objs = test_objs

                labels = []
                features = []
                semantic_features = []
                visual_features = []
                histograms = []
                descriptors = []

                # each data point has form:
                # (label, context, extracted_base_features, extracted_grasp_semantic_features, extracted_object_semantic_parts)
                #
                # Specifically,
                # (
                #  label,
                #  (task, object_class, object_state),
                #  (features, histograms, descriptor),
                #  (grasp_affordance, grasp_material),
                #  ((part_affordance, part_material), (part_affordance, part_material), ...)
                # )

                for obj in objs:
                    for id in obj:
                        label = data[id][0]
                        labels.append(label)

                        # semantic features: categorical
                        extracted_grasp_semantic_features = data[id][3]
                        grasp = extracted_grasp_semantic_features
                        task = data[id][1][0]
                        object_class = data[id][1][1]
                        state = data[id][1][2]
                        parts = data[id][4]

                        # visual features: continuous
                        extracted_base_features = data[id][2]
                        visual_features.append(extracted_base_features[0])
                        histograms.append(extracted_base_features[1])
                        descriptors.append(extracted_base_features[2])

                        # important the order is changed from model 1
                        semantic_features.append((task, object_class, state, grasp, parts))

                # vectorize base features
                visual_features = np.array(visual_features)
                histograms = np.array(histograms)
                descriptors = np.array(descriptors)
                # preprocess features
                if split == "train":
                    scalar.fit(visual_features)
                visual_features = scalar.transform(visual_features)
                # there may be nan values in histograms
                np.nan_to_num(histograms)
                # concatenate
                base_features = np.nan_to_num(np.concatenate([visual_features, histograms, descriptors], axis=1))
                # base_features = np.nan_to_num(visual_features)
                base_features = base_features.tolist()

                # combine semantic and visual features
                assert len(base_features) == len(semantic_features)
                for i in range(len(semantic_features)):
                    features.append((semantic_features[i], base_features[i]))

                if split == "train":
                    X_train = features
                    Y_train = np.array(labels)
                elif split == "test":
                    X_test = features
                    Y_test = np.array(labels)

            if len(np.unique(Y_train)) <= 1:
                print("Skip this task because there is only one class")
                continue

            # calculate data stats
            train_stats = {}
            for label in np.unique(Y_train):
                train_stats[label] = np.sum(Y_train == label)
            test_stats = {}
            for label in np.unique(Y_test):
                test_stats[label] = np.sum(Y_test == label)
            print("train stats:", train_stats)
            print("test stats:", test_stats)

            # preparation for training the model
            batcher = Batcher(description, len(train_objs), len(test_objs),
                              X_train, Y_train, X_test, Y_test, batch_size=512, do_shuffle=True)

            # learn the model
            self.train(batcher)

            # make prediction
            Y_probs = self.test(batcher)

            # reshape
            Y_probs = Y_probs.reshape([len(test_objs), -1, len(np.unique(Y_train))])
            Y_test = Y_test.reshape([len(test_objs), -1])
            result = (description, Y_test.tolist(), Y_probs.tolist())
            results.append(result)

        # Important: This will only save the model of the last experiment
        if save_filename:
            with open(save_filename, "wb") as fh:
                pickle.dump(["wide_and_deep", self.model, batcher, scalar], fh)

        self.logger.close()

        return results

    def train(self, batcher):

        self.model = CAGEModel(affordance_vocab_size=batcher.get_affordance_vocab_size(),
                               material_vocab_size=batcher.get_material_vocab_size(),
                               task_vocab_size=batcher.get_task_vocab_size(),
                               object_vocab_size=batcher.get_object_vocab_size(),
                               state_vocab_size=batcher.get_state_vocab_size(),
                               affordance_embedding_dim=10, material_embedding_dim=10,
                               task_embedding_dim=10, object_embedding_dim=10, state_embedding_dim=10,
                               base_features_dim=batcher.get_base_features_dim(),
                               part_encoder_dim=10, object_encoder_dim=10, grasp_encoder_dim=10,
                               part_pooling_method="avg",
                               label_dim=batcher.get_label_dim(),
                               use_wide=True, use_deep_base_features=False, use_deep_semantic_features=True)

        # Separate optimizers for wide and deep parts
        # wide_params, deep_params = self.model.get_params()
        # optimizer_deep = optim.Adagrad(deep_params, lr=1e-1, weight_decay=1e-5)
        # optimizer_wide = optim.Adagrad(wide_params, lr=1e-1)

        # Important: Current best settings: optim.Adagrad(self.model.parameters(), lr=1e-1, weight_decay=1e-5) with 150 epochs
        optimizer = optim.Adagrad(self.model.parameters(), lr=1e-1, weight_decay=1e-5)

        # Important: The weight is not helping that much
        # criterion = torch.nn.NLLLoss(weight=batcher.get_class_weights())#.cuda()
        criterion = torch.nn.NLLLoss()

        # 1. training process
        for epoch in range(self.number_of_epochs):
            # self.scheduler.step()
            total_loss = 0
            start = time.time()

            batcher.reset()
            for batch_semantic_features, batch_base_features, batch_labels in batcher.get_train_batch():

                self.model.train()
                self.model.zero_grad()
                log_probs = self.model(batch_semantic_features, batch_base_features)
                loss = criterion(log_probs, batch_labels)

                loss.backward()
                # optimizer_deep.step()
                # optimizer_wide.step()
                optimizer.step()

                total_loss += loss.item()

            print("Epoch", epoch, "spent", time.time() - start, "with total loss:", total_loss)

            # total_test_loss = 0
            # with torch.no_grad():
            #     self.model.eval()
            #     batcher.reset()
            #     for batch_semantic_features, batch_base_features, batch_labels in batcher.get_test_batch():
            #         log_probs = self.model(batch_semantic_features, batch_base_features)
            #         loss = criterion(log_probs, batch_labels)
            #         total_test_loss += loss.item()
            #
            # self.logger.log_loss(total_loss, total_test_loss, epoch, batcher.description)

    def test(self, batcher):
        all_probs = None
        with torch.no_grad():
            self.model.eval()
            batcher.reset()
            for batch_semantic_features, batch_base_features, batch_labels in batcher.get_test_batch():
                log_probs = self.model(batch_semantic_features, batch_base_features)
                probs = torch.exp(log_probs).clone().cpu().data.numpy()
                if all_probs is None:
                    all_probs = probs
                else:
                    all_probs = np.vstack([all_probs, probs])
        return all_probs