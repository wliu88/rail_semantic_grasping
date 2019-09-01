import numpy as np
from collections import defaultdict
import time

from sklearn.utils import shuffle

import torch
import torch.optim as optim

from main.algorithms.Model2.CAGEModel import CAGEModel
from main.algorithms.Model2.Batcher import Batcher

# Important: should assign weights to different classes since the data is inbalanced


class CAGEAlgorithm:
    """
    This class implements the CAGE algorithm

    """

    def __init__(self, number_of_epochs=300):
        self.model = None
        self.number_of_epochs = number_of_epochs

    def run_experiments(self, data, experiments):
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

            # organize training and testing data
            for split in ["train", "test"]:

                if split == "train":
                    objs = train_objs
                elif split == "test":
                    objs = test_objs

                labels = []
                features = []

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
                        extracted_grasp_semantic_features = data[id][3]
                        grasp = extracted_grasp_semantic_features
                        task = data[id][1][0]
                        object_class = data[id][1][1]
                        state = data[id][1][2]
                        parts = data[id][4]

                        # important the order is changed from model 1
                        features.append((task, object_class, state, grasp, parts))

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
            batcher = Batcher(X_train, Y_train, X_test, Y_test, batch_size=256, do_shuffle=True)

            # learn the model
            self.train(batcher)

            # make prediction
            Y_probs = self.test(batcher)

            # reshape
            Y_probs = Y_probs.reshape([len(test_objs), -1, len(np.unique(Y_train))])
            Y_test = Y_test.reshape([len(test_objs), -1])
            result = (description, Y_test.tolist(), Y_probs.tolist())
            results.append(result)

        return results

    def train(self, batcher):

        self.model = CAGEModel(affordance_vocab_size=batcher.get_affordance_vocab_size(),
                               material_vocab_size=batcher.get_material_vocab_size(),
                               task_vocab_size=batcher.get_task_vocab_size(),
                               object_vocab_size=batcher.get_object_vocab_size(),
                               state_vocab_size=batcher.get_state_vocab_size(),
                               affordance_embedding_dim=5, material_embedding_dim=5,
                               task_embedding_dim=5, object_embedding_dim=5, state_embedding_dim=5,
                               part_encoder_dim=5, object_encoder_dim=5, grasp_encoder_dim=5,
                               part_pooling_method="max",
                               label_dim=batcher.get_label_dim())


        optimizer = optim.Adagrad(self.model.parameters(), lr=1e-1)
        # criterion = torch.nn.NLLLoss(weight=batcher.get_class_weights())#.cuda()
        criterion = torch.nn.NLLLoss(weight=batcher.get_class_weights())

        # 1. training process
        for epoch in range(self.number_of_epochs):
            # self.scheduler.step()
            total_loss = 0
            start = time.time()

            batcher.reset()
            for batch_features, batch_labels in batcher.get_train_batch():
                self.model.train()
                self.model.zero_grad()
                log_probs = self.model(batch_features)
                loss = criterion(log_probs, batch_labels)

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print("Epoch", epoch, "spent", time.time() - start, "with total loss:", total_loss)

    def test(self, batcher):

        all_probs = None

        with torch.no_grad():
            self.model.eval()
            batcher.reset()
            for batch_features, batch_labels in batcher.get_test_batch():
                log_probs = self.model(batch_features)
                probs = torch.exp(log_probs).clone().cpu().data.numpy()
                if all_probs is None:
                    all_probs = probs
                else:
                    all_probs = np.vstack([all_probs, probs])

        return all_probs