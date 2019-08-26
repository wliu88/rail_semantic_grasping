import numpy as np
from collections import defaultdict
import time

from sklearn.utils import shuffle

import torch
import torch.optim as optim

from main.algorithms.Model1.CAGEModel import CAGEModel

# Important: should assign weights to different classes since the data is inbalanced

class CAGEAlgorithm:
    """
    This class implements the CAGE algorithm

    """

    def __init__(self, number_of_epochs=20):
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

                # task, object_class, state, parts, grasp

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

                        features.append((task, object_class, state, parts, grasp))

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

            label_to_class = {}
            for l in np.sort(np.unique(Y_train)):
                c = len(label_to_class)
                label_to_class[l] = c

            # learn the model
            self.train(X_train, Y_train, label_to_class)

            # make prediction
            # Y_probs: [number of test grasps, number of label classes]
            Y_probs = np.zeros([len(Y_test), len(np.unique(Y_train))])
            self.test(X_test, Y_probs)

            # reshape
            Y_probs = Y_probs.reshape([len(test_objs), -1, len(np.unique(Y_train))])
            Y_test = Y_test.reshape([len(test_objs), -1])
            result = (description, Y_test.tolist(), Y_probs.tolist())
            results.append(result)

        return results

    def train(self, X, Y, label_to_class):

        self.model = CAGEModel(affordance_embedding_dim=5, material_embedding_dim=5,
                               task_embedding_dim=5, object_embedding_dim=5, state_embedding_dim=5,
                               part_encoder_dim=5, object_encoder_dim=5, grasp_encoder_dim=5,
                               part_pooling_method="max",
                               label_dim=len(np.unique(Y)))

        optimizer = optim.Adagrad(self.model.parameters(), lr=1e-1)
        criterion = torch.nn.NLLLoss()#.cuda()

        # 1. training process
        for epoch in range(self.number_of_epochs):
            # self.scheduler.step()
            total_loss = 0
            start = time.time()

            batch_count = 0
            batch_loss = 0

            X, Y = shuffle(X, Y)

            for x, label in zip(X, Y):
                self.model.train()
                self.model.zero_grad()
                task, object_class, state, parts, grasp = x
                log_probs = self.model(task, object_class, state, parts, grasp)
                loss = criterion(log_probs.view(1, -1), torch.tensor([label_to_class[label]], dtype=torch.long))#.cuda())

                # debug: is this right? is this stochastic gradient descent?
                batch_loss += loss
                batch_count += 1

                # optimizer.step()
                # total_loss += loss.item()

                if batch_count == 10:
                    avg_loss = batch_loss / 10
                    optimizer.zero_grad()
                    avg_loss.backward()
                    optimizer.step()
                    total_loss += avg_loss.item()
                    batch_count = 0
                    batch_loss = 0

            print("Epoch", epoch, "spent", time.time() - start, "with total loss:", total_loss)

    def test(self, X, Y_probs):

        with torch.no_grad():
            self.model.eval()
            for i, x in enumerate(X):
                task, object_class, state, parts, grasp = x
                log_probs = self.model(task, object_class, state, parts, grasp)
                probs = torch.exp(log_probs).clone().cpu().data.numpy()
                Y_probs[i, :] = probs

        return Y_probs