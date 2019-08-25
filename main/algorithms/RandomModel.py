import numpy as np
from collections import defaultdict


class RandomModel:
    """
    This class predicts the label of grasps randomly.

    """

    def __init__(self, use_affordance=True, use_material=False):
        assert use_affordance or use_material
        self.use_affordance = use_affordance
        self.use_material = use_material

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

                for obj in objs:
                    for id in obj:
                        labels.append(data[id][0])

                if split == "train":
                    Y_train = np.array(labels)
                elif split == "test":
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

            # make prediction
            # Y_probs: [number of test grasps, number of label classes]
            Y_probs = np.zeros([len(Y_test), len(np.unique(Y_train))])
            for i in range(len(Y_test)):
                random_probs = np.random.random_sample(len(np.unique(Y_train)))
                random_probs = random_probs / sum(random_probs)
                Y_probs[i, :] = random_probs

            # reshape
            Y_probs = Y_probs.reshape([len(test_objs), -1, len(np.unique(Y_train))])
            Y_test = Y_test.reshape([len(test_objs), -1])

            result = (description, Y_test.tolist(), Y_probs.tolist())
            results.append(result)

        return results