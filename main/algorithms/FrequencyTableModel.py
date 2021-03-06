import numpy as np
from collections import defaultdict


class FrequencyTableModel:
    """
    This class implements the frequency-based method. For each experiment, the most frequently used affordance for
    grasping will be learned.

    """

    def __init__(self, use_affordance=True, use_material=False, use_context=True):
        assert use_affordance or use_material
        self.use_affordance = use_affordance
        self.use_material = use_material

        # which is task and object state
        self.use_context = use_context

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
                grasp_affordances = []
                grasp_materials = []
                contexts = []

                for obj in objs:
                    for id in obj:
                        label = data[id][0]
                        extracted_grasp_semantic_features = data[id][3]
                        grasp_affordances.append(extracted_grasp_semantic_features[0])
                        grasp_materials.append(extracted_grasp_semantic_features[1])

                        task = data[id][1][0]
                        state = data[id][1][2]
                        contexts.append((task, state))

                        labels.append(label)

                if split == "train":
                    if self.use_affordance and not self.use_material:
                        X_train = grasp_affordances
                    elif not self.use_affordance and self.use_material:
                        X_train = grasp_materials
                    X_train_contexts = contexts
                    Y_train = np.array(labels)
                elif split == "test":
                    if self.use_affordance and not self.use_material:
                        X_test = grasp_affordances
                    elif not self.use_affordance and self.use_material:
                        X_test = grasp_materials
                    X_test_contexts = contexts
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

            if self.use_context:
                # learn the model: frequency table
                freq_table = {}
                for i in range(len(X_train)):
                    semantic_grasp = X_train[i]
                    label = Y_train[i]
                    context = X_train_contexts[i]
                    if context not in freq_table:
                        freq_table[context] = defaultdict(list)
                    freq_table[context][semantic_grasp].append(label)

                # make prediction
                # Y_probs: [number of test grasps, number of label classes]
                Y_probs = np.zeros([len(Y_test), len(np.unique(Y_train))])
                for i in range(len(X_test)):
                    semantic_grasp = X_test[i]
                    context = X_train_contexts[i]
                    if context not in freq_table:
                        random_probs = np.random.random(len(np.unique(Y_train)))
                        random_probs = random_probs * 1.0 / random_probs.sum()
                        Y_probs[i, :] = random_probs
                    else:
                        observed_labels = freq_table[context][semantic_grasp]
                        # print("observed labels for {} are {}".format(semantic_grasp, freq_table[semantic_grasp]))
                        label_classes = np.sort(np.unique(observed_labels))
                        for li, label_class in enumerate(label_classes):
                            Y_probs[i, li] = np.sum(observed_labels == label_class) * 1.0 / len(observed_labels)

            else:
                # learn the model: frequency table
                freq_table = defaultdict(list)
                for i in range(len(X_train)):
                    semantic_grasp = X_train[i]
                    label = Y_train[i]
                    freq_table[semantic_grasp].append(label)

                # make prediction
                # Y_probs: [number of test grasps, number of label classes]
                Y_probs = np.zeros([len(Y_test), len(np.unique(Y_train))])
                for i in range(len(X_test)):
                    semantic_grasp = X_test[i]
                    observed_labels = freq_table[semantic_grasp]
                    # print("observed labels for {} are {}".format(semantic_grasp, freq_table[semantic_grasp]))
                    label_classes = np.sort(np.unique(observed_labels))
                    for li, label_class in enumerate(label_classes):
                        Y_probs[i, li] = np.sum(observed_labels == label_class) * 1.0 / len(observed_labels)


            # reshape
            Y_probs = Y_probs.reshape([len(test_objs), -1, len(np.unique(Y_train))])
            Y_test = Y_test.reshape([len(test_objs), -1])
            result = (description, Y_test.tolist(), Y_probs.tolist())
            results.append(result)

        return results




