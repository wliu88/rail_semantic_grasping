import os
import glob
import pickle
from collections import OrderedDict
import numpy as np

import DataSpecification

np.random.seed(0)


class DataReader:
    """
    This class is used to read semantic object data and base features data from pickle files and organize them based on
    task and object classes. This class also structure training and testing data for different experiments.
    """

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labeled_data_dir = os.path.join(self.data_dir, "labeled")
        self.base_features_dir = os.path.join(self.data_dir, "base_features")

        # Important:
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
        self.data = []

        # Important:
        # this is used to group pointers of raw data based on object classes and tasks
        # The structure is task -> object class -> object id -> object state -> list of grasps
        self.grouped_data_indices = OrderedDict()
        for task in DataSpecification.TASKS:
            self.grouped_data_indices[task] = OrderedDict()
            for obj in DataSpecification.OBJECTS:
                self.grouped_data_indices[task][obj] = OrderedDict()

        #
        self.unique_id = {}

        self.read_data()

    def read_data(self):
        # grab all sessions in the labeled data dir
        session_dirs = glob.glob(os.path.join(self.labeled_data_dir, "*"))

        for session_dir in session_dirs:
            # find corresponding session folder in base features folder
            base_features_session_dir = os.path.join(self.base_features_dir, session_dir.split("/")[-1])

            object_files = glob.glob(os.path.join(session_dir, "*.pkl"))
            # iterate through objects
            for object_file in object_files:
                base_features_file = os.path.join(base_features_session_dir, object_file.split("/")[-1])

                # read base features
                base_features_list = load_pickle(base_features_file)

                # read object
                semantic_objects = load_pickle(object_file)
                semantic_object = semantic_objects.objects[0]

                # check
                assert len(semantic_object.labeled_grasps) == len(base_features_list)

                # extract features
                object_class = semantic_object.name
                object_id = len(self.grouped_data_indices[DataSpecification.TASKS[0]][object_class])

                self.unique_id[(object_class, object_id)] = object_file.split("/")[-1].split(".")[0]

                for grasp, base_features in zip(semantic_object.labeled_grasps, base_features_list):

                    # check grasp and base features are matched correctly
                    assert base_features.task == grasp.task
                    assert base_features.label == grasp.score

                    task = base_features.task
                    label = base_features.label

                    # Important: because we use the task field in semantic grasp msg definition to include both the task
                    #            and the object state. We need to get them back here.
                    task, object_state = task.split("_")

                    # extract context
                    context = (task, object_class, object_state)

                    # extract base features
                    features = []
                    features.append(base_features.object_spherical_resemblance)
                    features.append(base_features.object_cylindrical_resemblance)
                    features.extend(base_features.object_elongatedness)
                    features.append(base_features.object_volume)
                    features.extend(base_features.grasp_relative_position)
                    features.append(base_features.object_opening)
                    features.append(base_features.grasp_opening_angle)
                    features.append(base_features.grasp_opening_distance)
                    features.append(base_features.grasp_color_mean)
                    features.append(base_features.grasp_color_variance)
                    features.append(base_features.grasp_color_entropy)
                    histograms = []
                    histograms.extend(base_features.grasp_intensity_histogram)
                    histograms.extend(base_features.grasp_first_gradient_histogram)
                    histograms.extend(base_features.grasp_second_gradient_histogram)
                    histograms.extend(base_features.grasp_color_histogram)
                    descriptor = base_features.object_esf_descriptor
                    extracted_base_features = (features, histograms, descriptor)

                    # extract grasp semantic features
                    grasp_affordance = grasp.grasp_part_affordance
                    grasp_material = grasp.grasp_part_material
                    extracted_grasp_semantic_features = (grasp_affordance, grasp_material)

                    # extract object semantic parts
                    # extracted_object_semantic_parts are a tuple of variable length depending on the number of parts
                    extracted_object_semantic_parts = []
                    for part in semantic_object.parts:
                        part_affordance = part.affordance
                        part_material = part.material
                        extracted_object_semantic_parts.append((part_affordance, part_material))
                    extracted_object_semantic_parts = tuple(extracted_object_semantic_parts)

                    # add to data list and data indices
                    data_id = len(self.data)

                    # Important: here is the definition of each data point
                    self.data.append([label,
                                      context,
                                      extracted_base_features,
                                      extracted_grasp_semantic_features,
                                      extracted_object_semantic_parts])

                    if object_id not in self.grouped_data_indices[task][object_class]:
                        self.grouped_data_indices[task][object_class][object_id] = OrderedDict()
                    if object_state not in self.grouped_data_indices[task][object_class][object_id]:
                        self.grouped_data_indices[task][object_class][object_id][object_state] = []

                    self.grouped_data_indices[task][object_class][object_id][object_state].append(data_id)

        # # Important: We are going to modify labels here as we don't want to model negative preferences
        # for task in self.grouped_data_indices:
        #     for object_class in self.grouped_data_indices[task]:
        #         for object_id in self.grouped_data_indices[task][object_class]:
        #             for object_state in self.grouped_data_indices[task][object_class][object_id]:
        #                 assert len(self.grouped_data_indices[task][object_class][object_id][object_state]) == 20
        #
        #                 # compute number of unique labels
        #                 labels = []
        #                 for data_id in self.grouped_data_indices[task][object_class][object_id][object_state]:
        #                     labels.append(self.data[data_id][0])
        #                 labels = np.unique(labels)
        #
        #                 # change labels based on situations
        #                 if len(labels) == 1:
        #                     pass
        #                 elif 1 in labels and -1 in labels and 0 not in labels:
        #                     pass
        #                 elif 1 in labels and 0 in labels and -1 not in labels:
        #                     for data_id in self.grouped_data_indices[task][object_class][object_id][object_state]:
        #                         if self.data[data_id][0] == 0:
        #                             self.data[data_id][0] = -1
        #                 elif 0 in labels and -1 in labels and 1 not in labels:
        #                     for data_id in self.grouped_data_indices[task][object_class][object_id][object_state]:
        #                         if self.data[data_id][0] == 0:
        #                             self.data[data_id][0] = 1
        #                 elif 0 in labels and 1 in labels and -1 in labels:
        #                     for data_id in self.grouped_data_indices[task][object_class][object_id][object_state]:
        #                         if self.data[data_id][0] == 0:
        #                             self.data[data_id][0] = 1
        #                 else:
        #                     print("Error: unique labels are {}".format(labels))
        #                     exit()

        print(self.unique_id)

    def prepare_data_1(self, test_percentage=0.3, repeat_num=10):
        """
        This method is used to split data for experiment 1: semantic grasp transfer between object instances
        Instances of each object class for each task will be split into train and test

        Note: the output data should be a list of (description, train_objs, test_objs) tuples.
              Each tuple represent data for an object class.
              The algorithm needs to train a separate model and test the model for each tuple.

        :return:
        """
        # each experiment is a tuple of (description, train_objs, test_objs)
        experiments = []

        for task in self.grouped_data_indices:
            for object_class in self.grouped_data_indices[task]:
                print("Preparing split for task {} and object {}".format(task, object_class))

                num_instances = len(self.grouped_data_indices[task][object_class])
                object_instance_ids = np.array(range(num_instances))
                num_test = int(num_instances * test_percentage)
                print("Number of instances:", num_instances)
                print("Number of test instances:", num_test)
                if not num_test:
                    print("Not enough instance to test")
                    experiments.append(("{}:{}".format(task, object_class), (), ()))

                # Repeatedly create split
                for test_iter in range(repeat_num):
                    test_object_ids = np.random.choice(object_instance_ids, num_test, replace=False)
                    train_object_ids = np.array(list(set(object_instance_ids) - set(test_object_ids)))
                    # print(train_object_ids)

                    train_objs = []
                    test_objs = []
                    for object_id in train_object_ids:
                        for object_state in self.grouped_data_indices[task][object_class][object_id]:
                            train_objs.append([data_id for data_id in self.grouped_data_indices[task][object_class][object_id][object_state]])
                    for object_id in test_object_ids:
                        for object_state in self.grouped_data_indices[task][object_class][object_id]:
                            test_objs.append([data_id for data_id in self.grouped_data_indices[task][object_class][object_id][object_state]])

                    experiments.append(("{}:{}:{}".format(task, object_class, test_iter), train_objs, test_objs))

        for exp in experiments:
            print(exp)

        return experiments

    def prepare_data_2(self):
        """
        This method is used to split data for experiment 2: semantic grasp transfer between object classes
        Object classes for each task will be split into train and test

        :param test_percentage:
        :param repeat_num:
        :return:
        """
        # each experiment is a tuple of (description, train_objs, test_objs)
        experiments = []

        for ti, task in enumerate(self.grouped_data_indices):
            print("Preparing split for task {}".format(task))

            # perform leave one out test. test_object_class is the object class that is left out.
            for test_object_class in self.grouped_data_indices[task]:
                train_objs = []
                test_objs = []

                for object_id in self.grouped_data_indices[task][test_object_class]:
                    for object_state in self.grouped_data_indices[task][test_object_class][object_id]:
                        test_objs.append([data_id for data_id in self.grouped_data_indices[task][test_object_class][object_id][object_state]])

                for object_class in self.grouped_data_indices[task]:
                    if object_class == test_object_class:
                        continue
                    for object_id in self.grouped_data_indices[task][object_class]:
                        for object_state in self.grouped_data_indices[task][object_class][object_id]:
                            train_objs.append(
                                [data_id for data_id in self.grouped_data_indices[task][object_class][object_id][object_state]])

                experiments.append(("{}:{}".format(task, test_object_class), train_objs, test_objs))

        for exp in experiments:
            print(exp)

        return experiments

    def prepare_data_3(self):
        """
        This method is used to split data for experiment 3: semantic grasp transfer between tasks

        :return:
        """

        # each experiment is a tuple of (description, train_objs, test_objs)
        experiments = []

        for test_task in self.grouped_data_indices:
            for test_object_class in self.grouped_data_indices[test_task]:
                print("Preparing split for testing task {} object {}".format(test_task, test_object_class))
                train_objs = []
                test_objs = []

                for object_id in self.grouped_data_indices[test_task][test_object_class]:
                    for object_state in self.grouped_data_indices[test_task][test_object_class][object_id]:
                        test_objs.append([data_id for data_id in self.grouped_data_indices[test_task][test_object_class][object_id][object_state]])

                for task in self.grouped_data_indices:
                    for object_class in self.grouped_data_indices[task]:
                        if task == test_task and object_class == test_object_class:
                            continue

                        for object_id in self.grouped_data_indices[task][object_class]:
                            for object_state in self.grouped_data_indices[task][object_class][object_id]:
                                train_objs.append([data_id for data_id in
                                             self.grouped_data_indices[task][object_class][object_id][object_state]])

                experiments.append(("{}:{}".format(test_task, test_object_class), train_objs, test_objs))

        for exp in experiments:
            print(exp)

        return experiments

    def prepare_data_4(self, repeat_num=10, test_percentage=0.3):
        """
        This method is used to split data for experiment 4: semantic grasp transfer between

        :return:
        """

        # each experiment is a tuple of (description, train_objs, test_objs)
        experiments = []

        # repeatedly create split
        for test_iter in range(repeat_num):
            print("Preparing split for run {}".format(test_iter))
            train_objs = []
            test_objs = []

            # an instance in this case is the set of grasps for an object instance for a task
            instances = []
            for task in self.grouped_data_indices:
                for object_class in self.grouped_data_indices[task]:
                    for object_id in self.grouped_data_indices[task][object_class]:
                        instances.append(":".join([task, object_class, str(object_id)]))

            num_instances = len(instances)
            num_test = int(num_instances * test_percentage)
            # print("Number of instances:", num_instances)
            # print("Number of test instances:", num_test)
            if not num_test:
                print("Not enough instance to test")
                experiments.append(("{}".format(test_iter), (), ()))

            test_instances = list(np.random.choice(instances, num_test, replace=False))
            train_instances = list(set(instances) - set(test_instances))

            print("Test instances:")
            for instance in test_instances:
                task, object_class, object_id = instance.split(":")
                print(object_class, self.unique_id[(object_class, int(object_id))], task)
                object_id = int(object_id)
                for object_state in self.grouped_data_indices[task][object_class][object_id]:
                    test_objs.append(self.grouped_data_indices[task][object_class][object_id][object_state])

            for instance in train_instances:
                task, object_class, object_id = instance.split(":")
                object_id = int(object_id)
                for object_state in self.grouped_data_indices[task][object_class][object_id]:
                    train_objs.append(self.grouped_data_indices[task][object_class][object_id][object_state])

            experiments.append(("{}".format(test_iter), train_objs, test_objs))

        for exp in experiments:
            print(exp)

        return experiments

    def prepare_data_5(self, repeat_num=10, test_percentage=0.3):

        # each experiment is a tuple of (description, train_objs, test_objs)
        experiments = []

        # repeatedly create split
        for test_iter in range(repeat_num):
            print("Preparing split for run {}".format(test_iter))
            train_objs = []
            test_objs = []

            instances = []
            for task in self.grouped_data_indices:
                for object_class in self.grouped_data_indices[task]:
                    for object_id in self.grouped_data_indices[task][object_class]:
                        for object_state in self.grouped_data_indices[task][object_class][object_id]:
                            instances.append(":".join([task, object_class, str(object_id), object_state]))

            num_instances = len(instances)
            num_test = int(num_instances * test_percentage)
            # print("Number of instances:", num_instances)
            # print("Number of test instances:", num_test)
            if not num_test:
                print("Not enough instance to test")
                experiments.append(("{}".format(test_iter), (), ()))

            test_instances = list(np.random.choice(instances, num_test, replace=False))
            train_instances = list(set(instances) - set(test_instances))

            for instance in test_instances:
                task, object_class, object_id, object_state = instance.split(":")
                object_id = int(object_id)
                test_objs.append(self.grouped_data_indices[task][object_class][object_id][object_state])

            for instance in train_instances:
                task, object_class, object_id, object_state = instance.split(":")
                object_id = int(object_id)
                train_objs.append(self.grouped_data_indices[task][object_class][object_id][object_state])

            experiments.append(("{}".format(test_iter), train_objs, test_objs))

        for exp in experiments:
            print(exp)

        return experiments


def load_pickle(pickle_file):
    """
    Helper function for opening pickle saved in python2 in python3
    :param pickle_file:
    :return:
    """
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data





