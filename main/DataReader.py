import os
import glob
import pickle
from collections import OrderedDict
import numpy as np

from rail_semantic_grasping.msg import SemanticObjectList, SemanticObject, SemanticGrasp, BaseFeatures, SemanticPart
import DataSpecification


# ToDo: Debug!!!!!

class DataReader:

    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.labeled_data_dir = os.path.join(self.data_dir, "labeled")
        self.base_features_dir = os.path.join(self.data_dir, "base_features")

        # each data point has form:
        # ((features, histograms, descriptor),(grasp_affordance, grasp_material),((part_affordance, part_material)...))
        self.data = []

        # this is used to group pointers of raw data based on object classes and tasks
        self.grouped_data_indices = OrderedDict()
        for task in DataSpecification.TASKS:
            self.grouped_data_indices[task] = OrderedDict()
            for obj in DataSpecification.OBJECTS:
                self.grouped_data_indices[task][obj] = OrderedDict()

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

                # read base features (there is one set of base features for each grasp)
                base_features_list = load_pickle(base_features_file)

                # read object
                semantic_objects = load_pickle(object_file)
                semantic_object = semantic_objects.objects[0]

                # check
                assert len(semantic_object.labeled_grasps) == len(base_features_list)

                # extract features
                object_class = semantic_object.name
                object_id = len(self.grouped_data_indices[DataSpecification.TASKS[0]][object_class])
                for grasp, base_features in zip(semantic_object.labeled_grasps, base_features_list):
                    # check grasp and base features are matched correctly
                    assert base_features.task == grasp.task
                    assert base_features.label == grasp.score

                    task = base_features.task
                    label = base_features.label

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
                    extracted_object_semantic_parts = []
                    for part in semantic_object.parts:
                        part_affordance = part.affordance
                        part_material = part.material
                        extracted_object_semantic_parts.append((part_affordance, part_material))
                    extracted_object_semantic_parts = tuple(extracted_object_semantic_parts)

                    # add to data list and data indices
                    data_id = len(self.data)
                    self.data.append((label,
                                      extracted_base_features,
                                      extracted_grasp_semantic_features,
                                      extracted_object_semantic_parts))
                    if object_id not in self.grouped_data_indices[task][object_class]:
                        self.grouped_data_indices[task][object_class][object_id] = []
                    self.grouped_data_indices[task][object_class][object_id].append(data_id)

    def prepare_data_1(self, test_percentage=0.3, repeat_num=10):
        """
        This method is used to split data for experiment 1
        Instances of each object class for each task will be split into train and test

        Note: the output data should be a list of (train, test) tuples. Each tuple represent data for an object class.
              The algorithm needs to train a seperate model and test the model for each tuple.

        :return:
        """

        # each experiment is a tuple of (description, train_ids, test_ids)
        experiments = []

        for ti, task in enumerate(self.grouped_data_indices):
            for oi, object_class in enumerate(self.grouped_data_indices[task]):
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

                    train_ids = []
                    test_ids = []
                    for object_id in train_object_ids:
                        train_ids.extend([data_id for data_id in self.grouped_data_indices[task][object_class][object_id]])
                    for object_id in test_object_ids:
                        test_ids.extend([data_id for data_id in self.grouped_data_indices[task][object_class][object_id]])

                    experiments.append(("{}:{}:{}".format(task, object_class, test_iter), train_ids, test_ids))

        for exp in experiments:
            print(exp)


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


if __name__ == "__main__":
    data_reader = DataReader("/home/weiyu/catkin_ws/src/rail_semantic_grasping/data")
    data_reader.prepare_data_1()






