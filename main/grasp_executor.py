import os
import glob
import pickle
# from collections import OrderedDict
import numpy as np
import pickle

import torch
np.random.seed(0)


class GraspExecutor:
    """
    This class is used to read semantic object data and base features data from pickle files and organize them based on
    task and object classes. This class also structure training and testing data for different experiments.
    """

    def __init__(self, data_dir):

        self.data_dir = data_dir
        self.unlabeled_data_dir = os.path.join(self.data_dir, "test_execution")
        # self.base_features_dir = os.path.join(self.data_dir, "base_features")

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

        self.read_data()

    def read_data(self):
        # grab all sessions in the labeled data dir
        session_dirs = glob.glob(os.path.join(self.unlabeled_data_dir, "*"))

        for session_dir in session_dirs:
            # find corresponding session folder in base features folder
            # base_features_session_dir = os.path.join(self.base_features_dir, session_dir.split("/")[-1])

            object_files = glob.glob(os.path.join(session_dir, "*.pkl"))
            # iterate through objects
            for object_file in object_files:
                # base_features_file = os.path.join(base_features_session_dir, object_file.split("/")[-1])

                # read base features
                # base_features_list = load_pickle(base_features_file)

                # read object
                semantic_objects = load_pickle(object_file)
                semantic_object = semantic_objects.objects[0]

                # check
                # assert len(semantic_object.unlabeled_grasps) == len(base_features_list)

                # extract features
                object_class = semantic_object.name

                print("Please enter the task and the object state.")
                print("Possible tasks are pour, scoop, stab, cut, lift, hammer, handover")
                task = input("This task is ? ")
                print("Possible states are hot, cold, empty, filled, has stuff, lid on, lid off")
                object_state = input("The current object state is ? ")

                # for grasp, base_features in zip(semantic_object.unlabeled_grasps, base_features_list):
                for grasp in semantic_object.grasps:

                    # label = base_features.label
                    label = 8

                    # extract context
                    context = (task, object_class, object_state)

                    # extract base features
                    # features = []
                    # features.append(base_features.object_spherical_resemblance)
                    # features.append(base_features.object_cylindrical_resemblance)
                    # features.extend(base_features.object_elongatedness)
                    # features.append(base_features.object_volume)
                    # features.extend(base_features.grasp_relative_position)
                    # features.append(base_features.object_opening)
                    # features.append(base_features.grasp_opening_angle)
                    # features.append(base_features.grasp_opening_distance)
                    # features.append(base_features.grasp_color_mean)
                    # features.append(base_features.grasp_color_variance)
                    # features.append(base_features.grasp_color_entropy)
                    # histograms = []
                    # histograms.extend(base_features.grasp_intensity_histogram)
                    # histograms.extend(base_features.grasp_first_gradient_histogram)
                    # histograms.extend(base_features.grasp_second_gradient_histogram)
                    # histograms.extend(base_features.grasp_color_histogram)
                    # descriptor = base_features.object_esf_descriptor
                    # extracted_base_features = (features, histograms, descriptor)
                    extracted_base_features = (None, None, None)

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

                    # Important: here is the definition of each data point
                    self.data.append([label,
                                      context,
                                      extracted_base_features,
                                      extracted_grasp_semantic_features,
                                      extracted_object_semantic_parts])

    # def rank_with_base_features_model(self, model, scalar):
    #     features = []
    #     histograms = []
    #     descriptors = []
    #
    #     for grasp in self.data:
    #         extracted_base_features = grasp[2]
    #         features.append(extracted_base_features[0])
    #         histograms.append(extracted_base_features[1])
    #         descriptors.append(extracted_base_features[2])
    #
    #     features = np.array(features)
    #     histograms = np.array(histograms)
    #     descriptors = np.array(descriptors)
    #
    #     features = scalar.transform(features)
    #     X_test = np.nan_to_num(np.concatenate([features, histograms, descriptors], axis=1))
    #
    #     Y_probs = model.predict_proba(X_test)
    #     pos_preds = Y_probs[:, -1]
    #     sort_indices = np.argsort(pos_preds)[::-1]

    def rank_with_wide_and_deep_model(self, model_filename):

        with open(model_filename, "rb") as fh:
            model_name, model, batcher, scalar = pickle.load(fh)

        labels = []
        features = []
        semantic_features = []
        visual_features = []
        histograms = []
        descriptors = []

        for data_pt in self.data:

            label = data_pt[0]
            labels.append(label)

            # semantic features: categorical
            grasp = data_pt[3]
            task = data_pt[1][0]
            object_class = data_pt[1][1]
            state = data_pt[1][2]
            parts = data_pt[4]

            # visual features: continuous
            extracted_base_features = data_pt[2]
            visual_features.append(extracted_base_features[0])
            histograms.append(extracted_base_features[1])
            descriptors.append(extracted_base_features[2])

            # important the order is changed from model 1
            semantic_features.append((task, object_class, state, grasp, parts))

        # # vectorize base features
        # visual_features = np.array(visual_features)
        # histograms = np.array(histograms)
        # descriptors = np.array(descriptors)
        #
        # visual_features = scalar.transform(visual_features)
        # np.nan_to_num(histograms)
        # # concatenate
        # base_features = np.nan_to_num(np.concatenate([visual_features, histograms, descriptors], axis=1))
        # # base_features = np.nan_to_num(visual_features)
        # base_features = base_features.tolist()

        for i in range(len(semantic_features)):
            # features.append((semantic_features[i], base_features[i]))
            features.append((semantic_features[i], None))

        batch_semantic_features, batch_base_features = batcher.batch_one_object(features)
        batch_semantic_features = torch.LongTensor(batch_semantic_features)

        model.eval()
        log_probs = model(batch_semantic_features, None)
        probs = torch.exp(log_probs).clone().cpu().data.numpy()
        pos_preds = probs[:, -1]
        sort_indices = np.argsort(pos_preds)[::-1]

        print(sort_indices)


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
    ge = GraspExecutor("/home/weiyu/catkin_ws/src/rail_semantic_grasping/data")
    ge.rank_with_wide_and_deep_model("/home/weiyu/catkin_ws/src/rail_semantic_grasping/models/exp4_wd.pkl")



