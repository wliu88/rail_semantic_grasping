#!/usr/bin/env python

import os
import shutil
import glob
from datetime import datetime
import pickle
import copy
import numpy as np
from collections import defaultdict, OrderedDict
import pandas as pd

import rospy
import rospkg

import cv2
import cv_bridge

from rail_semantic_grasping.msg import SemanticObjectList, SemanticGrasp
from geometry_msgs.msg import Pose, Point, PoseStamped
from sensor_msgs.msg import Image, PointCloud2
# from rail_part_affordance_detection.msg import ObjectPartAffordance
# from rail_part_affordance_detection.srv import DetectAffordances, DetectAffordancesResponse
from visualization_msgs.msg import Marker, MarkerArray
from rail_semantic_grasping.srv import ComputeBaseFeatures

from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.utils import shuffle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier

from pylmnn import LargeMarginNearestNeighbor as LMNN


class BaseFeaturesCollection:

    TASKS = ["pour", "scoop", "stab", "cut", "lift", "hammer", "handover"]
    OBJECTS = ["cup", "spatula", "bowl", "pan"]

    def __init__(self):

        self.compute_base_features_topic = rospy.get_param("~compute_base_features_topic",
                                                           "/base_features_computation_node/compute_base_features")

        self.data_dir = rospy.get_param("~data_dir_path",
                                        os.path.join(rospkg.RosPack().get_path("rail_semantic_grasping"), "data"))
        self.labeled_data_dir = os.path.join(self.data_dir, "labeled")
        self.base_features_dir = os.path.join(self.data_dir, "base_features")
        if not os.path.exists(self.base_features_dir):
            os.mkdir(self.base_features_dir)

        # Set up data folder
        if not os.path.exists(self.data_dir):
            rospy.loginfo("Data folder {} does not exist. Exiting!".format(self.data_dir))
            exit()
        if not os.path.exists(self.data_dir):
            rospy.loginfo("Labeled data folder {} does not exist. Exiting!".format(self.labeled_data_dir))
            exit()

        # self.data = OrderedDict()
        # self.semantic_data = OrderedDict()
        # for task in BaseFeaturesModel.TASKS:
        #     self.data[task] = OrderedDict()
        #     self.semantic_data[task] = OrderedDict()
        #     for object in BaseFeaturesModel.OBJECTS:
        #         self.data[task][object] = OrderedDict()
        #         self.semantic_data[task][object] = OrderedDict()

        # Set up publishers
        self.markers_pub = rospy.Publisher("~data_collection/markers", MarkerArray, queue_size=10, latch=True)
        self.grasp_pub = rospy.Publisher("~data_collection/grasp", PoseStamped, queue_size=10, latch=True)
        self.marker_pub = rospy.Publisher("~data_collection/marker", Marker, queue_size=10, latch=True)
        self.color_image_pub = rospy.Publisher("~data_collection/color_image", Image, queue_size=10, latch=True)
        self.pc_pub = rospy.Publisher("~data_collection/point_cloud", PointCloud2, queue_size=10, latch=True)

        # Set up service client
        rospy.wait_for_service(self.compute_base_features_topic)
        self.compute_base_features = rospy.ServiceProxy(self.compute_base_features_topic, ComputeBaseFeatures)

    def compute_features(self):
        # grab all sessions in the labeled data dir
        session_dirs = glob.glob(os.path.join(self.labeled_data_dir, "*"))

        for session_dir in session_dirs:
            # create corresponding session folder in base features folder
            base_features_session_dir = os.path.join(self.base_features_dir, session_dir.split("/")[-1])
            if not os.path.exists(base_features_session_dir):
                os.mkdir(base_features_session_dir)

            object_files = glob.glob(os.path.join(session_dir, "*.pkl"))

            # iterate through objects
            for object_file in object_files:
                with open(object_file, "rb") as fh:
                    semantic_objects = pickle.load(fh)

                # assume there is only one object in the list
                if not semantic_objects.objects:
                    continue

                # visualize semantic object
                markers = MarkerArray()
                # assume there is only one object in the list
                semantic_object = semantic_objects.objects[0]
                marker = semantic_object.marker
                for semantic_part in semantic_object.parts:
                    markers.markers.append(semantic_part.marker)
                    markers.markers.append(semantic_part.text_marker)
                self.markers_pub.publish(markers)
                self.color_image_pub.publish(semantic_object.color_image)
                self.pc_pub.publish(semantic_object.point_cloud)
                self.marker_pub.publish(marker)

                # compute base features
                try:
                    resp = self.compute_base_features(semantic_objects)
                except rospy.ServiceException:
                    rospy.loginfo("Service call failed")
                    exit()

                # save base features to pickle file
                base_features_file = os.path.join(base_features_session_dir, object_file.split("/")[-1])
                with open(base_features_file, "wb") as fh:
                    pickle.dump(resp.base_features_list, fh)

        rospy.loginfo("Computation of base features for all objects are finished.")












#
#
#
#
#                 # add features to data
#                 # each instance is a list of [task, label, features, histograms, descriptor]
#                 for base_features in resp.base_features_list:
#                     histograms = []
#                     features = []
#                     features.append(base_features.object_spherical_resemblance)
#                     features.append(base_features.object_cylindrical_resemblance)
#                     features.extend(base_features.object_elongatedness)
#                     features.append(base_features.object_volume)
#                     features.extend(base_features.grasp_relative_position)
#                     features.append(base_features.object_opening)
#                     features.append(base_features.grasp_opening_angle)
#                     features.append(base_features.grasp_opening_distance)
#                     features.append(base_features.grasp_color_mean)
#                     features.append(base_features.grasp_color_variance)
#                     features.append(base_features.grasp_color_entropy)
#                     histograms.extend(base_features.grasp_intensity_histogram)
#                     histograms.extend(base_features.grasp_first_gradient_histogram)
#                     histograms.extend(base_features.grasp_second_gradient_histogram)
#                     histograms.extend(base_features.grasp_color_histogram)
#                     descriptor = base_features.object_esf_descriptor
#                     task = base_features.task
#                     label = base_features.label
#
#                     if object_id not in self.data[task][object_class]:
#                         self.data[task][object_class][object_id] = []
#                     self.data[task][object_class][object_id].append([label, features, histograms, descriptor])
#
#                 for grasp in semantic_object.labeled_grasps:
#                     task = grasp.task
#                     label = grasp.score
#                     affordance = grasp.grasp_part_affordance
#                     material = grasp.grasp_part_material
#                     if object_id not in self.semantic_data[task][object_class]:
#                         self.semantic_data[task][object_class][object_id] = []
#                     self.semantic_data[task][object_class][object_id].append([label, affordance, material])
#
#     def run_knn(self):
#         """
#         This function runs KNN on base features
#         :return:
#         """
#
#         TEST_PERCENTAGE = 0.5
#
#         map_scores = np.ones([len(BaseFeaturesModel.TASKS), len(BaseFeaturesModel.OBJECTS)]) * -1.0
#         map_scores_1 = np.ones([len(BaseFeaturesModel.TASKS), len(BaseFeaturesModel.OBJECTS)]) * -1.0
#         for ti, task in enumerate(self.data):
#             for oi, object_class in enumerate(self.data[task]):
#
#                 # # debug
#                 # if object_class != "cup":
#                 #     continue
#
#                 print("#"*50)
#                 print("Run KNN for task {} and object {}".format(task, object_class))
#                 objects_data = self.data[task][object_class]
#
#                 semantic_objects_data = self.semantic_data[task][object_class]
#
#                 num_objects = len(objects_data)
#                 num_test = int(num_objects * TEST_PERCENTAGE)
#                 print("Number of objects:", num_objects)
#                 print("Number of test objects:", num_test)
#                 if not num_test:
#                     print("Not enough object to test")
#                     continue
#
#                 APs = []
#                 APs_1 = []
#                 # repeat test 10 times
#                 for test in range(10):
#                     train_object_ids = np.random.choice(num_objects, num_objects - num_test, replace=False)
#
#                     # Low level features
#                     features = []
#                     histograms = []
#                     descriptors = []
#                     labels = []
#                     for id in train_object_ids:
#                         object_data = objects_data[id]
#                         features.extend([instance[1] for instance in object_data])
#                         histograms.extend([instance[2] for instance in object_data])
#                         descriptors.extend([instance[3] for instance in object_data])
#                         labels.extend([instance[0] for instance in object_data])
#
#                     features = np.array(features)
#                     histograms = np.array(histograms)
#                     descriptors = np.array(descriptors)
#
#                     # preprocess features
#                     scalar = preprocessing.StandardScaler()
#                     scalar.fit(features)
#                     features = scalar.transform(features)
#
#                     X_train = np.concatenate([features, histograms, descriptors], axis=1)
#                     Y_train = np.array(labels)
#                     Y_train[Y_train == -1] = 0
#                     if np.sum(Y_train) == 0:
#                         print("Skip this task because there is no positive examples")
#                         continue
#                     # print("X shape", X_train.shape)
#                     # print("Y shape", Y_train.shape)
#                     print("Neg:Pos ratio: {}".format((Y_train.size - np.sum(Y_train)) / np.sum(Y_train) * 1.0))
#
#                     classifier = KNeighborsClassifier(n_neighbors=5)
#                     # classifier = LogisticRegressionCV(cv=10, tol=0.0001, class_weight='balanced', random_state=42,
#                     #                                  multi_class='ovr', verbose=False)
#                     classifier.fit(X_train, Y_train)
#
#                     # Test
#                     for id in range(num_objects):
#                         if id in train_object_ids:
#                             continue
#                         object_data = objects_data[id]
#                         features = [instance[1] for instance in object_data]
#                         histograms = [instance[2] for instance in object_data]
#                         descriptors = [instance[3] for instance in object_data]
#                         labels = [instance[0] for instance in object_data]
#
#                         features = np.array(features)
#                         histograms = np.array(histograms)
#                         descriptors = np.array(descriptors)
#
#                         # preprocess features
#                         features = scalar.transform(features)
#
#                         X_test = np.concatenate([features, histograms, descriptors], axis=1)
#                         Y_test = np.array(labels)
#                         Y_test[Y_test == -1] = 0
#
#                         # predict
#                         Y_probs = classifier.predict_proba(X_test)[:, 1]
#
#                         # calculate AP
#                         sort_indices = np.argsort(Y_probs)[::-1]
#                         Y_probs = Y_probs[sort_indices]
#                         Y_test = Y_test[sort_indices]
#
#                         num_corrects = 0.0
#                         num_predictions = 0.0
#                         total_precisions = []
#                         for i in range(len(Y_test)):
#                             num_predictions += 1
#                             if Y_test[i] == 1:
#                                 num_corrects += 1
#                                 total_precisions.append(num_corrects / num_predictions)
#                         ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(total_precisions) > 0 else None
#                         if ap is not None:
#                             APs.append(ap)
#
#                     # Semantic grasp features
#                     # grasp affordance
#                     context_to_score = defaultdict(list)
#                     labels = []
#                     for id in train_object_ids:
#                         semantic_object_data = semantic_objects_data[id]
#                         for instance in semantic_object_data:
#                             score = instance[0]
#                             context = instance[1]
#                             context_to_score[context].append(score)
#
#                     for context in context_to_score:
#                         context_to_score[context] = sum(context_to_score[context]) * 1.0 / len(context_to_score[context])
#
#                     # Test
#                     for id in range(num_objects):
#                         if id in train_object_ids:
#                             continue
#
#                         semantic_object_data = semantic_objects_data[id]
#                         Y_test = []
#                         Y_probs = []
#                         for instance in semantic_object_data:
#                             score = instance[0]
#                             context = instance[1]
#                             if context not in context_to_score:
#                                 prob = 0
#                             else:
#                                 prob = context_to_score[context]
#                             Y_test.append(score)
#                             Y_probs.append(prob)
#
#                         Y_test = np.array(Y_test)
#                         Y_probs = np.array(Y_probs)
#
#                         # calculate AP
#                         sort_indices = np.argsort(Y_probs)[::-1]
#                         Y_probs = Y_probs[sort_indices]
#                         Y_test = Y_test[sort_indices]
#
#
#                         num_corrects = 0.0
#                         num_predictions = 0.0
#                         total_precisions = []
#                         for i in range(len(Y_test)):
#                             num_predictions += 1
#                             if Y_test[i] == 1:
#                                 num_corrects += 1
#                                 total_precisions.append(num_corrects / num_predictions)
#                         ap = sum(total_precisions) * 1.0 / len(total_precisions) if len(
#                             total_precisions) > 0 else None
#                         if ap is not None:
#                             APs_1.append(ap)
#
#                 if APs:
#                     MAP = np.average(APs)
#                     print("MAP:", MAP)
#                     map_scores[ti, oi] = MAP
#
#                 if APs_1:
#                     MAP = np.average(APs_1)
#                     print("MAP:", MAP)
#                     map_scores_1[ti, oi] = MAP
#
#         map_scores = pd.DataFrame(map_scores, index=BaseFeaturesModel.TASKS, columns=BaseFeaturesModel.OBJECTS)
#         map_scores_1 = pd.DataFrame(map_scores_1, index=BaseFeaturesModel.TASKS, columns=BaseFeaturesModel.OBJECTS)
#         print(map_scores)
#         print(map_scores_1)
#
#
# def train_classifier_logistic(X_train, X_test, Y_train, Y_test):
#     classifier = LogisticRegressionCV(cv=10, tol=0.0001, class_weight='balanced', random_state=42,
#                                       multi_class='ovr', verbose=False)
#     classifier.fit(X_train, Y_train)
#     Y_pred = classifier.predict(X_test)
#
#     print("Logistic Regression")
#     print("accuracy: ", accuracy_score(Y_test, Y_pred))
#     print("classification_report: ")
#     print(classification_report(Y_test, Y_pred))
#
#
# def train_classifier_knn(X_train, X_test, Y_train, Y_test):
#     classifier = KNeighborsClassifier(n_neighbors=5)
#     classifier.fit(X_train, Y_train)
#     Y_pred = classifier.predict(X_test)
#
#     print("KNN")
#     print("accuracy: ", accuracy_score(Y_test, Y_pred))
#     print("classification_report: ")
#     print(classification_report(Y_test, Y_pred))
#
#
# def train_classifier_lmnn(X_train, X_test, Y_train, Y_test):
#     # set up the hyperparamters
#     k_train, k_test, n_components, max_iter = 3, 3, min(X_train.shape), 180
#
#     # Learn the metric
#     lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter, n_components=n_components)
#     lmnn.fit(X_train, Y_train)
#
#     # Train KNN
#     classifier = KNeighborsClassifier(n_neighbors=k_test)
#     classifier.fit(lmnn.transform(X_train), Y_train)
#     Y_pred = classifier.predict(lmnn.transform(X_test))
#
#     print("LMNN")
#     print("accuracy: ", accuracy_score(Y_test, Y_pred))
#     print("classification_report: ")
#     print(classification_report(Y_test, Y_pred))
#     print(lmnn.components_.shape)
