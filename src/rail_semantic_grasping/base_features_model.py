#!/usr/bin/env python

import os
import shutil
import glob
from datetime import datetime
import pickle
import copy
import numpy as np
from collections import defaultdict

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

class BaseFeaturesModel:
    TASKS = ["pour", "scoop", "stab", "cut", "lift", "hammer", "handover"]

    def __init__(self):

        self.compute_base_features_topic = rospy.get_param("~compute_base_features_topic",
                                                           "/base_features_computation_node/compute_base_features")

        self.data_dir = rospy.get_param("~data_dir_path",
                                        os.path.join(rospkg.RosPack().get_path("rail_semantic_grasping"), "data"))
        self.labeled_data_dir = os.path.join(self.data_dir, "labeled")

        # Set up data folder
        if not os.path.exists(self.data_dir):
            print("Data folder {} does not exist. Exiting!".format(self.data_dir))
            exit()
        if not os.path.exists(self.data_dir):
            print("Labeled data folder {} does not exist. Exiting!".format(self.labeled_data_dir))
            exit()

        self.data = defaultdict(list)

        # Set up service client
        rospy.wait_for_service(self.compute_base_features_topic)
        self.compute_base_features = rospy.ServiceProxy(self.compute_base_features_topic, ComputeBaseFeatures)

        # Listen to semantic objects with grasps
        # self.semantic_objects_sub = rospy.Subscriber(self.semantic_objects_topic,
        #                                              SemanticObjectList, self.semantic_objects_callback)
        # rospy.loginfo("Listen to semantic objects with grasp from {}".format(self.semantic_objects_topic))

        # Set up publishers
        self.markers_pub = rospy.Publisher("~data_collection/markers", MarkerArray, queue_size=10, latch=True)
        self.grasp_pub = rospy.Publisher("~data_collection/grasp", PoseStamped, queue_size=10, latch=True)
        self.marker_pub = rospy.Publisher("~data_collection/marker", Marker, queue_size=10, latch=True)
        self.color_image_pub = rospy.Publisher("~data_collection/color_image", Image, queue_size=10, latch=True)
        self.pc_pub = rospy.Publisher("~data_collection/point_cloud", PointCloud2, queue_size=10, latch=True)

    def compute_features(self):
        # grab all sessions in the unlabeled data dir
        session_dirs = glob.glob(os.path.join(self.labeled_data_dir, "*"))

        for session_dir in session_dirs:
            object_files = glob.glob(os.path.join(session_dir, "*.pkl"))

            # iterate through objects
            for object_file in object_files:
                with open(object_file, "rb") as fh:
                    semantic_objects = pickle.load(fh)
                # key = raw_input("Proceed with semantic objects: {}? y/n ".format(object_file))
                # if key != "y":
                #     continue

                # assume there is only one object in the list
                if not semantic_objects.objects:
                    continue

                # visualize semantic object
                markers = MarkerArray()
                marker = Marker()
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
                        print("Service call failed")
                        exit()

                # add features to data
                # each instance is a list of [task, label, features, histograms, descriptor]
                for base_features in resp.base_features_list:
                    histograms = []
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
                    histograms.extend(base_features.grasp_intensity_histogram)
                    histograms.extend(base_features.grasp_first_gradient_histogram)
                    histograms.extend(base_features.grasp_second_gradient_histogram)
                    histograms.extend(base_features.grasp_color_histogram)
                    descriptor = base_features.object_esf_descriptor
                    task = base_features.task
                    label = base_features.label
                    self.data[task].append([label, features, histograms, descriptor])

    def run_knn(self):
        """
        This function runs KNN on base features
        :return:
        """

        for task in self.data:
            print("#"*50)
            print("Run KNN for task {}".format(task))
            data_for_task = self.data[task]

            # preprocess features
            features = np.array([instance[1] for instance in data_for_task])
            scalar = preprocessing.StandardScaler()
            scalar.fit(features)
            features = scalar.transform(features)

            # concatenate features with histograms and descriptor
            histograms = np.array([instance[2] for instance in data_for_task])
            descriptors = np.array([instance[3] for instance in data_for_task])
            X = np.concatenate([features, histograms, descriptors], axis=1)
            Y = np.array([instance[0] for instance in data_for_task])
            Y[Y==-1] = 0
            print("X shpae", X.shape)
            print("Y shape", Y.shape)
            print("Neg:Pos ratio: {}".format((Y.size - np.sum(Y))/np.sum(Y)*1.0))

            # split data
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)

            # run algorithm
            train_classifier_logistic(X_train, X_test, Y_train, Y_test)
            train_classifier_knn(X_train, X_test, Y_train, Y_test)
            train_classifier_lmnn(X_train, X_test, Y_train, Y_test)


def train_classifier_logistic(X_train, X_test, Y_train, Y_test):
    classifier = LogisticRegressionCV(cv=10, tol=0.0001, class_weight='balanced', random_state=42,
                                      multi_class='ovr', verbose=False)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    print("Logistic Regression")
    print("accuracy: ", accuracy_score(Y_test, Y_pred))
    print("classification_report: ")
    print(classification_report(Y_test, Y_pred))


def train_classifier_knn(X_train, X_test, Y_train, Y_test):
    classifier = KNeighborsClassifier(n_neighbors=5)
    classifier.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)

    print("KNN")
    print("accuracy: ", accuracy_score(Y_test, Y_pred))
    print("classification_report: ")
    print(classification_report(Y_test, Y_pred))


def train_classifier_lmnn(X_train, X_test, Y_train, Y_test):
    # set up the hyperparamters
    k_train, k_test, n_components, max_iter = 3, 3, min(X_train.shape), 180

    # Learn the metric
    lmnn = LMNN(n_neighbors=k_train, max_iter=max_iter, n_components=n_components)
    lmnn.fit(X_train, Y_train)

    # Train KNN
    classifier = KNeighborsClassifier(n_neighbors=k_test)
    classifier.fit(lmnn.transform(X_train), Y_train)
    Y_pred = classifier.predict(lmnn.transform(X_test))

    print("LMNN")
    print("accuracy: ", accuracy_score(Y_test, Y_pred))
    print("classification_report: ")
    print(classification_report(Y_test, Y_pred))
    print(lmnn.components_.shape)
