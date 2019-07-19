#!/usr/bin/env python

import os
import shutil
import glob
from datetime import datetime
import pickle
import copy
import numpy as np

import rospy
import rospkg

from rail_semantic_grasping.msg import SemanticObjectList, SemanticGrasp
from geometry_msgs.msg import Pose, Point, PoseStamped
from sensor_msgs.msg import Image, PointCloud2
# from rail_part_affordance_detection.msg import ObjectPartAffordance
# from rail_part_affordance_detection.srv import DetectAffordances, DetectAffordancesResponse
from visualization_msgs.msg import Marker, MarkerArray


class BaseFeaturesModel:
    TASKS = ["pour", "scoop", "stab", "cut", "lift", "hammer", "handover"]

    def __init__(self):

        self.semantic_objects_topic = rospy.get_param("~semantic_objects_with_grasps_topic")

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

        # Set up service client
        rospy.wait_for_service("")
        self.compute_base_features = rospy.ServiceProxy("", ComputeBaseFeatures)

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
                key = raw_input("Proceed with semantic objects: {}? y/n ".format(object_file))
                if key != "y":
                    continue

                # assume there is only one object in the list
                if not semantic_objects.objects:
                    continue

                # compute base features
                try:
                    self.compute_base_features(semantic_objects)
                except rospy.ServiceException:
                        print("Service call failed")
