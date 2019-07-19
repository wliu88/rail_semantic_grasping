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


class DataCollection:
    """
    This class is responsible for manage semantic objects and grasps. It has two modes: collect objects or collect
    grasps.

    In collect objects mode, this class is continually listening to semantic objects and saves them to pickle files

    In collect grasps mode, this class loads saved pickle file and collects users' preference of grasps in different
    context (task, environment, and etc)

    """

    TASKS = ["pour", "scoop", "stab", "cut", "lift", "hammer", "handover"]
    TASK_DESCRIPTIONS = {"pour": "Grasp the object to pour the liquid out",
                         "scoop": "Grasp the object to scoop something",
                         "stab": "Grasp the object to stab",
                         "cut": "Grasp the object to cut",
                         "lift": "Grasp the object to used it for lifting something up. For example, use the spatula to lift an fried egg up.",
                         "hammer": "Grasp the object to hammer a nail",
                         "handover": "Grasp the object to hand it over to someone"}
    NUM_SAMPLE_GRASPS = 20

    def __init__(self, collect_objects=True):

        # Initialize semantic objects subscribers
        if collect_objects:
            self.semantic_objects_topic = rospy.get_param("~semantic_objects_with_grasps_topic")
        self.data_dir = rospy.get_param("~data_dir_path",
                                        os.path.join(rospkg.RosPack().get_path("rail_semantic_grasping"), "data"))

        # Set up data folder
        if not os.path.exists(self.data_dir):
            os.mkdir(self.data_dir)
            rospy.loginfo("Data folder is set up at {}".format(self.data_dir))
        else:
            rospy.loginfo("Data folder is at {}".format(self.data_dir))

        self.unlabeled_data_dir = os.path.join(self.data_dir, "unlabeled")
        if not os.path.exists(self.unlabeled_data_dir):
            os.mkdir(self.unlabeled_data_dir)

        self.labeled_data_dir = os.path.join(self.data_dir, "labeled")
        if not os.path.exists(self.labeled_data_dir):
            os.mkdir(self.labeled_data_dir)

        # set up session folder
        if collect_objects:
            time = datetime.now()  # fetch time
            date = time.strftime("%Y_%m_%d_%H_%M")
            self.session_dir = os.path.join(self.unlabeled_data_dir, date)
            os.mkdir(self.session_dir)
            rospy.loginfo("Start data collection session---data will be saved to {}".format(self.session_dir))

            self.object_counter = 0

        if collect_objects:
            # Listen to semantic objects with grasps
            self.semantic_objects_sub = rospy.Subscriber(self.semantic_objects_topic,
                                                         SemanticObjectList, self.semantic_objects_callback)
            rospy.loginfo("Listen to semantic objects with grasp from {}".format(self.semantic_objects_topic))
        else:
            # Set up publishers
            self.markers_pub = rospy.Publisher("~data_collection/markers", MarkerArray, queue_size=10, latch=True)
            self.grasp_pub = rospy.Publisher("~data_collection/grasp", PoseStamped, queue_size=10, latch=True)
            self.marker_pub = rospy.Publisher("~data_collection/marker", Marker, queue_size=10, latch=True)
            self.color_image_pub = rospy.Publisher("~data_collection/color_image", Image, queue_size=10, latch=True)
            self.pc_pub = rospy.Publisher("~data_collection/point_cloud", PointCloud2, queue_size=10, latch=True)


    def semantic_objects_callback(self, semantic_objects):
        """
        This method is used for listening to semantic objects and saves them to files.

        :param semantic_objects:
        :return:
        """
        object_file_path = os.path.join(self.session_dir, str(self.object_counter) + ".pkl")
        with open(object_file_path, "wb") as fh:
            pickle.dump(semantic_objects, fh)
            rospy.loginfo("Saved object No.{}".format(self.object_counter))
        self.object_counter += 1

    def collect_grasps(self):
        """
        This method is the used for collecting users' preferences of grasps.

        :return:
        """
        # grab all sessions in the unlabeled data dir
        session_dirs = glob.glob(os.path.join(self.unlabeled_data_dir, "*"))

        for session_dir in session_dirs:
            object_files = glob.glob(os.path.join(session_dir, "*.pkl"))

            # prepare dir for saving labeled data
            labeled_session_dir = session_dir.replace("unlabeled", "labeled")
            if not os.path.exists(labeled_session_dir):
                os.mkdir(labeled_session_dir)

            # iterate through objects
            for object_file in object_files:
                with open(object_file, "rb") as fh:
                    semantic_objects = pickle.load(fh)
                key = raw_input("Proceed with semantic objects: {}? y/n ".format(object_file))
                if key != "y":
                    continue

                # visualize semantic object
                markers = MarkerArray()
                marker = Marker()
                # assume there is only one object in the list
                if not semantic_objects.objects:
                    continue
                semantic_object = semantic_objects.objects[0]
                marker = semantic_object.marker
                for semantic_part in semantic_object.parts:
                    markers.markers.append(semantic_part.marker)
                    markers.markers.append(semantic_part.text_marker)
                self.markers_pub.publish(markers)
                self.color_image_pub.publish(semantic_object.color_image)
                self.pc_pub.publish(semantic_object.point_cloud)
                self.marker_pub.publish(marker)

                # iterate through grasps
                if not semantic_object.grasps:
                    continue
                rospy.loginfo("#"*100)
                rospy.loginfo("Current object has {} grasps".format(len(semantic_object.grasps)))
                # Important: sample some number of grasps
                sampled_grasps = np.random.choice(semantic_object.grasps, DataCollection.NUM_SAMPLE_GRASPS,
                                                  replace=False).tolist()
                rospy.loginfo("Sample {} grasps for labeling".format(len(sampled_grasps)))
                rospy.loginfo("*" * 100)
                rospy.loginfo("*" * 100)

                labeled_grasps = []
                skip_object = False
                # label grasp preferences for each task
                for task in DataCollection.TASKS:

                    rospy.loginfo("For task: {}".format(task))
                    rospy.loginfo("Task description: '{}'".format(DataCollection.TASK_DESCRIPTIONS[task]))
                    rospy.loginfo("*" * 100)
                    rospy.loginfo("*" * 100)
                    grasps_for_task = copy.deepcopy(sampled_grasps)

                    for gi, semantic_grasp in enumerate(grasps_for_task):
                        semantic_grasp.task = task
                        pose_stamped = PoseStamped()
                        pose_stamped.header.frame_id = semantic_objects.header.frame_id
                        pose_stamped.pose = semantic_grasp.grasp_pose
                        self.grasp_pub.publish(pose_stamped)
                        rospy.loginfo("Grasp No.{}/{} is on the part with affordance {} and material {}".format(gi+1,
                                      DataCollection.NUM_SAMPLE_GRASPS, semantic_grasp.grasp_part_affordance,
                                      semantic_grasp.grasp_part_material))
                        valid = False
                        while not valid:
                            key = raw_input("Is this grasp semantically correct? absolutely(press 1) / ok(press 2) / definitely not(press 3) ")
                            if key == "1":
                                semantic_grasp.score = 1
                                valid = True
                            elif key == "3":
                                semantic_grasp.score = -1
                                valid = True
                            elif key == "2" or key == "":
                                # Important: score is initialized to 0
                                semantic_grasp.score = 0
                                valid = True
                            elif key == "q":
                                skip_object = True
                                break
                            else:
                                rospy.loginfo("Not a valid input, try again")
                        if skip_object:
                            break

                    if skip_object:
                        break
                    labeled_grasps += grasps_for_task

                if skip_object:
                    continue
                semantic_object.labeled_grasps = labeled_grasps

                rospy.loginfo("Saving labeled grasps...\n")
                new_object_file = object_file.replace("unlabeled", "labeled")
                with open(new_object_file, "wb") as fh:
                    pickle.dump(semantic_objects, fh)

                # remove markers and pc
                for i in range(len(markers.markers)):
                    markers.markers[i].action = 2
                marker.action = 2
                self.markers_pub.publish(markers)
                self.marker_pub.publish(marker)
                clear_pc = PointCloud2()
                clear_pc.header = semantic_object.point_cloud.header
                self.pc_pub.publish(clear_pc)

                valid = False
                while not valid:
                    key = raw_input("Next object? type '!' to continue")
                    if key == "!":
                        semantic_grasp.score = 1
                        valid = True
                    else:
                        rospy.loginfo("Not a valid input, try again")
                if skip_object:
                    break

        rospy.loginfo("All objects has finished labeling. Exiting!")
        exit()

    def visualize_grasps(self):
        """
        This method is used for visualizing labeled grasps.

        :return:
        """
        # grab all sessions in the labeled data dir
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

                # visualize semantic object
                markers = MarkerArray()
                # assume there is only one object in the list
                rospy.loginfo("{}".format(len(semantic_objects.objects)))
                if not semantic_objects.objects:
                    continue
                semantic_object = semantic_objects.objects[0]
                for semantic_part in semantic_object.parts:
                    markers.markers.append(semantic_part.marker)
                    markers.markers.append(semantic_part.text_marker)
                self.markers_pub.publish(markers)

                # iterate through grasps
                if not semantic_object.grasps:
                    continue
                rospy.loginfo("Current object has {} grasps".format(len(semantic_object.grasps)))
                for gi, semantic_grasp in enumerate(semantic_object.grasps):
                    pose_stamped = PoseStamped()
                    pose_stamped.header.frame_id = semantic_objects.header.frame_id
                    pose_stamped.pose = semantic_grasp.grasp_pose
                    self.grasp_pub.publish(pose_stamped)
                    print(semantic_grasp.score)
                    rospy.loginfo("Grasp No.{} on the part with affordance {} is semantically correct? {}".format(gi, semantic_grasp.grasp_part_affordance, semantic_grasp.score))
                    key = raw_input("enter to continue")

        rospy.loginfo("All objects has finished visualizing. Exiting!")
        exit()

    def summarize_grasps(self):
        """
        This method is used for visualizing labeled grasps.

        :return:
        """
        TEST_RATIO = 0.7

        # grab all sessions in the labeled data dir
        session_dirs = glob.glob(os.path.join(self.labeled_data_dir, "*"))

        for session_dir in session_dirs:
            object_files = glob.glob(os.path.join(session_dir, "*.pkl"))

            # mapping from (object, task, material, affordance) to sum score

            preference_table_raw = {}

            # iterate through objects
            for object_file in object_files:
                with open(object_file, "rb") as fh:
                    semantic_objects = pickle.load(fh)
                    # assume only one object in the list
                    semantic_object = semantic_objects.objects[0]

                    obj_cls = semantic_object.name

                    # for part in semantic_object.parts:
                    #     part_mat = part.material
                    #     part_aff = part.affordance

                    for semantic_grasp in semantic_object.labeled_grasps:

                        if np.random.random() > (1 - TEST_RATIO):
                            continue

                        task = semantic_grasp.task
                        part_aff = semantic_grasp.grasp_part_affordance
                        part_mat = semantic_grasp.grasp_part_material
                        score = int(semantic_grasp.score)
                        if score == 0:
                            score = -1

                        context = (obj_cls, task, part_aff, part_mat)

                        if context not in preference_table_raw:
                            preference_table_raw[context] = []
                        preference_table_raw[context].append(score)

            # 1. Object
            # a mapping from object to task to affordance to material to score
            object_preference_table = {}
            for context in preference_table_raw:
                object = context[0]
                if object not in object_preference_table:
                    object_preference_table[object] = {}

                task = context[1]
                if task not in object_preference_table[object]:
                    object_preference_table[object][task] = {}

                affordance = context[2]
                if affordance not in object_preference_table[object][task]:
                    object_preference_table[object][task][affordance] = {}

                material = context[3]
                if affordance not in object_preference_table[object][task][affordance]:
                    object_preference_table[object][task][affordance][material] = 0

                object_preference_table[object][task][affordance][material] = sum(preference_table_raw[context]) * 1.0 / len(preference_table_raw[context])

            print("There are {} contexts".format(len(preference_table_raw)))
            context_occurances = []
            for context in preference_table_raw:
                context_occurances.append(len(preference_table_raw[context]))
            print("There are {} instances on average for each context".format(sum(context_occurances)*1.0/len(context_occurances)))

            # print
            # for object in object_preference_table:
            #     print("#"*100)
            #     print("Object: " + object)
            #     for task in object_preference_table[object]:
            #         print("-" * 50)
            #         print("Task: " + task)
            #         for affordance in object_preference_table[object][task]:
            #             if affordance == "None":
            #                 print("\nAffordance: Unknown")
            #             else:
            #                 print("\nAffordance: " + affordance)
            #             for material in object_preference_table[object][task][affordance]:
            #                 if material == "":
            #                     print("\tUnknown: " + str(object_preference_table[object][task][affordance][material]))
            #                 else:
            #                     print("\t"+ material + ": " + str(object_preference_table[object][task][affordance][material]))

            total_correct = 0
            total_prediction = 0
            for object_file in object_files:
                with open(object_file, "rb") as fh:
                    semantic_objects = pickle.load(fh)
                    # assume only one object in the list
                    semantic_object = semantic_objects.objects[0]
                    obj_cls = semantic_object.name
                    for semantic_grasp in semantic_object.labeled_grasps:
                        total_prediction += 1

                        task = semantic_grasp.task
                        part_aff = semantic_grasp.grasp_part_affordance
                        part_mat = semantic_grasp.grasp_part_material
                        score = int(semantic_grasp.score)
                        if score == 0:
                            score = -1

                        prediction = np.random.random() * 2 - 1
                        try:
                            prediction = object_preference_table[obj_cls][task][part_aff][part_mat]
                        except:
                            pass

                        if abs(score - prediction) < 1:
                            total_correct += 1
            print("Frequency baseline accuracy: {}".format(total_correct*1.0/total_prediction))

            total_correct = 0
            total_prediction = 0
            for object_file in object_files:
                with open(object_file, "rb") as fh:
                    semantic_objects = pickle.load(fh)
                    # assume only one object in the list
                    semantic_object = semantic_objects.objects[0]
                    obj_cls = semantic_object.name
                    for semantic_grasp in semantic_object.labeled_grasps:
                        total_prediction += 1

                        task = semantic_grasp.task
                        part_aff = semantic_grasp.grasp_part_affordance
                        part_mat = semantic_grasp.grasp_part_material
                        score = int(semantic_grasp.score)
                        if score == 0:
                            score = -1

                        prediction = np.random.random() * 2 - 1
                        if abs(score - prediction) < 1:
                            total_correct += 1
            print("Random baseline accuracy: {}".format(total_correct * 1.0 / total_prediction))


        exit()