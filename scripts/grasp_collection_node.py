#!/usr/bin/env python
# The node for running grasp collection, which collects users' preferences of grasps in different contexts

import rospy

from rail_semantic_grasping.data_collection import DataCollection


def main():
    rospy.init_node('grasp_collection')
    dc = DataCollection(collect_objects=False)
    dc.collect_grasps()
    # dc.summarize_grasps()
    rospy.spin()


if __name__ == '__main__':
    main()
