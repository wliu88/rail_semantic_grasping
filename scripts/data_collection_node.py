#!/usr/bin/env python
# The node for running data collection, which saves semantic objects to pickle files

import rospy

from rail_semantic_grasping.data_collection import DataCollection


def main():
    rospy.init_node('data_collection')
    dc = DataCollection()
    rospy.spin()


if __name__ == '__main__':
    main()
