#!/usr/bin/env python
# The node for running base features model, one of the baselines

import rospy

from rail_semantic_grasping.base_features_collection import BaseFeaturesCollection


def main():
    rospy.init_node('base_features_collection')
    bfm = BaseFeaturesCollection()
    bfm.compute_features()
    rospy.spin()


if __name__ == '__main__':
    main()