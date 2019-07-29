#!/usr/bin/env python
# The node for running base features model, one of the baselines

import rospy

from rail_semantic_grasping.base_features_model import BaseFeaturesModel


def main():
    rospy.init_node('base_features_model')
    bfm = BaseFeaturesModel()
    bfm.compute_features()
    bfm.run_knn()
    rospy.spin()


if __name__ == '__main__':
    main()