#include "rail_semantic_grasping/base_features_computation.h"

using namespace std;
using namespace rail::semantic_grasping;

BaseFeaturesComputation::BaseFeaturesComputation() : private_node_("~"), tf2_(tf_buffer_)
{
    compute_base_features_srv_ = private_node_.advertiseService("compute_base_features",
            &BaseFeaturesComputation::computeBaseFeaturesCallback, this);
}

bool BaseFeaturesComputation::computeBaseFeaturesCallback(rail_semantic_grasping::ComputeBaseFeaturesRequest &req,
                                                          rail_semantic_grasping::ComputeBaseFeaturesResponse &res)
{
    rail_semantic_grasping::SemanticObject semantic_object = req.semantic_objects.objects[0];
    ROS_INFO("The object has %zu parts and %zu grasps", semantic_object.parts.size(),
            semantic_object.labeled_grasps.size());

    // compute object-level features



    // compute features for each grasp
    for (size_t gi = 0; gi < semantic_object.labeled_grasps.size(); ++gi)
    {
        rail_semantic_grasping::SemanticGrasp grasp = semantic_object.labeled_grasps[gi];

    }


    return true;
}

