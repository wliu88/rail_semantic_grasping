#include "rail_semantic_grasping/base_features_computation.h"

using namespace std;
using namespace rail::semantic_grasping;

/*!
 * Creates and runs the base_features_computation node.
 *
 * \param argc argument count that is passed to ros::init.
 * \param argv arguments that are passed to ros::init.
 * \return EXIT_SUCCESS if the node runs correctly or EXIT_FAILURE if an error occurs.
 */
int main(int argc, char **argv)
{
  // initialize ROS and the node
  ros::init(argc, argv, "base_features_computation_node");
  BaseFeaturesComputation base_features_computation;
  // check if everything started okay
  if (true)
  {
    ros::spin();
    return EXIT_SUCCESS;
  } else
  {
    return EXIT_FAILURE;
  }
}
