#ifndef RAIL_SEMANTIC_GRASPING_BASE_FEATURES_COMPUTATION_H_
#define RAIL_SEMANTIC_GRASPING_BASE_FEATURES_COMPUTATION_H_

// ROS
#include <pcl_ros/point_cloud.h>
#include <pcl_ros/transforms.h>
#include <ros/package.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>
#include <geometry_msgs/Point.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/TransformStamped.h>
#include <std_srvs/Empty.h>
#include <tf/transform_listener.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

//#include <rail_manipulation_msgs/SegmentedObject.h>
//#include <rail_manipulation_msgs/SegmentObjects.h>
//#include <rail_manipulation_msgs/SegmentObjectsFromPointCloud.h>
//#include <rail_manipulation_msgs/ProcessSegmentedObjects.h>

#include <rail_semantic_grasping/SemanticObjectList.h>
#include <rail_semantic_grasping/SemanticObject.h>
#include <rail_semantic_grasping/SemanticPart.h>
#include <rail_semantic_grasping/SegmentSemanticObjects.h>
#include <rail_semantic_grasping/ComputeBaseFeatures.h>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
#include<pcl/features/esf.h>
//#include <pcl/filters/passthrough.h>
//#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/segmentation/extract_clusters.h>
//#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/features/normal_3d.h>

// OPENCV
#include <opencv2/core/version.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>

// YAML
//#include <yaml-cpp/yaml.h>

// BOOST
#include <boost/thread/mutex.hpp>

// C++ Standard Library
#include <fstream>
#include <string>
#include <map>
#include <algorithm>    // std::sort

namespace rail
{
namespace semantic_grasping
{

class BaseFeaturesComputation
{
public:

  /*!
   * \brief Create a Segmenter and associated ROS information.
   *
   * Creates a ROS node handle, subscribes to the relevant topics and servers, and creates services for requesting
   * segmenations.
   */
  BaseFeaturesComputation();


private:

  bool computeBaseFeaturesCallback(rail_semantic_grasping::ComputeBaseFeaturesRequest &req,
                                   rail_semantic_grasping::ComputeBaseFeaturesResponse &res);

  bool debug_;

  int cylinder_segmentation_normal_k_, shape_segmentation_max_iteration_;
  double cylinder_segmentation_normal_distance_weight_, cylinder_segmentation_distance_threshold_ratio_,
         sphere_segmentation_distance_threshold_, sphere_segmentation_probability_;

  ros::ServiceServer compute_base_features_srv_;

  /*! The global and private ROS node handles. */
  ros::NodeHandle node_, private_node_;

  ros::Publisher debug_pc_pub_, debug_pc_pub_2_, debug_pc_pub_3_, debug_pose_pub_, debug_pose_pub_2_, debug_img_pub_;

  /*! Main transform listener. */
  tf::TransformListener tf_;
  /*! The transform tree buffer for the tf2 listener. */
  tf2_ros::Buffer tf_buffer_;
  /*! The buffered trasnform client. */
  tf2_ros::TransformListener tf2_;

};

}
}

#endif
