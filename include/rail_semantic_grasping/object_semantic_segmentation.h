/*!
 * \file Segmenter.h
 * \brief The main segmentation node object.
 *
 * The segmenter is responsible for segmenting clusters from a point cloud topic. Visualization and data latched topics
 * are published after each request. A persistent array of objects is maintained internally.
 */

#ifndef RAIL_SEMANTIC_GRASPING_AFFORDANCE_SEGMENTATION_H_
#define RAIL_SEMANTIC_GRASPING_AFFORDANCE_SEGMENTATION_H_

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

#include <rail_part_affordance_detection/DetectAffordances.h>
#include <rail_part_affordance_detection/ObjectPartAffordance.h>

#include <rail_manipulation_msgs/SegmentedObject.h>
#include <rail_manipulation_msgs/SegmentObjects.h>
#include <rail_manipulation_msgs/SegmentObjectsFromPointCloud.h>
#include <rail_manipulation_msgs/ProcessSegmentedObjects.h>

#include <rail_semantic_grasping/SemanticObjectList.h>
#include <rail_semantic_grasping/SemanticObject.h>
#include <rail_semantic_grasping/SemanticPart.h>
#include <rail_semantic_grasping/SegmentSemanticObjects.h>

// PCL
#include <pcl/common/common.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/extract_indices.h>
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




// YAML
//#include <yaml-cpp/yaml.h>

// BOOST
#include <boost/thread/mutex.hpp>

// C++ Standard Library
#include <fstream>
#include <string>
#include <map>

namespace rail
{
namespace semantic_grasping
{

/*!
 * \class Segmenter
 * \brief The main grasp collector node object.
 *
 * The grasp collector is responsible for capturing and storing grasps. An action server is started is the main entry
 * point to grasp collecting.
 */
class ObjectSemanticSegmentation
{
public:
    std::map<int, std::string> idx_to_affordance;
    std::map<int, std::string> idx_to_object_class;

//#if __cplusplus >= 201103L
//  /*! If a topic should be created to display debug information such as point clouds. */
//  static constexpr bool DEFAULT_DEBUG = false;
//  /*! The angle epsilon (delta) threshold for the plane segmenter. */
//  static constexpr double SAC_EPS_ANGLE = 0.15;
//  /*! The distance threshold for the plane segmenter. */
//  static constexpr double SAC_DISTANCE_THRESHOLD = 0.01;
//  /*! The maximum interations for the plane segmenter */
//  static constexpr int SAC_MAX_ITERATIONS = 100;
//  /*! The padding for surface removal. */
//  static constexpr double SURFACE_REMOVAL_PADDING = 0.005;
//  /*! The minimum cluster size. */
//  static constexpr int DEFAULT_MIN_CLUSTER_SIZE = 200;
//  /*! The maximum cluster size. */
//  static constexpr int DEFAULT_MAX_CLUSTER_SIZE = 10000;
//  /*! The cluster tolerance level. */
//  static constexpr double CLUSTER_TOLERANCE = 0.02;
//  /*! The color tolerance level, only for RGB segmentation */
//  static constexpr double POINT_COLOR_THRESHOLD = 10;
//  /*! The region color tolerance, only for small region merging in RGB segmentation */
//  static constexpr double REGION_COLOR_THRESHOLD = 10;
//  /*! Leaf size of the voxel grid for downsampling. */
    static constexpr float DOWNSAMPLE_LEAF_SIZE = 0.01;
//  /*! Size of the marker visualization scale factor. */
    static constexpr double MARKER_SCALE = 0.01;
//#else
//  /*! If a topic should be created to display debug information such as point clouds. */
//  static const bool DEFAULT_DEBUG = false;
//  /*! The angle epsilon (delta) threshold for the plane segmenter. */
//  static const double SAC_EPS_ANGLE = 0.15;
//  /*! The distance threshold for the plane segmenter. */
//  static const double SAC_DISTANCE_THRESHOLD = 0.01;
//  /*! The maximum interations for the plane segmenter */
//  static const int SAC_MAX_ITERATIONS = 100;
//  /*! The padding for surface removal. */
//  static const double SURFACE_REMOVAL_PADDING = 0.005;
//  /*! The minimum cluster size. */
//  static const int DEFAULT_MIN_CLUSTER_SIZE = 200;
//  /*! The maximum cluster size. */
//  static const int DEFAULT_MAX_CLUSTER_SIZE = 10000;
//  /*! The cluster tolerance level. */
//  static const double CLUSTER_TOLERANCE = 0.02;
//  /*! The color tolerance level, only for RGB segmentation */
//  static const double POINT_COLOR_THRESHOLD = 10;
//  /*! The region color tolerance, only for small region merging in RGB segmentation */
//  static const double REGION_COLOR_THRESHOLD = 10;
//  /*! Leaf size of the voxel grid for downsampling. */
//  static const float DOWNSAMPLE_LEAF_SIZE = 0.01;
//  /*! Size of the marker visualization scale factor. */
//  static const double MARKER_SCALE = 0.01;
//#endif
  /*!
   * \brief Create a Segmenter and associated ROS information.
   *
   * Creates a ROS node handle, subscribes to the relevant topics and servers, and creates services for requesting
   * segmenations.
   */
  ObjectSemanticSegmentation();

  /*!
   * \brief A check for a valid Segmenter.
   *
   * This function will return true if valid segmenation zones were parsed from a YAML config file.
   *
   * \return True if valid segmenation zones were parsed.
   */
//  bool okay() const;

private:
  /*!
   * \brief Callback for the point cloud topic.
   *
   * Saves a copy of the latest point cloud internally.
   *
   * \param pc The current point cloud message.
   */
  void pointCloudCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &pc);

  /*!
 * \brief Callback for the point cloud topic.
 *
 * Saves a copy of the latest point cloud internally.
 *
 * \param pc The current point cloud message.
 */
  void colorImageCallback(const sensor_msgs::ImageConstPtr &color_img);

    /*!
 * \brief Callback for the point cloud topic.
 *
 * Saves a copy of the latest point cloud internally.
 *
 * \param pc The current point cloud message.
 */
  void depthImageCallback(const sensor_msgs::ImageConstPtr &depth_img);

//  /*!
//   * \brief Determine the current zone based on the latest state of the TF tree.
//   *
//   * Checks each segmenation zone criteria based on teh latest state of the TF tree and returns a reference to that
//   * zone. If multiple zones are met, the first is returned. If no valid zone is found, the first zone is returned
//   * and a warning is sent to ROS_WARN.
//   *
//   * \return The zone that matches the current state of the TF tree.
//   */
//  const SegmentationZone &getCurrentZone() const;
//
//  /*!
//   * \brief Callback for the remove object request.
//   *
//   * Remote the object from the segmented object list with a given ID. This will publish both an updated segmented
//   * object list and a marker array with a delete action for the given marker index.
//   *
//   * \param req The request with the index to use.
//   * \param res The empty response (unused).
//   * \return Returns true if a valid index was provided.
//   */
//  bool removeObjectCallback(rail_segmentation::RemoveObject::Request &req,
//      rail_segmentation::RemoveObject::Response &res);
//
  /*!
   * \brief Callback for the clear request.
   *
   * Clears the current segmented object list. This will publish both an empty segmented object list and a marker
   * array with delete actions from the last segmentation request.
   *
   * \param req The empty request (unused).
   * \param res The empty response (unused).
   * \return Will always return true.
   */
  bool clearCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);

  /*!
   * \brief Callback for the main segmentation request.
   *
   * Performs a segmenation with the latest point cloud. This will publish both a segmented object list and a marker
   * array of the resulting segmentation.
   *
   * \param req The empty request (unused).
   * \param res The empty response (unused).
   * \return Returns true if the segmentation was successful.
   */
  bool segmentCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res);

//  bool calculateFeaturesCallback(rail_manipulation_msgs::ProcessSegmentedObjects::Request &req,
//      rail_manipulation_msgs::ProcessSegmentedObjects::Response &res);

//  /*!
//   * \brief Callback for the main segmentation request.
//   *
//   * Performs a segmenation with the latest point cloud. This will publish both a segmented object list and a marker
//   * array of the resulting segmentation.
//   *
//   * \param req The empty request (unused).
//   * \param res The resulting segmented object list.
//   * \return Returns true if the segmentation was successful.
//   */
  bool segmentObjectsCallback(rail_semantic_grasping::SegmentSemanticObjectsRequest &req,
                              rail_semantic_grasping::SegmentSemanticObjectsResponse &res);

  /*!
   * \brief Callback for the semantic segmentation request.
   *
   * Performs a segmenation with the latest point cloud. This will publish both a segmented object list and a marker
   * array of the resulting segmentation.
   *
   * \param objects List for resulting segmented objects.
   */
  bool segmentObjects(rail_semantic_grasping::SemanticObjectList &objects);

  /*!
   * \brief Callback for the geometric segmentation request.
   *
   * Performs an additional segmenation on segmented object. This will publish both a segmented object list and a marker
   * array of the resulting segmentation.
   *
   * \param objects List for resulting segmented objects.
   */
  bool segmentObjectsGeometric(rail_semantic_grasping::SemanticObjectList &objects);

  /*!
   * \brief Extract the handle and the body from an object.
   *
   */
  bool extractHandle(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &object,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &body,
                     pcl::PointCloud<pcl::PointXYZRGB>::Ptr &handle);

  /*!
   * \brief Extract the opening part from an object.
   *
   */
  bool extractOpening(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &object,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &body,
                      pcl::PointCloud<pcl::PointXYZRGB>::Ptr &opening);

//  /*!
//   * \brief Find and remove a surface from the given point cloud.
//   *
//   * Find a surface in the input point cloud and attempt to remove it. The surface must be within the bounds provided
//   * in order to be removed. The resulting point cloud is placed in the output cloud. If no surface is found, no
//   * effect is made to the output cloud and negative infinity is returned.
//   *
//   * \param in The input point cloud.
//   * \param indices_in The indices in the point cloud to consider.
//   * \param z_min The minimum height of a surface to remove.
//   * \param z_max The maximum height of a surface to remove.
//   * \param indices_out The set of points that are not part of the surface.
//   * \param table_out the table as a segmented object, to be published on a separate topic
//   * \return true if a surface was found, false otherwise
//   */
//  bool findSurface(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in, const pcl::IndicesConstPtr &indices_in,
//      const SegmentationZone &zone, const pcl::IndicesPtr &indices_out,
//      rail_manipulation_msgs::SegmentedObject &table_out) const;
//
//  /*!
//   * \brief Find clusters in a point cloud.
//   *
//   * Find the clusters in the given point cloud using euclidean cluster extraction and a KD search tree.
//   *
//   * \param in The point cloud to search for point clouds from.
//   * \param indices_in The indices in the point cloud to consider.
//   * \param clusters The indices of each cluster in the point cloud.
//   */
//  void extractClustersEuclidean(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in, const pcl::IndicesConstPtr &indices_in,
//      std::vector<pcl::PointIndices> &clusters) const;
//
//
//  /*!
//   * \brief Find clusters in a point cloud.
//   *
//   * Find the clusters in the given point cloud using RGB region growing and a KD search tree.
//   *
//   * \param in The point cloud to search for point clouds from.
//   * \param indices_in The indices in the point cloud to consider.
//   * \param clusters The indices of each cluster in the point cloud.
//   */
//  void extractClustersRGB(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in, const pcl::IndicesConstPtr &indices_in,
//                          std::vector<pcl::PointIndices> &clusters) const;
//
//  /*!
//   * \brief Bound a point cloud based on the inverse of a set of conditions.
//   *
//   * Extract a new point cloud based on the inverse of a set of conditions.
//   *
//   * \param in The point cloud to take points from.
//   * \param indices_in The indices in the point cloud to consider.
//   * \param conditions The conditions specifying which points to ignore.
//   * \param indices_out The set of points that pass the condition test.
//   */
//  void inverseBound(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in, const pcl::IndicesConstPtr &indices_in,
//      const pcl::ConditionBase<pcl::PointXYZRGB>::Ptr &conditions, const pcl::IndicesPtr &indices_out) const;
//
//  /*!
//   * \brief Extract a new point cloud based on the given indices.
//   *
//   * Extract a new point cloud from the given indices. The resulting point cloud will be unorganized.
//   *
//   * \param in The point cloud to take points from.
//   * \param indices_in The indices to create a new point cloud from.
//   * \param out The point cloud to fill.
//   */
//  void extract(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in, const pcl::IndicesConstPtr &indices_in,
//      const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out) const;
//
//  /*!
//   * \brief Find the average Z value of the point vector.
//   *
//   * Finds the average Z value of the point vector. An empty vector will return an average of 0.
//   *
//   * \param v The vector of points to average.
//   * \return The average Z value of the provided points.
//   */
//  double averageZ(const std::vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > &v) const;
//
  /*!
   * \brief Create a Marker from the given point cloud.
   *
   * Creates a new Marker message based on the PCL point cloud. The point cloud will first be downsampled.
   *
   * \param pc The PCL point cloud to create a marker for.
   * \return The corresponding marker for the given point cloud.
   */
  visualization_msgs::Marker createMarker(const pcl::PCLPointCloud2::ConstPtr &pc,
      const std::string &marker_namespace) const;

  visualization_msgs::Marker createTextMarker(const std::string &label, const std_msgs::Header &header,
      const geometry_msgs::Point &position, const std::string &marker_namespace) const;
//
//  /*!
//   * \brief Create a cropped image of the segmented object.
//   *
//   * Creates a new ROS image based on the cropped segmented object.
//   *
//   * \param in The original organized point cloud.
//   * \param cluster The indicies of the current cluster in the point cloud.
//   * \return The corresponding image for the given cluster.
//   */
//  sensor_msgs::Image createImage(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in,
//      const pcl::PointIndices &cluster) const;

  std::string point_cloud_topic_, camera_color_topic_, camera_depth_topic_;
  /*! The debug, okay check, first point cloud, and color segmentation flags. */
  bool debug_, okay_, first_pc_in_, use_color_, first_color_in_, first_depth_in_;
  /*! Cluster parameters. */
  int min_cluster_size_, max_cluster_size_;
  /*! Mutex for locking on the point cloud and current messages. */
  boost::mutex pc_mutex_, msg_mutex_, color_img_mutex_, depth_img_mutex_;
  /*! List of segmentation zones. */
  // std::vector<SegmentationZone> zones_;
  /*! Flag for cropping the point cloud before table detection or after */
  bool crop_first_;
  /*! Flag for labeling cluster markers with their index */
  bool label_markers_;
  /*! Settable euclidean distance tolerance for including a point in a cluster */
  double cluster_tolerance_;
  /*! Minimal number of pixels for including an affordance */
  int min_affordance_pixels_;
  /*! Minimal number of pixels for including an affordance */
  std::string geometric_segmentation_frame_;
  /*! Parameters for cylinder segmentation */
  double cylinder_segmentation_normal_distance_weight_, cylinder_segmentation_distance_threshold_ratio_,
         cylinder_segmentation_cluster_tolerance_;
  int cylinder_segmentation_max_iteration_, cylinder_segmentation_normal_k_, cylinder_segmentation_min_cluster_size_,
      cylinder_segmentation_max_cluster_size_;


  /*! The global and private ROS node handles. */
  ros::NodeHandle node_, private_node_;

  ros::ServiceClient detect_part_affordances_client_, segment_objects_client_, segment_objects_from_point_cloud_client_,
                     calculate_features_client_;

  /*! Services advertised by this node */
  ros::ServiceServer segment_srv_, segment_objects_srv_, clear_srv_, remove_object_srv_, calculate_features_srv_;
  /*! Publishers used in the node. */
  ros::Publisher semantic_objects_pub_, table_pub_, markers_pub_, table_marker_pub_, debug_pc_pub_, debug_pc_pub_2_,
                 debug_pc_pub_3_, debug_img_pub_, debug_pose_pub_;
  /*! Subscribers used in the node. */
  ros::Subscriber point_cloud_sub_, color_image_sub_, depth_image_sub_;
  /*! Main transform listener. */
  tf::TransformListener tf_;
  /*! The transform tree buffer for the tf2 listener. */
  tf2_ros::Buffer tf_buffer_;
  /*! The buffered trasnform client. */
  tf2_ros::TransformListener tf2_;

  /*! Latest point cloud. */
  pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr pc_;
  /*! Latest color image. */
  sensor_msgs::ImageConstPtr color_img_;
  /*! Latest depth image. */
  sensor_msgs::ImageConstPtr depth_img_;
  /*! Current object list. */
  rail_semantic_grasping::SemanticObjectList object_list_;
  /*! Current table object. */
  //rail_manipulation_msgs::SegmentedObject table_;
  /*! Current marker array. */
  visualization_msgs::MarkerArray markers_;
  /*! Current marker label array (only used if label_markers_ is true). */
  visualization_msgs::MarkerArray text_markers_;
};

}
}

//Eigen::Vector3f RGB2Lab (const Eigen::Vector3f& colorRGB);

#endif
