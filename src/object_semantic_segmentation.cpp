/*!
 * \file Segmenter.cpp
 * \brief The main segmentation node object.
 *
 * The segmenter is responsible for segmenting clusters from a point cloud topic. Visualization and data latched topics
 * are published after each request. A persistent array of objects is maintained internally.
 *
 * \author Russell Toris, WPI - russell.toris@gmail.com
 * \author David Kent, GT - dekent@gatech.edu
 * \date January 12, 2016
 */

// RAIL Segmentation
#include "rail_semantic_grasping/object_semantic_segmentation.h"

using namespace std;
using namespace rail::semantic_grasping;

//constant definitions (to use in functions with reference parameters, e.g. param())
//const bool ObjectSemanticSegmentation::DEFAULT_DEBUG;
//const int ObjectSemanticSegmentation::DEFAULT_MIN_CLUSTER_SIZE;
//const int ObjectSemanticSegmentation::DEFAULT_MAX_CLUSTER_SIZE;
//const double ObjectSemanticSegmentation::CLUSTER_TOLERANCE;

ObjectSemanticSegmentation::ObjectSemanticSegmentation() : private_node_("~"), tf2_(tf_buffer_)
{
  // define affordance labels
  idx_to_affordance[0] = "background";
  idx_to_affordance[1] = "contain";
  idx_to_affordance[2] = "cut";
  idx_to_affordance[3] = "display";
  idx_to_affordance[4] = "engine";
  idx_to_affordance[5] = "grasp";
  idx_to_affordance[6] = "hit";
  idx_to_affordance[7] = "pound";
  idx_to_affordance[8] = "support";
  idx_to_affordance[9] = "w_grasp";

  // define object class labels
  idx_to_object_class[0] = "background";
  idx_to_object_class[1] = "bowl";
  idx_to_object_class[2] = "tvm";
  idx_to_object_class[3] = "pan";
  idx_to_object_class[4] = "hammer";
  idx_to_object_class[5] = "knife";
  idx_to_object_class[6] = "cup";
  idx_to_object_class[7] = "drill";
  idx_to_object_class[8] = "racket";
  idx_to_object_class[9] = "spatula";
  idx_to_object_class[10] = "bottle";

  // flag for the first point cloud coming in
  first_pc_in_ = false;

  // set defaults
  //string point_cloud_topic("/camera/depth_registered/points");
//  string zones_file(ros::package::getPath("rail_segmentation") + "/config/zones.yaml");

  // grab any parameters we need
//  private_node_.param("debug", debug_, DEFAULT_DEBUG);
//  private_node_.param("min_cluster_size", min_cluster_size_, DEFAULT_MIN_CLUSTER_SIZE);
//  private_node_.param("max_cluster_size", max_cluster_size_, DEFAULT_MAX_CLUSTER_SIZE);
//  private_node_.param("cluster_tolerance", cluster_tolerance_, CLUSTER_TOLERANCE);
//  private_node_.param("use_color", use_color_, false);
//  private_node_.param("crop_first", crop_first_, false);
  private_node_.param<string>("point_cloud_topic", point_cloud_topic_, "/head_camera/depth_registered/points");
  private_node_.param<string>("camera_color_topic", camera_color_topic_, "/head_camera/rgb/image_rect_color");
  private_node_.param<string>("camera_depth_topic", camera_depth_topic_, "/head_camera/depth/image_rect");
  private_node_.param("label_markers", label_markers_, false);
//  private_node_.getParam("point_cloud_topic", point_cloud_topic);
//  private_node_.getParam("zones_config", zones_file);
  private_node_.param("min_affordance_pixels", min_affordance_pixels_, 0);
  private_node_.param<string>("geometric_segmentation_frame", geometric_segmentation_frame_, "base_link");
  private_node_.param("cylinder_segmentation_normal_k", cylinder_segmentation_normal_k_, 200);
  private_node_.param("cylinder_segmentation_normal_distance_weight", cylinder_segmentation_normal_distance_weight_, 0.1);
  private_node_.param("cylinder_segmentation_max_iteration", cylinder_segmentation_max_iteration_, 10000);
  private_node_.param("cylinder_segmentation_distance_threshold_ratio", cylinder_segmentation_distance_threshold_ratio_, 0.8);
  private_node_.param("cylinder_segmentation_cluster_tolerance", cylinder_segmentation_cluster_tolerance_, 0.01);
  private_node_.param("cylinder_segmentation_min_cluster_size", cylinder_segmentation_min_cluster_size_, 30);
  private_node_.param("cylinder_segmentation_max_cluster_size", cylinder_segmentation_max_cluster_size_, 10000);

  // setup publishers/subscribers we need
  segment_srv_ = private_node_.advertiseService("segment", &ObjectSemanticSegmentation::segmentCallback, this);
  segment_objects_srv_ = private_node_.advertiseService("segment_objects", &ObjectSemanticSegmentation::segmentObjectsCallback, this);
  clear_srv_ = private_node_.advertiseService("clear", &ObjectSemanticSegmentation::clearCallback, this);
//  remove_object_srv_ = private_node_.advertiseService("remove_object", &ObjectSemanticSegmentation::removeObjectCallback, this);
//  calculate_features_srv_ = private_node_.advertiseService("calculate_features", &ObjectSemanticSegmentation::calculateFeaturesCallback,
//      this);
  semantic_objects_pub_ = private_node_.advertise<rail_semantic_grasping::SemanticObjectList>(
      "semantic_objects", 1, true);
//  table_pub_ = private_node_.advertise<rail_manipulation_msgs::SegmentedObject>(
//      "segmented_table", 1, true
//  );
  markers_pub_ = private_node_.advertise<visualization_msgs::MarkerArray>("markers", 1, true);
//  table_marker_pub_ = private_node_.advertise<visualization_msgs::Marker>("table_marker", 1, true);
  point_cloud_sub_ = node_.subscribe(point_cloud_topic_, 1, &ObjectSemanticSegmentation::pointCloudCallback, this);
  color_image_sub_ = node_.subscribe(camera_color_topic_, 1, &ObjectSemanticSegmentation::colorImageCallback, this);
  depth_image_sub_ = node_.subscribe(camera_depth_topic_, 1, &ObjectSemanticSegmentation::depthImageCallback, this);

  detect_part_affordances_client_ =
      node_.serviceClient<rail_part_affordance_detection::DetectAffordances>("rail_part_affordance_detection/detect");

  segment_objects_client_ =
      node_.serviceClient<rail_manipulation_msgs::SegmentObjects>("rail_segmentation/segment_objects");

  segment_objects_from_point_cloud_client_ =
      node_.serviceClient<rail_manipulation_msgs::SegmentObjectsFromPointCloud>("rail_segmentation/segment_objects_from_point_cloud");

  calculate_features_client_ =
      node_.serviceClient<rail_manipulation_msgs::ProcessSegmentedObjects>("rail_segmentation/calculate_features");

  debug_pc_pub_ = private_node_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("debug_pc", 1, true);
  debug_pc_pub_2_ = private_node_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("debug_pc_2", 1, true);
  debug_pc_pub_3_ = private_node_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("debug_pc_3", 1, true);
  debug_pose_pub_ = private_node_.advertise<geometry_msgs::PoseStamped>("debug_pose", 1, true);
  debug_img_pub_ = private_node_.advertise<sensor_msgs::Image>("debug_img", 1, true);
//  // setup a debug publisher if we need it
//  if (debug_)
//  {
//    debug_pc_pub_ = private_node_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("debug_pc", 1, true);
//    debug_img_pub_ = private_node_.advertise<sensor_msgs::Image>("debug_img", 1, true);
//  }

//  // check the YAML version
//#ifdef YAMLCPP_GT_0_5_0
//  // parse the segmentation zones
//  YAML::Node zones_config = YAML::LoadFile(zones_file);
//  for (size_t i = 0; i < zones_config.size(); i++)
//  {
//    YAML::Node cur = zones_config[i];
//    // create a zone with the frame ID information
//    SegmentationZone zone(cur["name"].as<string>(), cur["parent_frame_id"].as<string>(),
//                          cur["child_frame_id"].as<string>(), cur["bounding_frame_id"].as<string>(),
//                          cur["segmentation_frame_id"].as<string>());
//
//    // check for the remove surface flag
//    if (cur["remove_surface"].IsDefined())
//    {
//      zone.setRemoveSurface(cur["remove_surface"].as<bool>());
//    }
//
//    // check for the remove surface flag
//    if (cur["require_surface"].IsDefined())
//    {
//      zone.setRequireSurface(cur["require_surface"].as<bool>());
//    }
//
//    // check for any set limits
//    if (cur["roll_min"].IsDefined())
//    {
//      zone.setRollMin(cur["roll_min"].as<double>());
//    }
//    if (cur["roll_max"].IsDefined())
//    {
//      zone.setRollMax(cur["roll_max"].as<double>());
//    }
//    if (cur["pitch_min"].IsDefined())
//    {
//      zone.setPitchMin(cur["pitch_min"].as<double>());
//    }
//    if (cur["pitch_max"].IsDefined())
//    {
//      zone.setPitchMax(cur["pitch_max"].as<double>());
//    }
//    if (cur["yaw_min"].IsDefined())
//    {
//      zone.setYawMin(cur["yaw_min"].as<double>());
//    }
//    if (cur["yaw_max"].IsDefined())
//    {
//      zone.setYawMax(cur["yaw_max"].as<double>());
//    }
//    if (cur["x_min"].IsDefined())
//    {
//      zone.setXMin(cur["x_min"].as<double>());
//    }
//    if (cur["x_max"].IsDefined())
//    {
//      zone.setXMax(cur["x_max"].as<double>());
//    }
//    if (cur["y_min"].IsDefined())
//    {
//      zone.setYMin(cur["y_min"].as<double>());
//    }
//    if (cur["y_max"].IsDefined())
//    {
//      zone.setYMax(cur["y_max"].as<double>());
//    }
//    if (cur["z_min"].IsDefined())
//    {
//      zone.setZMin(cur["z_min"].as<double>());
//    }
//    if (cur["z_max"].IsDefined())
//    {
//      zone.setZMax(cur["z_max"].as<double>());
//    }
//
//    zones_.push_back(zone);
//  }
//#else
//  // parse the segmentation zones
//  ifstream fin(zones_file.c_str());
//  YAML::Parser zones_parser(fin);
//  YAML::Node zones_config;
//  zones_parser.GetNextDocument(zones_config);
//  for (size_t i = 0; i < zones_config.size(); i++)
//  {
//    // parse the required information
//    string name, parent_frame_id, child_frame_id, bounding_frame_id, segmentation_frame_id;
//    zones_config[i]["name"] >> name;
//    zones_config[i]["parent_frame_id"] >> parent_frame_id;
//    zones_config[i]["child_frame_id"] >> child_frame_id;
//    zones_config[i]["bounding_frame_id"] >> bounding_frame_id;
//    zones_config[i]["segmentation_frame_id"] >> segmentation_frame_id;
//
//    // create a zone with the frame ID information
//    SegmentationZone zone(name, parent_frame_id, child_frame_id, bounding_frame_id, segmentation_frame_id);
//
//    // check for the remove surface flag
//    if (zones_config[i].FindValue("remove_surface") != NULL)
//    {
//      bool remove_surface;
//      zones_config[i]["remove_surface"] >> remove_surface;
//      zone.setRemoveSurface(remove_surface);
//    }
//    if (zones_config[i].FindValue("require_surface") != NULL)
//    {
//      bool require_surface;
//      zones_config[i]["require_surface"] >> require_surface;
//      zone.setRequireSurface(require_surface);
//    }
//
//    // check for any set limits
//    if (zones_config[i].FindValue("roll_min") != NULL)
//    {
//      double roll_min;
//      zones_config[i]["roll_min"] >> roll_min;
//      zone.setRollMin(roll_min);
//    }
//    if (zones_config[i].FindValue("roll_max") != NULL)
//    {
//      double roll_max;
//      zones_config[i]["roll_max"] >> roll_max;
//      zone.setRollMax(roll_max);
//    }
//    if (zones_config[i].FindValue("pitch_min") != NULL)
//    {
//      double pitch_min;
//      zones_config[i]["pitch_min"] >> pitch_min;
//      zone.setPitchMin(pitch_min);
//    }
//    if (zones_config[i].FindValue("pitch_max") != NULL)
//    {
//      double pitch_max;
//      zones_config[i]["pitch_max"] >> pitch_max;
//      zone.setPitchMax(pitch_max);
//    }
//    if (zones_config[i].FindValue("yaw_min") != NULL)
//    {
//      double yaw_min;
//      zones_config[i]["yaw_min"] >> yaw_min;
//      zone.setYawMin(yaw_min);
//    }
//    if (zones_config[i].FindValue("yaw_max") != NULL)
//    {
//      double yaw_max;
//      zones_config[i]["yaw_max"] >> yaw_max;
//      zone.setYawMax(yaw_max);
//    }
//    if (zones_config[i].FindValue("x_min") != NULL)
//    {
//      double x_min;
//      zones_config[i]["x_min"] >> x_min;
//      zone.setXMin(x_min);
//    }
//    if (zones_config[i].FindValue("x_max") != NULL)
//    {
//      double x_max;
//      zones_config[i]["x_max"] >> x_max;
//      zone.setXMax(x_max);
//    }
//    if (zones_config[i].FindValue("y_min") != NULL)
//    {
//      double y_min;
//      zones_config[i]["y_min"] >> y_min;
//      zone.setYMin(y_min);
//    }
//    if (zones_config[i].FindValue("y_max") != NULL)
//    {
//      double y_max;
//      zones_config[i]["y_max"] >> y_max;
//      zone.setYMax(y_max);
//    }
//    if (zones_config[i].FindValue("z_min") != NULL)
//    {
//      double z_min;
//      zones_config[i]["z_min"] >> z_min;
//      zone.setZMin(z_min);
//    }
//    if (zones_config[i].FindValue("z_max") != NULL)
//    {
//      double z_max;
//      zones_config[i]["z_max"] >> z_max;
//      zone.setZMax(z_max);
//    }
//
//    zones_.push_back(zone);
//  }
//#endif

  // check how many zones we have
//  if (zones_.size() > 0)
//  {
//    ROS_INFO("%d segmenation zone(s) parsed.", (int) zones_.size());
//    ROS_INFO("ObjectSemanticSegmentation Successfully Initialized");
//    okay_ = true;
//  } else
//  {
//    ROS_ERROR("No valid segmenation zones defined. Check %s.", zones_file.c_str());
//    okay_ = false;
//  }
}

//bool ObjectSemanticSegmentation::okay() const
//{
//  return okay_;
//}

void ObjectSemanticSegmentation::pointCloudCallback(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &pc)
{
  // lock for the point cloud
  boost::mutex::scoped_lock lock(pc_mutex_);
  // simply store the latest point cloud
  first_pc_in_ = true;
  pc_ = pc;
}

void ObjectSemanticSegmentation::colorImageCallback(const sensor_msgs::ImageConstPtr &color_img)
{
  boost::mutex::scoped_lock lock(color_img_mutex_);
  first_color_in_ = true;
  color_img_ = color_img;
}

void ObjectSemanticSegmentation::depthImageCallback(const sensor_msgs::ImageConstPtr &depth_img)
{
  boost::mutex::scoped_lock lock(depth_img_mutex_);
  first_depth_in_ = true;
  depth_img_ = depth_img;
}

//const SegmentationZone &ObjectSemanticSegmentation::getCurrentZone() const
//{
//  // check each zone
//  for (size_t i = 0; i < zones_.size(); i++)
//  {
//    // get the current TF information
//    geometry_msgs::TransformStamped tf = tf_buffer_.lookupTransform(zones_[i].getParentFrameID(),
//                                                                    zones_[i].getChildFrameID(), ros::Time(0));
//
//    // convert to a Matrix3x3 to get RPY
//    tf2::Matrix3x3 mat(tf2::Quaternion(tf.transform.rotation.x, tf.transform.rotation.y, tf.transform.rotation.z,
//                                       tf.transform.rotation.w));
//    double roll, pitch, yaw;
//    mat.getRPY(roll, pitch, yaw);
//
//    // check if all the bounds meet
//    if (roll >= zones_[i].getRollMin() && pitch >= zones_[i].getPitchMin() && yaw >= zones_[i].getYawMin() &&
//        roll <= zones_[i].getRollMax() && pitch <= zones_[i].getPitchMax() && yaw <= zones_[i].getYawMax())
//    {
//      return zones_[i];
//    }
//  }
//
//  ROS_WARN("Current state not in a valid segmentation zone. Defaulting to first zone.");
//  return zones_[0];
//}
//
//bool ObjectSemanticSegmentation::removeObjectCallback(rail_segmentation::RemoveObject::Request &req,
//    rail_segmentation::RemoveObject::Response &res)
//{
//  // lock for the messages
//  boost::mutex::scoped_lock lock(msg_mutex_);
//  // check the index
//  if (req.index < object_list_.objects.size() && req.index < markers_.markers.size())
//  {
//    // remove
//    object_list_.objects.erase(object_list_.objects.begin() + req.index);
//    // set header information
//    object_list_.header.seq++;
//    object_list_.header.stamp = ros::Time::now();
//    object_list_.cleared = false;
//    // republish
//    segmented_objects_pub_.publish(object_list_);
//    // delete marker
//    markers_.markers[req.index].action = visualization_msgs::Marker::DELETE;
//    if (label_markers_)
//    {
//      text_markers_.markers[req.index].action = visualization_msgs::Marker::DELETE;
//    }
//
//    if (label_markers_)
//    {
//      visualization_msgs::MarkerArray marker_list;
//      marker_list.markers.reserve(markers_.markers.size() + text_markers_.markers.size());
//      marker_list.markers.insert(marker_list.markers.end(), markers_.markers.begin(), markers_.markers.end());
//      marker_list.markers.insert(marker_list.markers.end(), text_markers_.markers.begin(), text_markers_.markers.end());
//      markers_pub_.publish(marker_list);
//    }
//    else
//    {
//      markers_pub_.publish(markers_);
//    }
//
//    if (label_markers_)
//    {
//      text_markers_.markers.erase(text_markers_.markers.begin() + req.index);
//    }
//    markers_.markers.erase(markers_.markers.begin() + req.index);
//    return true;
//  } else
//  {
//    ROS_ERROR("Attempted to remove index %d from list of size %ld.", req.index, object_list_.objects.size());
//    return false;
//  }
//}

bool ObjectSemanticSegmentation::clearCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
  // lock for the messages
  boost::mutex::scoped_lock lock(msg_mutex_);
  // empty the list
  object_list_.objects.clear();
  object_list_.cleared = true;
  // set header information
  object_list_.header.seq++;
  object_list_.header.stamp = ros::Time::now();
  // republish
  semantic_objects_pub_.publish(object_list_);
  // delete markers
  for (size_t i = 0; i < markers_.markers.size(); i++)
  {
    markers_.markers[i].action = visualization_msgs::Marker::DELETE;
  }
  if (label_markers_)
  {
    for (size_t i = 0; i < text_markers_.markers.size(); i++)
    {
      text_markers_.markers[i].action = visualization_msgs::Marker::DELETE;
    }
  }
  if (label_markers_)
  {
    visualization_msgs::MarkerArray marker_list;
    marker_list.markers.reserve(markers_.markers.size() + text_markers_.markers.size());
    marker_list.markers.insert(marker_list.markers.end(), markers_.markers.begin(), markers_.markers.end());
    marker_list.markers.insert(marker_list.markers.end(), text_markers_.markers.begin(), text_markers_.markers.end());
    markers_pub_.publish(marker_list);
  } else
  {
    markers_pub_.publish(markers_);
  }
  markers_.markers.clear();
  if (label_markers_)
  {
    text_markers_.markers.clear();
  }
  return true;
}

bool ObjectSemanticSegmentation::segmentCallback(std_srvs::Empty::Request &req, std_srvs::Empty::Response &res)
{
  rail_semantic_grasping::SemanticObjectList objects;
  return segmentObjects(objects);
}

bool ObjectSemanticSegmentation::segmentObjectsCallback(rail_semantic_grasping::SegmentSemanticObjectsRequest &req,
                                                        rail_semantic_grasping::SegmentSemanticObjectsResponse &res)
{
  return segmentObjects(res.semantic_objects);
}

bool ObjectSemanticSegmentation::segmentObjects(rail_semantic_grasping::SemanticObjectList &objects)
{
  // check if we have a point cloud first
  {
    boost::mutex::scoped_lock lock(pc_mutex_);
    if (!first_pc_in_)
    {
      ROS_WARN("No point cloud received yet. Ignoring segmentation request.");
      return false;
    }
  }
  sensor_msgs::ImagePtr rgb_img(new sensor_msgs::Image);
  {
    boost::mutex::scoped_lock lock(color_img_mutex_);
    if (!first_color_in_)
    {
      ROS_WARN("No color image received yet. Ignoring segmentation request.");
      return false;
    } else
    {
      *rgb_img = *color_img_;
    }
  }
  sensor_msgs::ImagePtr dep_img(new sensor_msgs::Image);
  {
    boost::mutex::scoped_lock lock(depth_img_mutex_);
    if (!first_depth_in_)
    {
      ROS_WARN("No depth image received yet. Ignoring segmentation request.");
      return false;
    } else
    {
      *dep_img = *depth_img_;
    }
  }

  // clear the objects first
  std_srvs::Empty empty;
  this->clearCallback(empty.request, empty.response);

  // call part affordance detection
  rail_part_affordance_detection::DetectAffordances detect_affordances_srv;
  if (!detect_part_affordances_client_.call(detect_affordances_srv))
  {
    ROS_INFO("Could not detect part affordances! Aborting.");
    return false;
  } else if (detect_affordances_srv.response.object_part_affordance_list.empty())
  {
    ROS_INFO("No affordance detected! Aborting.");
    return false;
  } else
  {
    ROS_INFO("affordance detection succeeded");
  }

  // Important: doesn't have significant effect, so just copy for now
  // transform input point cloud (depth frame) to the frame that the affordance detection uses (rgb frame)
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
  {
    boost::mutex::scoped_lock lock(pc_mutex_);
    pcl::copyPointCloud(*pc_, *transformed_pc);
  }
//  std::string affordance_frame_id;
//  if (!detect_affordances_srv.response.object_part_affordance_list.empty())
//  {
//    {
//      boost::mutex::scoped_lock lock(pc_mutex_);
//      // perform the copy/transform using TF
//      affordance_frame_id = detect_affordances_srv.response.object_part_affordance_list[0].header.frame_id;
//      pcl_ros::transformPointCloud(affordance_frame_id,
//                                   ros::Time(0), *pc_, pc_->header.frame_id, *transformed_pc, tf_);
//      transformed_pc->header.frame_id = affordance_frame_id;
//      transformed_pc->header.seq = pc_->header.seq;
//      transformed_pc->header.stamp = pc_->header.stamp;
//    }
//  }

  // call rail_segmentation and get the list of objects
  rail_manipulation_msgs::SegmentObjectsFromPointCloud segment_objects_srv;
  sensor_msgs::PointCloud2 provide_pc;
  pcl::toROSMsg(*transformed_pc, provide_pc);
  ROS_INFO("check pc input %d", provide_pc.width);
  segment_objects_srv.request.point_cloud = provide_pc;
  if (!segment_objects_from_point_cloud_client_.call(segment_objects_srv))
  {
    ROS_INFO("Could not segment objects! Aborting.");
    return false;
  } else
  {
    ROS_INFO("object segmentation succeeded");
  }

  // match detected objects (from part affordance detection) with segmented objects (from segmentation)
  std::map<int, int> detected_to_segmented_object;
  std::map<int, int> match_distance;
  for (size_t i = 0; i < segment_objects_srv.response.segmented_objects.objects.size(); ++i)
  {
    // compute center of the segmented object
    vector<int> segmented_indices = segment_objects_srv.response.segmented_objects.objects[i].image_indices;
    int row_sum = 0;
    int col_sum = 0;
    for (size_t pi = 0; pi < segmented_indices.size(); ++pi)
    {
      int row = segmented_indices[pi] / transformed_pc->width;
      int col = segmented_indices[pi] - (row * transformed_pc->width);
      row_sum += row;
      col_sum += col;
    }
//    ROS_INFO("sums: %d, %d", row_sum, col_sum);
    int row_avg = int(row_sum / double(segmented_indices.size()));
    int col_avg = int(col_sum / double(segmented_indices.size()));
//    ROS_INFO("size %zu", segmented_indices.size());
//    ROS_INFO("width %d", transformed_pc->width);
//    ROS_INFO("Segmented object No.%zu has image coordinate %d, %d", i, col_avg, row_avg);

    for (size_t j = 0; j < detect_affordances_srv.response.object_part_affordance_list.size(); ++j)
    {
      uint16_t col_min = detect_affordances_srv.response.object_part_affordance_list[j].bounding_box[0];
      uint16_t row_min = detect_affordances_srv.response.object_part_affordance_list[j].bounding_box[1];
      uint16_t col_max = detect_affordances_srv.response.object_part_affordance_list[j].bounding_box[2];
      uint16_t row_max = detect_affordances_srv.response.object_part_affordance_list[j].bounding_box[3];

      // check if the center of the segmented object is in the bounding box of the detected object
      if (row_avg < row_max && row_avg > row_min && col_avg < col_max && col_avg > col_min)
      {
        // match the closest pair of detected bounding box center and segmented object center
        int row_center = ((int) row_max + (int) row_min) / 2;
        int col_center = ((int) col_max + (int) col_min) / 2;
        int distance =
            (row_avg - row_center) * (row_avg - row_center) + (col_avg - col_center) * (col_avg - col_center);
        if (detected_to_segmented_object.count(j) == 1)
        {
          if (distance < match_distance[j])
          {
            detected_to_segmented_object[j] = i;
            match_distance[j] = distance;
          }
        } else
        {
          detected_to_segmented_object[j] = i;
          match_distance[j] = distance;
        }
      }
    }
  }
  if (detected_to_segmented_object.empty())
  {
    ROS_INFO("Cannot match detected object parts with segmented object! Aborting.");
    return false;
  }
  for (map<int,int>::iterator it=detected_to_segmented_object.begin(); it!=detected_to_segmented_object.end(); ++it)
  {
    ROS_INFO("Detected object %d matches to segmented object %d", it->first, it->second);
  }

  // segment each object into parts based on affordances of parts
  // lock for the messages
  boost::mutex::scoped_lock lock(msg_mutex_);
  for (size_t oi = 0; oi < detect_affordances_srv.response.object_part_affordance_list.size(); oi++)
  {
    // match the detected object to the segmented object
    vector<int> segmented_indices;
    if (detected_to_segmented_object.count(oi) == 1)
    {
      segmented_indices = segment_objects_srv.response.segmented_objects.objects[detected_to_segmented_object.at(oi)].image_indices;
    }

    rail_part_affordance_detection::ObjectPartAffordance object_affordances;
    object_affordances = detect_affordances_srv.response.object_part_affordance_list[oi];
    std::string object_class = idx_to_object_class[object_affordances.object_class];
    ROS_INFO("");
    ROS_INFO("Detected object No.%zu is %s", oi, object_class.c_str());

    // iterate through affordance mask of the whole image and find number of unique affordances and their numbers of
    // occurances
    map<int, int> unique_affordances;
    for (size_t pi = 0; pi < object_affordances.affordance_mask.size(); pi++)
    {
      if (object_affordances.affordance_mask[pi] == 0) continue;
      unique_affordances[object_affordances.affordance_mask[pi]]++;
    }

    // construct a semantic object
    rail_semantic_grasping::SemanticObject semantic_object;
    semantic_object.name = object_class;
    // also combine pc of parts in the loop
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr combined_object_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    vector<int> combined_object_image_indices;

    // for each affordance, get the corresponding part
    for (map<int, int>::iterator aff_it=unique_affordances.begin(); aff_it!=unique_affordances.end(); ++aff_it)
    {
      int aff_idx = aff_it->first;
      // ignore background
      if (aff_idx == 0)
      {
        continue;
      }
      // filter out affordances that have small number of occurances
      if (aff_it->second < min_affordance_pixels_)
      {
        continue;
      }

      rail_semantic_grasping::SemanticPart semantic_part;
      string cluster_affordance = idx_to_affordance[aff_idx];
      ROS_INFO("Affordance %s has %d supports", cluster_affordance.c_str(), aff_it->second);

      // extract the point cloud based on the segmentation mask
      // create unorganized point cloud
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
      //ROS_INFO("width %d and height %d and is_dense %d", transformed_pc->width, transformed_pc->height, transformed_pc->is_dense);
      for (size_t pi = 0; pi < object_affordances.affordance_mask.size(); pi++)
      {
        if (object_affordances.affordance_mask[pi] == aff_idx)
        {
          if (pcl_isfinite(transformed_pc->points[pi].x) & pcl_isfinite(transformed_pc->points[pi].y) & pcl_isfinite(transformed_pc->points[pi].z))
          {
            // use the segmented object to filter out points
            if (find(segmented_indices.begin(), segmented_indices.end(), pi) != segmented_indices.end())
            {
              cluster->points.push_back(transformed_pc->points[pi]);
              combined_object_image_indices.push_back(pi);
            }
          }
        }
      }

      if (cluster->points.empty())
      {
        continue;
      }

      cluster->width = cluster->points.size();
      cluster->height = 1;
      cluster->is_dense = true;
      cluster->header.frame_id = transformed_pc->header.frame_id;

      // check if we need to transform to a different frame
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
      if (cluster->header.frame_id != geometric_segmentation_frame_)
      {
        pcl_ros::transformPointCloud(geometric_segmentation_frame_, ros::Time(0), *cluster, cluster->header.frame_id,
                                     *transformed_cluster, tf_);
        transformed_cluster->header.frame_id = geometric_segmentation_frame_;
        transformed_cluster->header.seq = cluster->header.seq;
        transformed_cluster->header.stamp = cluster->header.stamp;
      } else
      {
        pcl::copyPointCloud(*cluster, *transformed_cluster);
      }

      // add pc of this part to the combined object
      // the += operator should take care of width, height, is_dense, and header
      *combined_object_pc += *transformed_cluster;

      sensor_msgs::PointCloud2 part_pc;
      pcl::toROSMsg(*transformed_cluster, part_pc);
      semantic_part.point_cloud = part_pc;
      // semantic_part.image_indices = cluster_indices;
      semantic_part.affordance = cluster_affordance;
      semantic_object.parts.push_back(semantic_part);
    }

    // Combine semantic parts to get the semantic object
    combined_object_pc->header.frame_id = geometric_segmentation_frame_; // need to be set, otherwise will be empty
    // debug_pc_pub_.publish(combined_object_pc);
    sensor_msgs::PointCloud2 combined_object_pc_msg;
    pcl::toROSMsg(*combined_object_pc, combined_object_pc_msg);
    semantic_object.point_cloud = combined_object_pc_msg;

    semantic_object.image_indices = combined_object_image_indices;
    semantic_object.color_image = *rgb_img;
    semantic_object.depth_image = *dep_img;

    objects.objects.push_back(semantic_object);
  }

  // ToDo: process cluster
  // 1. cluster based on connected region
  // 2. detect handle
  // 3. detect opening

  // furthur segment the object based on geometric features
  this->segmentObjectsGeometric(objects);

  // compute features
  for (size_t oi = 0; oi < objects.objects.size(); ++oi)
  {
    string object_class = objects.objects[oi].name;
    for (size_t pi = 0; pi < objects.objects[oi].parts.size(); ++pi)
    {
      string part_affordance = objects.objects[oi].parts[pi].affordance;
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr part_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::fromROSMsg(objects.objects[oi].parts[pi].point_cloud, *part_pc);
      // ToDo: after parts are segmented, compute other features. also make the computation a seperate function.
      // ToDo: This can be achieved by calling calculate_features service in rail_segmentation
      // compute centroid of part
      Eigen::Vector4f centroid;
      pcl::compute3DCentroid(*part_pc, centroid);
      objects.objects[oi].parts[pi].centroid.x = centroid[0];
      objects.objects[oi].parts[pi].centroid.y = centroid[1];
      objects.objects[oi].parts[pi].centroid.z = centroid[2];

      // create visualization marker
      pcl::PCLPointCloud2::Ptr converted(new pcl::PCLPointCloud2);
      pcl::toPCLPointCloud2(*part_pc, *converted);
      // set namespace (e.g., obj_0/cup/contain)
      stringstream marker_ns;
      marker_ns << "obj_" << oi << "/" << object_class << "/" << part_affordance;

      visualization_msgs::Marker marker, text_marker;
      marker = this->createMarker(converted, marker_ns.str());
      markers_.markers.push_back(marker);
      text_marker = this->createTextMarker(part_affordance, marker.header, objects.objects[oi].parts[pi].centroid, marker_ns.str());
      text_markers_.markers.push_back(text_marker);
      objects.objects[oi].parts[pi].marker = marker;
      objects.objects[oi].parts[pi].text_marker = text_marker;
    }

    // compute features
    rail_manipulation_msgs::SegmentedObject input_object;
    input_object.point_cloud = objects.objects[oi].point_cloud;
    rail_manipulation_msgs::ProcessSegmentedObjects process_objects;
    process_objects.request.segmented_objects.objects.push_back(input_object);
    if (!calculate_features_client_.call(process_objects))
    {
      ROS_INFO("Could not call service to calculate segmented object features!");
      return false;
    }
    objects.objects[oi].centroid = process_objects.response.segmented_objects.objects[0].centroid;
    objects.objects[oi].center = process_objects.response.segmented_objects.objects[0].center;
    objects.objects[oi].bounding_volume = process_objects.response.segmented_objects.objects[0].bounding_volume;
    objects.objects[oi].width = process_objects.response.segmented_objects.objects[0].width;
    objects.objects[oi].depth = process_objects.response.segmented_objects.objects[0].depth;
    objects.objects[oi].height = process_objects.response.segmented_objects.objects[0].height;
    objects.objects[oi].rgb = process_objects.response.segmented_objects.objects[0].rgb;
    objects.objects[oi].cielab = process_objects.response.segmented_objects.objects[0].cielab;
    objects.objects[oi].orientation = process_objects.response.segmented_objects.objects[0].orientation;
    objects.objects[oi].marker = process_objects.response.segmented_objects.objects[0].marker;
  }

  // publish markers for each object
  if (label_markers_)
  {
    visualization_msgs::MarkerArray marker_list;
    marker_list.markers.reserve(markers_.markers.size() + text_markers_.markers.size());
    marker_list.markers.insert(marker_list.markers.end(), markers_.markers.begin(), markers_.markers.end());
    marker_list.markers.insert(marker_list.markers.end(), text_markers_.markers.begin(), text_markers_.markers.end());
    markers_pub_.publish(marker_list);
  } else
  {
    markers_pub_.publish(markers_);
  }

  // collect object part material
  for (size_t oi = 0; oi < objects.objects.size(); ++oi)
  {
    string object_class = objects.objects[oi].name;
    for (size_t pi = 0; pi < objects.objects[oi].parts.size(); ++pi)
    {
      string part_affordance = objects.objects[oi].parts[pi].affordance;

      stringstream marker_ns;
      marker_ns << "obj_" << oi << "/" << object_class << "/" << part_affordance;

      // collect object part material
      ROS_INFO("");
      ROS_INFO("%s has material (metal, ceramic, plastic, glass, wood, stone, paper): ", marker_ns.str().c_str());
      string material;
      std::cin >> material;
      ROS_INFO("This part has material %s!", material.c_str());

      objects.objects[oi].parts[pi].material = material;
    }
  }

  // Update object list and publish it
  objects.header.seq++;
  objects.header.stamp = ros::Time::now();
  objects.header.frame_id = geometric_segmentation_frame_;
  objects.cleared = false;
  object_list_ = objects;
  semantic_objects_pub_.publish(object_list_);

  return true;
}

bool ObjectSemanticSegmentation::segmentObjectsGeometric(rail_semantic_grasping::SemanticObjectList &objects)
{
  for (size_t i = 0; i < objects.objects.size(); ++i)
  {
    vector<rail_semantic_grasping::SemanticPart> new_parts;
    if (objects.objects[i].name == "cup")
    {
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::fromROSMsg(objects.objects[i].point_cloud, *object_pc);

      // segment opening from body
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr opening_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr body_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
      this->extractOpening(object_pc, body_pc, opening_pc);

      // segment body and handle
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr handle_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr new_body_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
      bool handle_segmented = this->extractHandle(body_pc, new_body_pc, handle_pc);
      if (handle_segmented)
      {
        body_pc = new_body_pc;
      }

      // assign newly segmented parts to the semantic object
      rail_semantic_grasping::SemanticPart body_part;
      pcl::toROSMsg(*body_pc, body_part.point_cloud);
      body_part.affordance = "body";
      new_parts.push_back(body_part);
      debug_pc_pub_.publish(body_pc);

      rail_semantic_grasping::SemanticPart opening_part;
      pcl::toROSMsg(*opening_pc, opening_part.point_cloud);
      opening_part.affordance = "opening";
      new_parts.push_back(opening_part);
      debug_pc_pub_2_.publish(opening_pc);

      if (handle_segmented)
      {
        rail_semantic_grasping::SemanticPart handle_part;
        pcl::toROSMsg(*handle_pc, handle_part.point_cloud);
        handle_part.affordance = "handle";
        new_parts.push_back(handle_part);
        debug_pc_pub_3_.publish(handle_pc);
      }
    }
    if (!new_parts.empty())
    {
      objects.objects[i].parts = new_parts;
    }
  }
  return true;
}

bool ObjectSemanticSegmentation::extractHandle(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &object,
                                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &body,
                                               pcl::PointCloud<pcl::PointXYZRGB>::Ptr &handle)
{
  // get dimension of the segmented object
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*object, min_pt, max_pt);
  double width = max_pt[0] - min_pt[0];

  // segment the cylinder and the remaining point cloud should have the handle
  pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
  pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> segmenter;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;

  pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
  pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
  pcl::PointIndices::Ptr cylinder_indices(new pcl::PointIndices);

  // Estimate point normals
  normal_estimator.setSearchMethod(tree);
  normal_estimator.setInputCloud(object);
  // estimate a point's normal using its K nearest neighbors
  normal_estimator.setKSearch(cylinder_segmentation_normal_k_);
  // another option: normal_estimator.setRadiusSearch(0.03); // 3cm
  normal_estimator.compute(*cloud_normals);

  // create the segmentation object for cylinder segmentation and set all the parameters
  segmenter.setOptimizeCoefficients(true);
  segmenter.setModelType(pcl::SACMODEL_CYLINDER);
  segmenter.setMethodType(pcl::SAC_RANSAC);
  segmenter.setNormalDistanceWeight(cylinder_segmentation_normal_distance_weight_);
  segmenter.setMaxIterations(cylinder_segmentation_max_iteration_);
  segmenter.setDistanceThreshold(width * cylinder_segmentation_distance_threshold_ratio_);
  segmenter.setRadiusLimits(0, width);
  segmenter.setInputCloud(object);
  segmenter.setInputNormals(cloud_normals);
  segmenter.segment(*cylinder_indices, *coefficients_cylinder);

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_pc(new pcl::PointCloud<pcl::PointXYZRGB> ());
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr handle_pc(new pcl::PointCloud<pcl::PointXYZRGB> ());

  // extract the cylinder point cloud and the remaining point cloud
  extract.setInputCloud(object);
  extract.setIndices(cylinder_indices);
  extract.setNegative(false);
  extract.filter(*cylinder_pc);
  extract.setNegative(true);
  extract.filter(*handle_pc);
  if (cylinder_pc->points.empty() or handle_pc->points.empty())
  {
    ROS_INFO("Cannot find the cylinder or the handle.");
    return false;
  }

  // further refine the point cloud of the handle. The largest cluster should correspond to the handle
  vector<pcl::PointIndices> clusters;
  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr cluster_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
  cluster_tree->setInputCloud(handle_pc);
  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> cluster_extractor;
  cluster_extractor.setClusterTolerance(cylinder_segmentation_cluster_tolerance_);
  cluster_extractor.setMinClusterSize(cylinder_segmentation_min_cluster_size_);
  cluster_extractor.setMaxClusterSize(cylinder_segmentation_max_cluster_size_);
  cluster_extractor.setSearchMethod(cluster_tree);
  cluster_extractor.setInputCloud(handle_pc);
  cluster_extractor.extract(clusters);

  size_t max_cluster_size=0;
  int max_cluster_index=0;
  if (!clusters.empty())
  {
    for (size_t ci = 0; ci < clusters.size(); ++ci)
    {
      if (clusters[ci].indices.size() > max_cluster_size)
      {
        max_cluster_size = clusters[ci].indices.size();
        max_cluster_index = ci;
      }
    }
  } else
  {
    ROS_INFO("Cannot cluster remaining point cloud that are not part of the cylinder to find the handle.");
    return false;
  }

  // extract the largest cluster that corresponds to the handle
  pcl::PointIndices::Ptr handle_indices(new pcl::PointIndices);
  *handle_indices = clusters[max_cluster_index];
  extract.setInputCloud(handle_pc);
  extract.setIndices(handle_indices);
  extract.setNegative(false);
  extract.filter(*handle_pc);

  // add point cloud that are not part of the handle back to the cylinder point cloud
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr remain_pc(new pcl::PointCloud<pcl::PointXYZRGB> ());
  extract.setNegative(true);
  extract.filter(*remain_pc);
  // add pc of this part to the combined object
  // the += operator should take care of width, height, is_dense, and header
  *cylinder_pc += *remain_pc;

  body = cylinder_pc;
  handle = handle_pc;
  return true;
}

bool ObjectSemanticSegmentation::extractOpening(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &object,
                                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr &body,
                                                pcl::PointCloud<pcl::PointXYZRGB>::Ptr &opening)
{
  // The following part is used to segment a pc into two parts based on height
  // calculate the axis-aligned bounding box
  Eigen::Vector4f min_pt, max_pt;
  pcl::getMinMax3D(*object, min_pt, max_pt);
  double max_z = max_pt[2];
  //      double width = max_pt[0] - min_pt[0];
  //      double depth = max_pt[1] - min_pt[1];
  //      double height = max_pt[2] - min_pt[2];
  //      geometry_msgs::PoseStamped center;
  //      center.header.frame_id = geometric_segmentation_frame_;
  //      center.header.stamp = ros::Time(0);
  //      center.pose.position.x = (max_pt[0] + min_pt[0]) / 2.0;
  //      center.pose.position.y = (max_pt[1] + min_pt[1]) / 2.0;
  //      center.pose.position.z = max_pt[2];
  //      debug_pose_pub_.publish(center);

  // filter out pc that are not less than max_z - 0.05
  double height_constraint = max_z - 0.02;
  pcl::ConditionAnd<pcl::PointXYZRGB>::Ptr bounds (new pcl::ConditionAnd<pcl::PointXYZRGB>());
  bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
      new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LE, height_constraint)));

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr remaining_pc (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ConditionalRemoval<pcl::PointXYZRGB> removal(true);
  removal.setCondition(bounds);
  removal.setInputCloud(object);
  removal.filter(*remaining_pc);
  const pcl::IndicesConstPtr &filter_indices = removal.getRemovedIndices();

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr filtered_pc (new pcl::PointCloud<pcl::PointXYZRGB>);
  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
  extract.setInputCloud(object);
  extract.setIndices(filter_indices);
  extract.filter(*filtered_pc);

  body = remaining_pc;
  opening = filtered_pc;
  return true;
}

//bool ObjectSemanticSegmentation::segmentObjects(rail_manipulation_msgs::SegmentedObjectList &objects)
//{
//  // check if we have a point cloud first
//  {
//    boost::mutex::scoped_lock lock(pc_mutex_);
//    if (!first_pc_in_)
//    {
//      ROS_WARN("No point cloud received yet. Ignoring segmentation request.");
//      return false;
//    }
//  }
//
//  // clear the objects first
//  std_srvs::Empty empty;
//  this->clearCallback(empty.request, empty.response);
//
//  // determine the correct segmentation zone
//  const SegmentationZone &zone = this->getCurrentZone();
//  ROS_INFO("Segmenting in zone '%s'.", zone.getName().c_str());
//
//  // transform the point cloud to the fixed frame
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
//  // lock on the point cloud
//  {
//    boost::mutex::scoped_lock lock(pc_mutex_);
//    // perform the copy/transform using TF
//    pcl_ros::transformPointCloud(zone.getBoundingFrameID(), ros::Time(0), *pc_, pc_->header.frame_id,
//                                 *transformed_pc, tf_);
//    transformed_pc->header.frame_id = zone.getBoundingFrameID();
//    transformed_pc->header.seq = pc_->header.seq;
//    transformed_pc->header.stamp = pc_->header.stamp;
//  }
//
//  // start with every index
//  pcl::IndicesPtr filter_indices(new vector<int>);
//  filter_indices->resize(transformed_pc->points.size());
//  for (size_t i = 0; i < transformed_pc->points.size(); i++)
//  {
//    filter_indices->at(i) = i;
//  }
//
//  // check if we need to remove a surface
//  double z_min = zone.getZMin();
//  if (!crop_first_)
//  {
//    if (zone.getRemoveSurface())
//    {
//      bool surface_found = this->findSurface(transformed_pc, filter_indices, zone, filter_indices, table_);
//      if (zone.getRequireSurface() && !surface_found)
//      {
//        objects.objects.clear();
//        ROS_INFO("Could not find a surface within the segmentation zone.  Exiting segmentation with no objects found.");
//        return true;
//      }
//      double z_surface = table_.centroid.z;
//      // check the new bound for Z
//      z_min = max(zone.getZMin(), z_surface + SURFACE_REMOVAL_PADDING);
//    }
//  }
//
//  // check bounding areas (bound the inverse of what we want since PCL will return the removed indicies)
//  pcl::ConditionOr<pcl::PointXYZRGB>::Ptr bounds(new pcl::ConditionOr<pcl::PointXYZRGB>);
//  if (z_min > -numeric_limits<double>::infinity())
//  {
//    bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//        new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LE, z_min))
//    );
//  }
//  if (zone.getZMax() < numeric_limits<double>::infinity())
//  {
//    bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//        new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GE, zone.getZMax()))
//    );
//  }
//  if (zone.getYMin() > -numeric_limits<double>::infinity())
//  {
//    bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//        new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::LE, zone.getYMin()))
//    );
//  }
//  if (zone.getYMax() < numeric_limits<double>::infinity())
//  {
//    bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//        new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::GE, zone.getYMax()))
//    );
//  }
//  if (zone.getXMin() > -numeric_limits<double>::infinity())
//  {
//    bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//        new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::LE, zone.getXMin()))
//    );
//  }
//  if (zone.getXMax() < numeric_limits<double>::infinity())
//  {
//    bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//        new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::GE, zone.getXMax()))
//    );
//  }
//
//  // remove past the given bounds
//  this->inverseBound(transformed_pc, filter_indices, bounds, filter_indices);
//
//  if (crop_first_)
//  {
//    if (zone.getRemoveSurface())
//    {
//      bool surface_found = this->findSurface(transformed_pc, filter_indices, zone, filter_indices, table_);
//      if (zone.getRequireSurface() && !surface_found)
//      {
//        objects.objects.clear();
//        ROS_INFO("Could not find a surface within the segmentation zone.  Exiting segmentation with no objects found.");
//        return true;
//      }
//      double z_surface = table_.centroid.z;
//      // check the new bound for Z
//      z_min = max(zone.getZMin(), z_surface + SURFACE_REMOVAL_PADDING);
//
//      pcl::ConditionOr<pcl::PointXYZRGB>::Ptr table_bounds(new pcl::ConditionOr<pcl::PointXYZRGB>);
//      table_bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//          new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::LE, z_min))
//      );
//
//      // plane segmentation does adds back in the filtered indices, so we need to re-add the old bounds (this should
//      // be faster than conditionally merging the two lists of indices, which would require a bunch of searches the
//      // length of the point cloud's number of points)
//      if (zone.getZMax() < numeric_limits<double>::infinity())
//      {
//        table_bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//            new pcl::FieldComparison<pcl::PointXYZRGB>("z", pcl::ComparisonOps::GE, zone.getZMax()))
//        );
//      }
//      if (zone.getYMin() > -numeric_limits<double>::infinity())
//      {
//        table_bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//            new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::LE, zone.getYMin()))
//        );
//      }
//      if (zone.getYMax() < numeric_limits<double>::infinity())
//      {
//        table_bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//            new pcl::FieldComparison<pcl::PointXYZRGB>("y", pcl::ComparisonOps::GE, zone.getYMax()))
//        );
//      }
//      if (zone.getXMin() > -numeric_limits<double>::infinity())
//      {
//        table_bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//            new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::LE, zone.getXMin()))
//        );
//      }
//      if (zone.getXMax() < numeric_limits<double>::infinity())
//      {
//        table_bounds->addComparison(pcl::FieldComparison<pcl::PointXYZRGB>::ConstPtr(
//            new pcl::FieldComparison<pcl::PointXYZRGB>("x", pcl::ComparisonOps::GE, zone.getXMax()))
//        );
//      }
//
//      // remove below the table bounds
//      this->inverseBound(transformed_pc, filter_indices, table_bounds, filter_indices);
//    }
//  }
//
//  // publish the filtered and bounded PC pre-segmentation
//  if (debug_)
//  {
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr debug_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
//    this->extract(transformed_pc, filter_indices, debug_pc);
//    debug_pc_pub_.publish(debug_pc);
//  }
//
//  // extract clusters
//  vector<pcl::PointIndices> clusters;
//  if (use_color_)
//    this->extractClustersRGB(transformed_pc, filter_indices, clusters);
//  else
//    this->extractClustersEuclidean(transformed_pc, filter_indices, clusters);
//
//  if (clusters.size() > 0)
//  {
//    // lock for the messages
//    boost::mutex::scoped_lock lock(msg_mutex_);
//    // check each cluster
//    for (size_t i = 0; i < clusters.size(); i++)
//    {
//      // grab the points we need
//      pcl::PointCloud<pcl::PointXYZRGB>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
//      for (size_t j = 0; j < clusters[i].indices.size(); j++)
//      {
//        cluster->points.push_back(transformed_pc->points[clusters[i].indices[j]]);
//      }
//      cluster->width = cluster->points.size();
//      cluster->height = 1;
//      cluster->is_dense = true;
//      cluster->header.frame_id = transformed_pc->header.frame_id;
//
//      // check if we need to transform to a different frame
//      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
//      pcl::PCLPointCloud2::Ptr converted(new pcl::PCLPointCloud2);
//      if (zone.getBoundingFrameID() != zone.getSegmentationFrameID())
//      {
//        // perform the copy/transform using TF
//        pcl_ros::transformPointCloud(zone.getSegmentationFrameID(), ros::Time(0), *cluster, cluster->header.frame_id,
//                                     *transformed_cluster, tf_);
//        transformed_cluster->header.frame_id = zone.getSegmentationFrameID();
//        transformed_cluster->header.seq = cluster->header.seq;
//        transformed_cluster->header.stamp = cluster->header.stamp;
//        pcl::toPCLPointCloud2(*transformed_cluster, *converted);
//      } else
//      {
//        pcl::toPCLPointCloud2(*cluster, *converted);
//      }
//
//      // convert to a SegmentedObject message
//      rail_manipulation_msgs::SegmentedObject segmented_object;
//      segmented_object.recognized = false;
//
//      // set the RGB image
//      segmented_object.image = this->createImage(transformed_pc, clusters[i]);
//
//      // check if we want to publish the image
//      if (debug_)
//      {
//        debug_img_pub_.publish(segmented_object.image);
//      }
//
//      // set the point cloud
//      pcl_conversions::fromPCL(*converted, segmented_object.point_cloud);
//      segmented_object.point_cloud.header.stamp = ros::Time::now();
//      // create a marker and set the extra fields
//      segmented_object.marker = this->createMarker(converted);
//      segmented_object.marker.id = i;
//
//      // calculate color features
//      Eigen::Vector3f rgb, lab;
//      rgb[0] = segmented_object.marker.color.r;
//      rgb[1] = segmented_object.marker.color.g;
//      rgb[2] = segmented_object.marker.color.b;
//      lab = RGB2Lab(rgb);
//      segmented_object.rgb.resize(3);
//      segmented_object.cielab.resize(3);
//      segmented_object.rgb[0] = rgb[0];
//      segmented_object.rgb[1] = rgb[1];
//      segmented_object.rgb[2] = rgb[2];
//      segmented_object.cielab[0] = lab[0];
//      segmented_object.cielab[1] = lab[1];
//      segmented_object.cielab[2] = lab[2];
//
//      // set the centroid
//      Eigen::Vector4f centroid;
//      if (zone.getBoundingFrameID() != zone.getSegmentationFrameID())
//      {
//        pcl::compute3DCentroid(*transformed_cluster, centroid);
//      } else
//      {
//        pcl::compute3DCentroid(*cluster, centroid);
//      }
//      segmented_object.centroid.x = centroid[0];
//      segmented_object.centroid.y = centroid[1];
//      segmented_object.centroid.z = centroid[2];
//
//      // calculate the minimum volume bounding box (assuming the object is resting on a flat surface)
//      segmented_object.bounding_volume = BoundingVolumeCalculator::computeBoundingVolume(segmented_object.point_cloud);
//
//      // calculate the axis-aligned bounding box
//      Eigen::Vector4f min_pt, max_pt;
//      pcl::getMinMax3D(*cluster, min_pt, max_pt);
//      segmented_object.width = max_pt[0] - min_pt[0];
//      segmented_object.depth = max_pt[1] - min_pt[1];
//      segmented_object.height = max_pt[2] - min_pt[2];
//
//      // calculate the center
//      segmented_object.center.x = (max_pt[0] + min_pt[0]) / 2.0;
//      segmented_object.center.y = (max_pt[1] + min_pt[1]) / 2.0;
//      segmented_object.center.z = (max_pt[2] + min_pt[2]) / 2.0;
//
//      // calculate the orientation
//      pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
//      // project point cloud onto the xy plane
//      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
//      coefficients->values.resize(4);
//      coefficients->values[0] = 0;
//      coefficients->values[1] = 0;
//      coefficients->values[2] = 1.0;
//      coefficients->values[3] = 0;
//      pcl::ProjectInliers<pcl::PointXYZRGB> proj;
//      proj.setModelType(pcl::SACMODEL_PLANE);
//      if (zone.getBoundingFrameID() != zone.getSegmentationFrameID())
//      {
//        proj.setInputCloud(transformed_cluster);
//      } else
//      {
//        proj.setInputCloud(cluster);
//      }
//      proj.setModelCoefficients(coefficients);
//      proj.filter(*projected_cluster);
//
//      //calculate the Eigen vectors of the projected point cloud's covariance matrix, used to determine orientation
//      Eigen::Vector4f projected_centroid;
//      Eigen::Matrix3f covariance_matrix;
//      pcl::compute3DCentroid(*projected_cluster, projected_centroid);
//      pcl::computeCovarianceMatrixNormalized(*projected_cluster, projected_centroid, covariance_matrix);
//      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
//      Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
//      eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));
//      //calculate rotation from eigenvectors
//      const Eigen::Quaternionf qfinal(eigen_vectors);
//
//      //convert orientation to a single angle on the 2D plane defined by the segmentation coordinate frame
//      tf::Quaternion tf_quat;
//      tf_quat.setValue(qfinal.x(), qfinal.y(), qfinal.z(), qfinal.w());
//      double r, p, y;
//      tf::Matrix3x3 m(tf_quat);
//      m.getRPY(r, p, y);
//      double angle = r + y;
//      while (angle < -M_PI)
//      {
//        angle += 2 * M_PI;
//      }
//      while (angle > M_PI)
//      {
//        angle -= 2 * M_PI;
//      }
//      segmented_object.orientation = tf::createQuaternionMsgFromYaw(angle);
//
//      // add to the final list
//      objects.objects.push_back(segmented_object);
//      // add to the markers
//      markers_.markers.push_back(segmented_object.marker);
//
//      if (label_markers_)
//      {
//        // create a text marker to label the current marker
//        visualization_msgs::Marker text_marker;
//        text_marker.header = segmented_object.marker.header;
//        text_marker.ns = "segmentation_labels";
//        text_marker.id = i;
//        text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
//        text_marker.action = visualization_msgs::Marker::ADD;
//
//        text_marker.pose.position.x = segmented_object.center.x;
//        text_marker.pose.position.y = segmented_object.center.y;
//        text_marker.pose.position.z = segmented_object.center.z + 0.05 + segmented_object.height/2.0;
//
//        text_marker.scale.x = .1;
//        text_marker.scale.y = .1;
//        text_marker.scale.z = .1;
//
//        text_marker.color.r = 1;
//        text_marker.color.g = 1;
//        text_marker.color.b = 1;
//        text_marker.color.a = 1;
//
//        stringstream marker_label;
//        marker_label << "i:" << i;
//        text_marker.text = marker_label.str();
//
//        text_markers_.markers.push_back(text_marker);
//      }
//    }
//
//    // create the new list
//    objects.header.seq++;
//    objects.header.stamp = ros::Time::now();
//    objects.header.frame_id = zone.getSegmentationFrameID();
//    objects.cleared = false;
//
//    // update the new list and publish it
//    object_list_ = objects;
//    segmented_objects_pub_.publish(object_list_);
//
//    // publish the new marker array
//    if (label_markers_)
//    {
//      visualization_msgs::MarkerArray marker_list;
//      marker_list.markers.reserve(markers_.markers.size() + text_markers_.markers.size());
//      marker_list.markers.insert(marker_list.markers.end(), markers_.markers.begin(), markers_.markers.end());
//      marker_list.markers.insert(marker_list.markers.end(), text_markers_.markers.begin(), text_markers_.markers.end());
//      markers_pub_.publish(marker_list);
//    } else
//    {
//      markers_pub_.publish(markers_);
//    }
//
//    // add to the markers
//    table_marker_ = table_.marker;
//
//    // publish the new list
//    table_pub_.publish(table_);
//
//    // publish the new marker array
//    table_marker_pub_.publish(table_marker_);
//
//  } else
//  {
//    ROS_WARN("No segmented objects found.");
//  }
//
//  return true;
//}
//
//bool ObjectSemanticSegmentation::calculateFeaturesCallback(rail_manipulation_msgs::ProcessSegmentedObjects::Request &req,
//    rail_manipulation_msgs::ProcessSegmentedObjects::Response &res)
//{
//  res.segmented_objects.header = req.segmented_objects.header;
//  res.segmented_objects.cleared = req.segmented_objects.cleared;
//  res.segmented_objects.objects.resize(req.segmented_objects.objects.size());
//
//  for (size_t i = 0; i < res.segmented_objects.objects.size(); i ++)
//  {
//    // convert to a SegmentedObject message
//    res.segmented_objects.objects[i].recognized = req.segmented_objects.objects[i].recognized;
//
//    // can't recalculate this after initial segmentation has already happened...
//    res.segmented_objects.objects[i].image = req.segmented_objects.objects[i].image;
//
//    // set the point cloud
//    res.segmented_objects.objects[i].point_cloud = req.segmented_objects.objects[i].point_cloud;
//
//    // get point cloud as pcl point cloud
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr pcl_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
//    pcl::PCLPointCloud2::Ptr temp_cloud(new pcl::PCLPointCloud2);
//    pcl_conversions::toPCL(res.segmented_objects.objects[i].point_cloud, *temp_cloud);
//    pcl::fromPCLPointCloud2(*temp_cloud, *pcl_cloud);
//
//    // create a marker and set the extra fields
//    res.segmented_objects.objects[i].marker = this->createMarker(temp_cloud);
//    res.segmented_objects.objects[i].marker.id = i;
//
//    // calculate color features
//    Eigen::Vector3f rgb, lab;
//    rgb[0] = res.segmented_objects.objects[i].marker.color.r;
//    rgb[1] = res.segmented_objects.objects[i].marker.color.g;
//    rgb[2] = res.segmented_objects.objects[i].marker.color.b;
//    lab = RGB2Lab(rgb);
//    res.segmented_objects.objects[i].rgb.resize(3);
//    res.segmented_objects.objects[i].cielab.resize(3);
//    res.segmented_objects.objects[i].rgb[0] = rgb[0];
//    res.segmented_objects.objects[i].rgb[1] = rgb[1];
//    res.segmented_objects.objects[i].rgb[2] = rgb[2];
//    res.segmented_objects.objects[i].cielab[0] = lab[0];
//    res.segmented_objects.objects[i].cielab[1] = lab[1];
//    res.segmented_objects.objects[i].cielab[2] = lab[2];
//
//    // set the centroid
//    Eigen::Vector4f centroid;
//    pcl::compute3DCentroid(*pcl_cloud, centroid);
//    res.segmented_objects.objects[i].centroid.x = centroid[0];
//    res.segmented_objects.objects[i].centroid.y = centroid[1];
//    res.segmented_objects.objects[i].centroid.z = centroid[2];
//
//    // calculate the minimum volume bounding box (assuming the object is resting on a flat surface)
//    res.segmented_objects.objects[i].bounding_volume =
//        BoundingVolumeCalculator::computeBoundingVolume(res.segmented_objects.objects[i].point_cloud);
//
//    // calculate the axis-aligned bounding box
//    Eigen::Vector4f min_pt, max_pt;
//    pcl::getMinMax3D(*pcl_cloud, min_pt, max_pt);
//    res.segmented_objects.objects[i].width = max_pt[0] - min_pt[0];
//    res.segmented_objects.objects[i].depth = max_pt[1] - min_pt[1];
//    res.segmented_objects.objects[i].height = max_pt[2] - min_pt[2];
//
//    // calculate the center
//    res.segmented_objects.objects[i].center.x = (max_pt[0] + min_pt[0]) / 2.0;
//    res.segmented_objects.objects[i].center.y = (max_pt[1] + min_pt[1]) / 2.0;
//    res.segmented_objects.objects[i].center.z = (max_pt[2] + min_pt[2]) / 2.0;
//
//    // calculate the orientation
//    pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
//    // project point cloud onto the xy plane
//    pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
//    coefficients->values.resize(4);
//    coefficients->values[0] = 0;
//    coefficients->values[1] = 0;
//    coefficients->values[2] = 1.0;
//    coefficients->values[3] = 0;
//    pcl::ProjectInliers<pcl::PointXYZRGB> proj;
//    proj.setModelType(pcl::SACMODEL_PLANE);
//    proj.setInputCloud(pcl_cloud);
//    proj.setModelCoefficients(coefficients);
//    proj.filter(*projected_cluster);
//
//    //calculate the Eigen vectors of the projected point cloud's covariance matrix, used to determine orientation
//    Eigen::Vector4f projected_centroid;
//    Eigen::Matrix3f covariance_matrix;
//    pcl::compute3DCentroid(*projected_cluster, projected_centroid);
//    pcl::computeCovarianceMatrixNormalized(*projected_cluster, projected_centroid, covariance_matrix);
//    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
//    Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
//    eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));
//    //calculate rotation from eigenvectors
//    const Eigen::Quaternionf qfinal(eigen_vectors);
//
//    //convert orientation to a single angle on the 2D plane defined by the segmentation coordinate frame
//    tf::Quaternion tf_quat;
//    tf_quat.setValue(qfinal.x(), qfinal.y(), qfinal.z(), qfinal.w());
//    double r, p, y;
//    tf::Matrix3x3 m(tf_quat);
//    m.getRPY(r, p, y);
//    double angle = r + y;
//    while (angle < -M_PI)
//    {
//      angle += 2 * M_PI;
//    }
//    while (angle > M_PI)
//    {
//      angle -= 2 * M_PI;
//    }
//    res.segmented_objects.objects[i].orientation = tf::createQuaternionMsgFromYaw(angle);
//  }
//
//  return true;
//}
//
//bool ObjectSemanticSegmentation::findSurface(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in,
//    const pcl::IndicesConstPtr &indices_in, const SegmentationZone &zone, const pcl::IndicesPtr &indices_out,
//    rail_manipulation_msgs::SegmentedObject &table_out) const
//{
//  // use a plane (SAC) segmenter
//  pcl::SACSegmentation<pcl::PointXYZRGB> plane_seg;
//  // set the segmenation parameters
//  plane_seg.setOptimizeCoefficients(true);
//  plane_seg.setModelType(pcl::SACMODEL_PERPENDICULAR_PLANE);
//  plane_seg.setAxis(Eigen::Vector3f(0, 0, 1));
//  plane_seg.setEpsAngle(SAC_EPS_ANGLE);
//  plane_seg.setMethodType(pcl::SAC_RANSAC);
//  plane_seg.setMaxIterations(SAC_MAX_ITERATIONS);
//  plane_seg.setDistanceThreshold(SAC_DISTANCE_THRESHOLD);
//
//  // create a copy to work with
//  pcl::PointCloud<pcl::PointXYZRGB>::Ptr pc_copy(new pcl::PointCloud<pcl::PointXYZRGB>(*in));
//  plane_seg.setInputCloud(pc_copy);
//  plane_seg.setIndices(indices_in);
//
//  // Check point height -- if the plane is too low or high, extract another
//  while (true)
//  {
//    // points included in the plane (surface)
//    pcl::PointIndices::Ptr inliers_ptr(new pcl::PointIndices);
//
//    // segment the the current cloud
//    pcl::ModelCoefficients coefficients;
//    plane_seg.segment(*inliers_ptr, coefficients);
//
//    // check if we found a surface
//    if (inliers_ptr->indices.size() == 0)
//    {
//      ROS_WARN("Could not find a surface above %fm and below %fm.", zone.getZMin(), zone.getZMax());
//      *indices_out = *indices_in;
//      table_out.centroid.z = -numeric_limits<double>::infinity();
//      return false;
//    }
//
//    // remove the plane
//    pcl::PointCloud<pcl::PointXYZRGB> plane;
//    pcl::ExtractIndices<pcl::PointXYZRGB> extract(true);
//    extract.setInputCloud(pc_copy);
//    extract.setIndices(inliers_ptr);
//    extract.setNegative(false);
//    extract.filter(plane);
//    extract.setKeepOrganized(true);
//    plane_seg.setIndices(extract.getRemovedIndices());
//
//    // check the heightdownsampled
//    double height = this->averageZ(plane.points);
//    if (height >= zone.getZMin() && height <= zone.getZMax())
//    {
//      ROS_INFO("Surface found at %fm.", height);
//      *indices_out = *plane_seg.getIndices();
//
//      // check if we need to transform to a different frame
//      pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
//      pcl::PCLPointCloud2::Ptr converted(new pcl::PCLPointCloud2);
//      if (zone.getBoundingFrameID() != zone.getSegmentationFrameID())
//      {
//        // perform the copy/transform using TF
//        pcl_ros::transformPointCloud(zone.getSegmentationFrameID(), ros::Time(0), plane, plane.header.frame_id,
//                                     *transformed_pc, tf_);
//        transformed_pc->header.frame_id = zone.getSegmentationFrameID();
//        transformed_pc->header.seq = plane.header.seq;
//        transformed_pc->header.stamp = plane.header.stamp;
//        pcl::toPCLPointCloud2(*transformed_pc, *converted);
//      } else
//      {
//        pcl::toPCLPointCloud2(plane, *converted);
//      }
//
//      // convert to a SegmentedObject message
//      table_out.recognized = false;
//
//      // set the RGB image
//      table_out.image = this->createImage(pc_copy, *inliers_ptr);
//
//      // check if we want to publish the image
//      if (debug_)
//      {
//        debug_img_pub_.publish(table_out.image);
//      }
//
//      // set the point cloud
//      pcl_conversions::fromPCL(*converted, table_out.point_cloud);
//      table_out.point_cloud.header.stamp = ros::Time::now();
//      // create a marker and set the extra fields
//      table_out.marker = this->createMarker(converted);
//      table_out.marker.id = 0;
//
//      // set the centroid
//      Eigen::Vector4f centroid;
//      if (zone.getBoundingFrameID() != zone.getSegmentationFrameID())
//      {
//        pcl::compute3DCentroid(*transformed_pc, centroid);
//      } else
//      {
//        pcl::compute3DCentroid(plane, centroid);
//      }
//      table_out.centroid.x = centroid[0];
//      table_out.centroid.y = centroid[1];
//      table_out.centroid.z = centroid[2];
//
//      // calculate the bounding box
//      Eigen::Vector4f min_pt, max_pt;
//      pcl::getMinMax3D(plane, min_pt, max_pt);
//      table_out.width = max_pt[0] - min_pt[0];
//      table_out.depth = max_pt[1] - min_pt[1];
//      table_out.height = max_pt[2] - min_pt[2];
//
//      // calculate the center
//      table_out.center.x = (max_pt[0] + min_pt[0]) / 2.0;
//      table_out.center.y = (max_pt[1] + min_pt[1]) / 2.0;
//      table_out.center.z = (max_pt[2] + min_pt[2]) / 2.0;
//
//
//      // calculate the orientation
//      pcl::PointCloud<pcl::PointXYZRGB>::Ptr projected_cluster(new pcl::PointCloud<pcl::PointXYZRGB>);
//      // project point cloud onto the xy plane
//      pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
//      coefficients->values.resize(4);
//      coefficients->values[0] = 0;
//      coefficients->values[1] = 0;
//      coefficients->values[2] = 1.0;
//      coefficients->values[3] = 0;
//      pcl::ProjectInliers<pcl::PointXYZRGB> proj;
//      proj.setModelType(pcl::SACMODEL_PLANE);
//      if (zone.getBoundingFrameID() != zone.getSegmentationFrameID())
//      {
//        proj.setInputCloud(transformed_pc);
//      } else
//      {
//        pcl::PointCloud<pcl::PointXYZRGB>::Ptr plane_ptr(new pcl::PointCloud<pcl::PointXYZRGB>(plane));
//        proj.setInputCloud(plane_ptr);
//      }
//      proj.setModelCoefficients(coefficients);
//      proj.filter(*projected_cluster);
//
//      //calculate the Eigen vectors of the projected point cloud's covariance matrix, used to determine orientation
//      Eigen::Vector4f projected_centroid;
//      Eigen::Matrix3f covariance_matrix;
//      pcl::compute3DCentroid(*projected_cluster, projected_centroid);
//      pcl::computeCovarianceMatrixNormalized(*projected_cluster, projected_centroid, covariance_matrix);
//      Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance_matrix, Eigen::ComputeEigenvectors);
//      Eigen::Matrix3f eigen_vectors = eigen_solver.eigenvectors();
//      eigen_vectors.col(2) = eigen_vectors.col(0).cross(eigen_vectors.col(1));
//      //calculate rotation from eigenvectors
//      const Eigen::Quaternionf qfinal(eigen_vectors);
//
//      //convert orientation to a single angle on the 2D plane defined by the segmentation coordinate frame
//      tf::Quaternion tf_quat;
//      tf_quat.setValue(qfinal.x(), qfinal.y(), qfinal.z(), qfinal.w());
//      double r, p, y;
//      tf::Matrix3x3 m(tf_quat);
//      m.getRPY(r, p, y);
//      double angle = r + y;
//      while (angle < -M_PI)
//      {
//        angle += 2 * M_PI;
//      }
//      while (angle > M_PI)
//      {
//        angle -= 2 * M_PI;
//      }
//      table_out.orientation = tf::createQuaternionMsgFromYaw(angle);
//
//      return true;
//    }
//  }
//}
//
//void ObjectSemanticSegmentation::extractClustersEuclidean(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in,
//    const pcl::IndicesConstPtr &indices_in, vector<pcl::PointIndices> &clusters) const
//{
//  // ignore NaN and infinite values
//  pcl::IndicesPtr valid(new vector<int>);
//  for (size_t i = 0; i < indices_in->size(); i++)
//  {
//    if (pcl_isfinite(in->points[indices_in->at(i)].x) & pcl_isfinite(in->points[indices_in->at(i)].y) &
//        pcl_isfinite(in->points[indices_in->at(i)].z))
//    {
//      valid->push_back(indices_in->at(i));
//    }
//  }
//
//  pcl::EuclideanClusterExtraction<pcl::PointXYZRGB> seg;
//  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
//  kd_tree->setInputCloud(in);
//  seg.setClusterTolerance(cluster_tolerance_);
//  seg.setMinClusterSize(min_cluster_size_);
//  seg.setMaxClusterSize(max_cluster_size_);
//  seg.setSearchMethod(kd_tree);
//  seg.setInputCloud(in);
//  seg.setIndices(valid);
//  seg.extract(clusters);
//}
//
//void ObjectSemanticSegmentation::extractClustersRGB(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in,
//                                   const pcl::IndicesConstPtr &indices_in, vector<pcl::PointIndices> &clusters) const
//{
//  // ignore NaN and infinite values
//  pcl::IndicesPtr valid(new vector<int>);
//  for (size_t i = 0; i < indices_in->size(); i++)
//  {
//    if (pcl_isfinite(in->points[indices_in->at(i)].x) & pcl_isfinite(in->points[indices_in->at(i)].y) &
//        pcl_isfinite(in->points[indices_in->at(i)].z))
//    {
//      valid->push_back(indices_in->at(i));
//    }
//  }
//  pcl::RegionGrowingRGB<pcl::PointXYZRGB> seg;
//  pcl::search::KdTree<pcl::PointXYZRGB>::Ptr kd_tree(new pcl::search::KdTree<pcl::PointXYZRGB>);
//  kd_tree->setInputCloud(in);
//  seg.setPointColorThreshold(POINT_COLOR_THRESHOLD);
//  seg.setRegionColorThreshold(REGION_COLOR_THRESHOLD);
//  seg.setDistanceThreshold(cluster_tolerance_);
//  seg.setMinClusterSize(min_cluster_size_);
//  seg.setMaxClusterSize(max_cluster_size_);
//  seg.setSearchMethod(kd_tree);
//  seg.setInputCloud(in);
//  seg.setIndices(valid);
//  seg.extract(clusters);
//}
//
//sensor_msgs::Image ObjectSemanticSegmentation::createImage(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in,
//    const pcl::PointIndices &cluster) const
//{
//  // determine the bounds of the cluster
//  int row_min = numeric_limits<int>::max();
//  int row_max = numeric_limits<int>::min();
//  int col_min = numeric_limits<int>::max();
//  int col_max = numeric_limits<int>::min();
//
//  for (size_t i = 0; i < cluster.indices.size(); i++)
//  {
//    // calculate row and column of this point
//    int row = cluster.indices[i] / in->width;
//    int col = cluster.indices[i] - (row * in->width);
//
//    if (row < row_min)
//    {
//      row_min = row;
//    } else if (row > row_max)
//    {
//      row_max = row;
//    }
//    if (col < col_min)
//    {
//      col_min = col;
//    } else if (col > col_max)
//    {
//      col_max = col;
//    }
//  }
//
//  // create a ROS image
//  sensor_msgs::Image msg;
//
//  // set basic information
//  msg.header.frame_id = in->header.frame_id;
//  msg.header.stamp = ros::Time::now();
//  msg.width = col_max - col_min;
//  msg.height = row_max - row_min;
//  // RGB data
//  msg.step = 3 * msg.width;
//  msg.data.resize(msg.step * msg.height);
//  msg.encoding = sensor_msgs::image_encodings::BGR8;
//
//  // extract the points
//  for (int h = 0; h < msg.height; h++)
//  {
//    for (int w = 0; w < msg.width; w++)
//    {
//      // extract RGB information
//      const pcl::PointXYZRGB &point = in->at(col_min + w, row_min + h);
//      // set RGB data
//      int index = (msg.step * h) + (w * 3);
//      msg.data[index] = point.b;
//      msg.data[index + 1] = point.g;
//      msg.data[index + 2] = point.r;
//    }
//  }
//
//  return msg;

//  // *******************************************************************************************
//  // This additional block of code is used to create object rgb image from the whole rgb image
//  // determine the bounds of the cluster
//  int row_min = numeric_limits<int>::max();
//  int row_max = numeric_limits<int>::min();
//  int col_min = numeric_limits<int>::max();
//  int col_max = numeric_limits<int>::min();
//
//  for (size_t ii = 0; ii < combined_object_image_indices.size(); ii++)
//  {
//  // calculate row and column of this point
//  int row = combined_object_image_indices[ii] / rgb_img->width;
//  int col = combined_object_image_indices[ii] - (row * rgb_img->width);
//
//  if (row < row_min)
//  {
//  row_min = row;
//  } else if (row > row_max)
//  {
//  row_max = row;
//  }
//  if (col < col_min)
//  {
//  col_min = col;
//  } else if (col > col_max)
//  {
//  col_max = col;
//  }
//  }
//
//  // create a ROS image
//  sensor_msgs::Image msg;
//  // set basic information
//  msg.header.frame_id = rgb_img->header.frame_id;
//  msg.header.stamp = ros::Time::now();
//  msg.width = col_max - col_min;
//  msg.height = row_max - row_min;
//  // RGB data
//  msg.step = 3 * msg.width;
//  msg.data.resize(msg.step * msg.height);
//  msg.encoding = sensor_msgs::image_encodings::BGR8;
//
//  ROS_INFO("new image width %d, height %d", col_max - col_min, row_max - row_min);
//  ROS_INFO("new image corner %d, %d", row_min, col_min);
//  ROS_INFO("old img width %d, height %d", rgb_img->width, rgb_img->height);
//  ROS_INFO("old img step %d", rgb_img->step);
//  ROS_INFO("old img encoding %s", rgb_img->encoding.c_str());
//
//  // extract the points
//  for (int h = 0; h < msg.height; h++)
//  {
//  for (int w = 0; w < msg.width; w++)
//  {
//  // set RGB data
//  int index = (msg.step * h) + (w * 3);
//  int old_index = (rgb_img->step * (h + row_min)) + ((w + col_min) * 3);
//  msg.data[index] = rgb_img->data[old_index + 2];
//  msg.data[index + 1] = rgb_img->data[old_index + 1];
//  msg.data[index + 2] = rgb_img->data[old_index];
//  }
//  }
//  debug_img_pub_.publish(msg);
//  // *******************************************************************************************
//}

visualization_msgs::Marker ObjectSemanticSegmentation::createMarker(const pcl::PCLPointCloud2::ConstPtr &pc,
    const std::string &marker_namespace) const
{
  visualization_msgs::Marker marker;
  // set header field
  marker.header.frame_id = pc->header.frame_id;

  // set marker namespace
  marker.ns = marker_namespace;

  // default position
  marker.pose.position.x = 0.0;
  marker.pose.position.y = 0.0;
  marker.pose.position.z = 0.0;
  marker.pose.orientation.x = 0.0;
  marker.pose.orientation.y = 0.0;
  marker.pose.orientation.z = 0.0;
  marker.pose.orientation.w = 1.0;

  // default scale
  marker.scale.x = MARKER_SCALE;
  marker.scale.y = MARKER_SCALE;
  marker.scale.z = MARKER_SCALE;

  // set the type of marker and our color of choice
  marker.type = visualization_msgs::Marker::CUBE_LIST;
  marker.color.a = 1.0;

  // downsample point cloud for visualization
  pcl::PCLPointCloud2 downsampled;
  pcl::VoxelGrid<pcl::PCLPointCloud2> voxel_grid;
  voxel_grid.setInputCloud(pc);
  voxel_grid.setLeafSize(DOWNSAMPLE_LEAF_SIZE, DOWNSAMPLE_LEAF_SIZE, DOWNSAMPLE_LEAF_SIZE);
  voxel_grid.filter(downsampled);

  // convert to an easy to use point cloud message
  sensor_msgs::PointCloud2 pc2_msg;
  pcl_conversions::fromPCL(downsampled, pc2_msg);
  sensor_msgs::PointCloud pc_msg;
  sensor_msgs::convertPointCloud2ToPointCloud(pc2_msg, pc_msg);

  // place in the marker message
  marker.points.resize(pc_msg.points.size());
  // int r = 0, g = 0, b = 0;
  for (size_t j = 0; j < pc_msg.points.size(); j++)
  {
    marker.points[j].x = pc_msg.points[j].x;
    marker.points[j].y = pc_msg.points[j].y;
    marker.points[j].z = pc_msg.points[j].z;

//    // use average RGB
//    uint32_t rgb = *reinterpret_cast<int *>(&pc_msg.channels[0].values[j]);
//    r += (int) ((rgb >> 16) & 0x0000ff);
//    g += (int) ((rgb >> 8) & 0x0000ff);
//    b += (int) ((rgb) & 0x0000ff);
  }

  // set average RGB
//  marker.color.r = ((float) r / (float) pc_msg.points.size()) / 255.0;
//  marker.color.g = ((float) g / (float) pc_msg.points.size()) / 255.0;
//  marker.color.b = ((float) b / (float) pc_msg.points.size()) / 255.0;
  marker.color.r = rand() / double(RAND_MAX);
  marker.color.g = rand() / double(RAND_MAX);
  marker.color.b = rand() / double(RAND_MAX);
  marker.color.a = 1.0;

  return marker;
}

visualization_msgs::Marker ObjectSemanticSegmentation::createTextMarker(const std::string &label, const std_msgs::Header &header,
    const geometry_msgs::Point &position, const std::string &marker_namespace) const
{
  // Create a text marker to label the current marker
  visualization_msgs::Marker text_marker;
  text_marker.header = header;
  text_marker.ns = marker_namespace;
  // part marker has id 0, label has id 1
  text_marker.id = 1;
  text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
  text_marker.action = visualization_msgs::Marker::ADD;

  text_marker.pose.position.x = position.x - 0.05;
  text_marker.pose.position.y = position.y;
  text_marker.pose.position.z = position.z + 0.02;

  text_marker.scale.x = .03;
  text_marker.scale.y = .03;
  text_marker.scale.z = .03;

  text_marker.color.r = 1;
  text_marker.color.g = 1;
  text_marker.color.b = 1;
  text_marker.color.a = 1;

  text_marker.text = label;

  return text_marker;
}

//
//void ObjectSemanticSegmentation::extract(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in, const pcl::IndicesConstPtr &indices_in,
//    const pcl::PointCloud<pcl::PointXYZRGB>::Ptr &out) const
//{
//  pcl::ExtractIndices<pcl::PointXYZRGB> extract;
//  extract.setInputCloud(in);
//  extract.setIndices(indices_in);
//  extract.filter(*out);
//}
//
//void ObjectSemanticSegmentation::inverseBound(const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr &in,
//    const pcl::IndicesConstPtr &indices_in,
//    const pcl::ConditionBase<pcl::PointXYZRGB>::Ptr &conditions,
//    const pcl::IndicesPtr &indices_out) const
//{
//  // use a temp point cloud to extract the indices
//  pcl::PointCloud<pcl::PointXYZRGB> tmp;
//  //pcl::ConditionalRemoval<pcl::PointXYZRGB> removal(conditions, true);
//  pcl::ConditionalRemoval<pcl::PointXYZRGB> removal(true);
//  removal.setCondition(conditions);
//  removal.setInputCloud(in);
//  removal.setIndices(indices_in);
//  removal.filter(tmp);
//  *indices_out = *removal.getRemovedIndices();
//}
//
//double ObjectSemanticSegmentation::averageZ(const vector<pcl::PointXYZRGB, Eigen::aligned_allocator<pcl::PointXYZRGB> > &v) const
//{
//  double avg = 0.0;
//  for (size_t i = 0; i < v.size(); i++)
//  {
//    avg += v[i].z;
//  }
//  return (avg / (double) v.size());
//}
//
////convert from RGB color space to CIELAB color space, adapted from pcl/registration/gicp6d
//Eigen::Vector3f RGB2Lab (const Eigen::Vector3f& colorRGB)
//{
//  // for sRGB   -> CIEXYZ see http://www.easyrgb.com/index.php?X=MATH&H=02#text2
//  // for CIEXYZ -> CIELAB see http://www.easyrgb.com/index.php?X=MATH&H=07#text7
//
//  double R, G, B, X, Y, Z;
//
//  R = colorRGB[0];
//  G = colorRGB[1];
//  B = colorRGB[2];
//
//  // linearize sRGB values
//  if (R > 0.04045)
//    R = pow ( (R + 0.055) / 1.055, 2.4);
//  else
//    R = R / 12.92;
//
//  if (G > 0.04045)
//    G = pow ( (G + 0.055) / 1.055, 2.4);
//  else
//    G = G / 12.92;
//
//  if (B > 0.04045)
//    B = pow ( (B + 0.055) / 1.055, 2.4);
//  else
//    B = B / 12.92;
//
//  // postponed:
//  //    R *= 100.0;
//  //    G *= 100.0;
//  //    B *= 100.0;
//
//  // linear sRGB -> CIEXYZ
//  X = R * 0.4124 + G * 0.3576 + B * 0.1805;
//  Y = R * 0.2126 + G * 0.7152 + B * 0.0722;
//  Z = R * 0.0193 + G * 0.1192 + B * 0.9505;
//
//  // *= 100.0 including:
//  X /= 0.95047;  //95.047;
//  //    Y /= 1;//100.000;
//  Z /= 1.08883;  //108.883;
//
//  // CIEXYZ -> CIELAB
//  if (X > 0.008856)
//    X = pow (X, 1.0 / 3.0);
//  else
//    X = 7.787 * X + 16.0 / 116.0;
//
//  if (Y > 0.008856)
//    Y = pow (Y, 1.0 / 3.0);
//  else
//    Y = 7.787 * Y + 16.0 / 116.0;
//
//  if (Z > 0.008856)
//    Z = pow (Z, 1.0 / 3.0);
//  else
//    Z = 7.787 * Z + 16.0 / 116.0;
//
//  Eigen::Vector3f colorLab;
//  colorLab[0] = static_cast<float> (116.0 * Y - 16.0);
//  colorLab[1] = static_cast<float> (500.0 * (X - Y));
//  colorLab[2] = static_cast<float> (200.0 * (Y - Z));
//
//  return colorLab;
//}
