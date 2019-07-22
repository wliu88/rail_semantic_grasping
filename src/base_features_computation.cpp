#include "rail_semantic_grasping/base_features_computation.h"

using namespace std;
using namespace rail::semantic_grasping;

BaseFeaturesComputation::BaseFeaturesComputation() : private_node_("~"), tf2_(tf_buffer_), debug(true)
{
    private_node_.param("shape_segmentation_max_iteration", shape_segmentation_max_iteration_, 10000);
    private_node_.param("cylinder_segmentation_normal_k", cylinder_segmentation_normal_k_, 200);
    private_node_.param("cylinder_segmentation_normal_distance_weight", cylinder_segmentation_normal_distance_weight_, 0.1);
    private_node_.param("cylinder_segmentation_distance_threshold_ratio", cylinder_segmentation_distance_threshold_ratio_, 0.3);
    private_node_.param("sphere_segmentation_distance_threshold", sphere_segmentation_distance_threshold_, 0.01);
    private_node_.param("sphere_segmentation_probability", sphere_segmentation_probability_, 0.95);

    debug_pc_pub_ = private_node_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("debug_pc", 1, true);
    debug_pc_pub_2_ = private_node_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("debug_pc_2", 1, true);
    debug_pc_pub_3_ = private_node_.advertise<pcl::PointCloud<pcl::PointXYZRGB> >("debug_pc_3", 1, true);
    debug_pose_pub_ = private_node_.advertise<geometry_msgs::PoseStamped>("debug_pose", 1, true);

    compute_base_features_srv_ = private_node_.advertiseService("compute_base_features",
            &BaseFeaturesComputation::computeBaseFeaturesCallback, this);
}

bool BaseFeaturesComputation::computeBaseFeaturesCallback(rail_semantic_grasping::ComputeBaseFeaturesRequest &req,
                                                          rail_semantic_grasping::ComputeBaseFeaturesResponse &res)
{
    rail_semantic_grasping::SemanticObject semantic_object = req.semantic_objects.objects[0];
    ROS_INFO("The object has %zu parts and %zu grasps", semantic_object.parts.size(),
            semantic_object.labeled_grasps.size());
    ROS_INFO("Object point cloud is in %s frame", semantic_object.point_cloud.header.frame_id.c_str());

    // 1. Compute object-level features
    // transform sensor_msgs pc to pcl pc
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(semantic_object.point_cloud, *object_pc);

    // 1.1 Elongatedness: ratio of minor axes to the major axis
    // width: x axis; depth: y axis; height: z axis
    ROS_INFO("Object width, height, and depth: %f, %f, %f", semantic_object.width, semantic_object.height,
            semantic_object.depth);

    vector<pair<double, int>> axes;
    axes.push_back(make_pair(semantic_object.width, 0));
    axes.push_back(make_pair(semantic_object.depth, 1));
    axes.push_back(make_pair(semantic_object.height, 2));
    sort(axes.begin(), axes.end());
    ROS_INFO("The axis order from shortest to longest is %d, %d, %d", axes[0].second, axes[1].second, axes[2].second);

    // computed feature
    vector<double> elongatedness = {axes[1].first / axes[2].first, axes[0].first / axes[2].first};

    // 1.2 Shape primitives: cylinder, sphere
    // get dimension of the segmented object
    Eigen::Vector4f min_pt, max_pt;
    pcl::getMinMax3D(*object_pc, min_pt, max_pt);
    double width = max_pt[0] - min_pt[0];

    // segment cylinder, sphere
    pcl::NormalEstimation<pcl::PointXYZRGB, pcl::Normal> normal_estimator;
    pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> segmenter;
    pcl::search::KdTree<pcl::PointXYZRGB>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZRGB> ());
    pcl::ExtractIndices<pcl::PointXYZRGB> extract;

    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_cylinder(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr cylinder_indices(new pcl::PointIndices);

    // Estimate point normals
    normal_estimator.setSearchMethod(tree);
    normal_estimator.setInputCloud(object_pc);
    // estimate a point's normal using its K nearest neighbors
    normal_estimator.setKSearch(cylinder_segmentation_normal_k_);
    // another option: normal_estimator.setRadiusSearch(0.03); // 3cm
    normal_estimator.compute(*cloud_normals);

    // create the segmentation object for cylinder segmentation and set all the parameters
    segmenter.setOptimizeCoefficients(true);
    segmenter.setModelType(pcl::SACMODEL_CYLINDER);
    segmenter.setMethodType(pcl::SAC_RANSAC);
    segmenter.setNormalDistanceWeight(cylinder_segmentation_normal_distance_weight_);
    segmenter.setMaxIterations(shape_segmentation_max_iteration_);
    segmenter.setDistanceThreshold(width * cylinder_segmentation_distance_threshold_ratio_);
    segmenter.setRadiusLimits(0, width);
    segmenter.setInputCloud(object_pc);
    segmenter.setInputNormals(cloud_normals);
    segmenter.segment(*cylinder_indices, *coefficients_cylinder);
    ROS_INFO("Fitting to cylinder, inliers %zu, total %zu", cylinder_indices->indices.size(), object_pc->size());

    if (debug)
    {
        // extract cylinder pc
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cylinder_pc(new pcl::PointCloud<pcl::PointXYZRGB> ());
        // extract the cylinder point cloud and the remaining point cloud
        extract.setInputCloud(object_pc);
        extract.setIndices(cylinder_indices);
        extract.setNegative(false);
        extract.filter(*cylinder_pc);
        debug_pc_pub_.publish(cylinder_pc);
    }

    // create the segmentation object for sphere segmentation and set all the parameters
    //pcl::SACSegmentationFromNormals<pcl::PointXYZRGB, pcl::Normal> sphere_segmenter;
    pcl::SACSegmentation<pcl::PointXYZRGB> sphere_segmenter;
    pcl::ModelCoefficients::Ptr coefficients_sphere(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr sphere_indices(new pcl::PointIndices);

    sphere_segmenter.setOptimizeCoefficients(true);
    sphere_segmenter.setModelType(pcl::SACMODEL_SPHERE);
    sphere_segmenter.setMethodType(pcl::SAC_RANSAC);
    sphere_segmenter.setMaxIterations(shape_segmentation_max_iteration_);
    sphere_segmenter.setDistanceThreshold(sphere_segmentation_distance_threshold_);
    sphere_segmenter.setProbability(sphere_segmentation_probability_);
    sphere_segmenter.setRadiusLimits(0, width);
    sphere_segmenter.setInputCloud(object_pc);
    sphere_segmenter.segment(*sphere_indices, *coefficients_sphere);
    ROS_INFO("Fitting to sphere, inliers %zu, total %zu", sphere_indices->indices.size(), object_pc->size());

    if (debug)
    {
        // extract sphere pc
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr sphere_pc(new pcl::PointCloud<pcl::PointXYZRGB> ());
        // extract the cylinder point cloud and the remaining point cloud
        extract.setInputCloud(object_pc);
        extract.setIndices(sphere_indices);
        extract.setNegative(false);
        extract.filter(*sphere_pc);
        debug_pc_pub_2_.publish(sphere_pc);
    }

    // computed feature
    double spherical_resemblance = sphere_indices->indices.size() * 1.0 / object_pc->size();
    double cylindrical_resemblance = cylinder_indices->indices.size() * 1.0 / object_pc->size();

    // 1.3 Volume
    ROS_INFO("Bounding volume frame %s", semantic_object.bounding_volume.pose.header.frame_id.c_str());
    debug_pose_pub_.publish(semantic_object.bounding_volume.pose);
    double volume = semantic_object.bounding_volume.dimensions.x * semantic_object.bounding_volume.dimensions.y *
            semantic_object.bounding_volume.dimensions.z * 100 * 100 * 100;    // unit: meter^2

    // 1.4 Global shape descriptor
    pcl::ESFEstimation<pcl::PointXYZRGB, pcl::ESFSignature640> esf_extractor;
    esf_extractor.setInputCloud(object_pc);
    pcl::PointCloud<pcl::ESFSignature640>::Ptr esf_signature(new pcl::PointCloud<pcl::ESFSignature640>);
    esf_extractor.compute(*esf_signature);

    // computed feature
    vector<double> esf_descriptor;
    for (size_t ei = 0; ei < 640; ++ei)
    {
        esf_descriptor.push_back(esf_signature->points[0].histogram[ei]);
    }

    // 1.5 Opening
    double opening = 0.0;
    if (semantic_object.name == "cup" or semantic_object.name == "bowl" or semantic_object.name == "bottle")
    {
        opening = 1.0;
    }

    // 2. Compute features for each grasp
    for (size_t gi = 0; gi < semantic_object.labeled_grasps.size(); ++gi)
    {
        rail_semantic_grasping::SemanticGrasp grasp = semantic_object.labeled_grasps[gi];

        // 2.1 Relative grasping position: (relative position in axis / axis) for each axis
        // The order is from the longest axis to the shortest axis
        // For each value, should be between [-0.5, 0.5] unless the grasp is outside of the object
        ROS_INFO("object center: %f, %f, %f", semantic_object.center.x, semantic_object.center.y, semantic_object.center.z);
        ROS_INFO("grasp: %f, %f, %f", grasp.grasp_pose.position.x, grasp.grasp_pose.position.y, grasp.grasp_pose.position.z);
        vector<double> center_xyz = {semantic_object.center.x, semantic_object.center.y, semantic_object.center.z};
        vector<double> grasp_position_difference = {grasp.grasp_pose.position.x - semantic_object.center.x,
                                                    grasp.grasp_pose.position.y - semantic_object.center.y,
                                                    grasp.grasp_pose.position.z - semantic_object.center.z};

        // computed features
        vector<double> relative_grasping_position = {grasp_position_difference[axes[2].second] / axes[2].first,
                                                     grasp_position_difference[axes[1].second] / axes[1].first,
                                                     grasp_position_difference[axes[0].second] / axes[0].first};

        ROS_INFO("Relative grasp position is %f, %f, %f", relative_grasping_position[0], relative_grasping_position[1],
                 relative_grasping_position[2]);

        // 2.2 Opening: angle to the opening plane and distance to the opening plane if an opening exists
        tf::Quaternion q(grasp.grasp_pose.orientation.x,
                         grasp.grasp_pose.orientation.y,
                         grasp.grasp_pose.orientation.z,
                         grasp.grasp_pose.orientation.w);
        tf::Matrix3x3 m(q);
        double roll, pitch, yaw;
        m.getRPY(roll, pitch, yaw);

    }


    return true;
}

