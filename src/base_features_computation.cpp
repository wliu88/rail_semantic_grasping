#include "rail_semantic_grasping/base_features_computation.h"

using namespace std;
using namespace rail::semantic_grasping;

BaseFeaturesComputation::BaseFeaturesComputation() : private_node_("~"), tf2_(tf_buffer_), debug_(false)
{
    // check opencv version. Since indigo, the default is 3.
    ROS_INFO("Computation of image features is based on OpenCV %d", CV_MAJOR_VERSION);

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
    debug_pose_pub_2_ = private_node_.advertise<geometry_msgs::PoseStamped>("debug_pose_2", 1, true);
    debug_img_pub_ = private_node_.advertise<sensor_msgs::Image>("debug_img", 1, true);

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

    //##################################################################################################################
    // 1. Compute object-level features
    // transform sensor_msgs pc to pcl pc
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr object_pc(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::fromROSMsg(semantic_object.point_cloud, *object_pc);

    // --------------------------------------------------------
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
    vector<double> object_elongatedness = {axes[1].first / axes[2].first, axes[0].first / axes[2].first};

    // --------------------------------------------------------
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

    if (debug_)
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

    if (debug_)
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
    double object_spherical_resemblance = sphere_indices->indices.size() * 1.0 / object_pc->size();
    double object_cylindrical_resemblance = cylinder_indices->indices.size() * 1.0 / object_pc->size();

    // --------------------------------------------------------
    // 1.3 Volume
    ROS_INFO("Bounding volume frame %s", semantic_object.bounding_volume.pose.header.frame_id.c_str());
    //debug_pose_pub_.publish(semantic_object.bounding_volume.pose);
    double object_volume = semantic_object.bounding_volume.dimensions.x * semantic_object.bounding_volume.dimensions.y *
            semantic_object.bounding_volume.dimensions.z * 100 * 100 * 100;    // unit: meter^2

    // --------------------------------------------------------
    // 1.4 Global shape descriptor
    pcl::ESFEstimation<pcl::PointXYZRGB, pcl::ESFSignature640> esf_extractor;
    esf_extractor.setInputCloud(object_pc);
    pcl::PointCloud<pcl::ESFSignature640>::Ptr esf_signature(new pcl::PointCloud<pcl::ESFSignature640>);
    esf_extractor.compute(*esf_signature);

    // computed feature
    vector<double> object_esf_descriptor;
    for (size_t ei = 0; ei < 640; ++ei)
    {
        object_esf_descriptor.push_back(esf_signature->points[0].histogram[ei]);
    }

    // --------------------------------------------------------
    // 1.5 Opening
    int object_opening = 0;
    if (semantic_object.name == "cup" or semantic_object.name == "bowl"
        or semantic_object.name == "bottle" or semantic_object.name == "pan")
    {
        object_opening = 1;
    }

    //##################################################################################################################
    // 2. Compute features for each grasp
    for (size_t gi = 0; gi < semantic_object.labeled_grasps.size(); ++gi)
    {
        rail_semantic_grasping::SemanticGrasp grasp = semantic_object.labeled_grasps[gi];

        // --------------------------------------------------------
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
        vector<double> grasp_relative_position = {grasp_position_difference[axes[2].second] / axes[2].first,
                                                     grasp_position_difference[axes[1].second] / axes[1].first,
                                                     grasp_position_difference[axes[0].second] / axes[0].first};

        ROS_INFO("Relative grasp position is %f, %f, %f", grasp_relative_position[0], grasp_relative_position[1],
                 grasp_relative_position[2]);

        // --------------------------------------------------------
        // 2.2 Opening
        double grasp_opening_angle = 0;
        double grasp_opening_distance = 0;
        if (object_opening == 1.0)
        {
            // 2.2.1 Angle (i.e., cos(theta) in [-1, 1]) to the opening plane
            // calculate the cos(theta), where theta is the shortest angle between the grasp approach vector and the
            // vector pointing in the negative z-axis of base_link
            geometry_msgs::PoseStamped grasp_pose;
            grasp_pose.pose = grasp.grasp_pose;
            grasp_pose.header.frame_id = semantic_object.point_cloud.header.frame_id;

            if (debug_)
            {
                debug_pose_pub_2_.publish(grasp_pose);
            }

            tf::Quaternion grasp_q(grasp.grasp_pose.orientation.x,
                                             grasp.grasp_pose.orientation.y,
                                             grasp.grasp_pose.orientation.z,
                                             grasp.grasp_pose.orientation.w);
            tf::Matrix3x3 grasp_rot_m(grasp_q);
            tf::Vector3 grasp_vector = grasp_rot_m.getColumn(0);
            tf::Vector3 opening_normal_vector(0, 0, -1);
            double cos_angle = grasp_vector.normalized().dot(opening_normal_vector.normalized());

            // computed feature
            grasp_opening_angle = cos_angle;
            ROS_INFO("Cos(angle) from grasp to opening is %f", grasp_opening_angle);

            // 2.2.2 Distance to the opening plane if an opening exists
            // the shortest distance from the grasp to the opening plane is in the z direction

            // computed feature
            grasp_opening_distance = grasp_pose.pose.position.z - semantic_object.center.z;
            ROS_INFO("Distance from grasp to opening is %f", grasp_opening_distance);
        }

        // --------------------------------------------------------
        // 2.3 locate image region that the grasp is near to
        // find distance between the position of the grasp and the point cloud of the object

        int grasp_image_window_width = 20;

        pcl::PointXYZRGB grasp_position;
        grasp_position.x = grasp.grasp_pose.position.x;
        grasp_position.y = grasp.grasp_pose.position.y;
        grasp_position.z = grasp.grasp_pose.position.z;

        pcl::KdTreeFLANN<pcl::PointXYZRGB> kdtree;
        kdtree.setInputCloud(object_pc);
        vector<int> indices;
        vector<float> sqr_dsts;
        kdtree.nearestKSearch(grasp_position, 1, indices, sqr_dsts);

        // create grasp rgb image from the whole rgb image
        sensor_msgs::ImagePtr rbg_img (new sensor_msgs::Image);
        *rbg_img = semantic_object.color_image;
        int grasp_full_image_index = semantic_object.image_indices[indices[0]];
        // calculate row and column of this point
        int row = grasp_full_image_index / semantic_object.color_image.width;
        int col = grasp_full_image_index - (row * semantic_object.color_image.width);
        int row_min = row - grasp_image_window_width / 2;
        int row_max = row + grasp_image_window_width / 2;
        int col_min = col - grasp_image_window_width / 2;
        int col_max = col + grasp_image_window_width / 2;
        // create a ROS image
        sensor_msgs::Image msg;
        // set basic information
        msg.header.frame_id = semantic_object.color_image.header.frame_id;
        msg.header.stamp = ros::Time::now();
        msg.width = col_max - col_min;
        msg.height = row_max - row_min;
        // RGB data
        msg.step = 3 * msg.width;
        msg.data.resize(msg.step * msg.height);
        msg.encoding = sensor_msgs::image_encodings::BGR8;
        // create the new img
        for (int h = 0; h < msg.height; h++)
        {
            for (int w = 0; w < msg.width; w++)
            {
                int index = (msg.step * h) + (w * 3);
                // set RGB data
                int old_index = (semantic_object.color_image.step * (h + row_min)) + ((w + col_min) * 3);
                msg.data[index] = semantic_object.color_image.data[old_index + 2];
                msg.data[index + 1] = semantic_object.color_image.data[old_index + 1];
                msg.data[index + 2] = semantic_object.color_image.data[old_index];
            }
        }
        if (debug_)
        {
            debug_img_pub_.publish(msg);
        }


// The following block of code create an image of the whole object and highlight the position of the grasp on the img.
//        // the image indices of the closest point
//        int grasp_image_index = semantic_object.image_indices[indices[0]];
//        // ROS_INFO("closest point pc index %d / %zu", indices[0], object_pc->points.size());
//        // ROS_INFO("closest point image index %d", grasp_image_index);
//        // ROS_INFO("Sqr Distance: %f", sqr_dsts[0]);
//        // create object rgb image from the whole rgb image
//        sensor_msgs::ImagePtr rbg_img (new sensor_msgs::Image);
//        *rbg_img = semantic_object.color_image;
//        // determine the bounds of the cluster
//        int row_min = numeric_limits<int>::max();
//        int row_max = numeric_limits<int>::min();
//        int col_min = numeric_limits<int>::max();
//        int col_max = numeric_limits<int>::min();
//        for (size_t ii = 0; ii < semantic_object.image_indices.size(); ii++)
//        {
//            // calculate row and column of this point
//            int row = semantic_object.image_indices[ii] / semantic_object.color_image.width;
//            int col = semantic_object.image_indices[ii] - (row * semantic_object.color_image.width);
//
//            if (row < row_min)
//            {
//                row_min = row;
//            } else if (row > row_max)
//            {
//                row_max = row;
//            }
//            if (col < col_min)
//            {
//                col_min = col;
//            } else if (col > col_max)
//            {
//                col_max = col;
//            }
//        }
//
//        // create a ROS image
//        sensor_msgs::Image msg;
//        // set basic information
//        msg.header.frame_id = semantic_object.color_image.header.frame_id;
//        msg.header.stamp = ros::Time::now();
//        msg.width = col_max - col_min;
//        msg.height = row_max - row_min;
//        // RGB data
//        msg.step = 3 * msg.width;
//        msg.data.resize(msg.step * msg.height);
//        msg.encoding = sensor_msgs::image_encodings::BGR8;
//        // ROS_INFO("new image width %d, height %d", col_max - col_min, row_max - row_min);
//        // ROS_INFO("new image corner %d, %d", row_min, col_min);
//        // ROS_INFO("old img width %d, height %d", semantic_object.color_image.width, semantic_object.color_image.height);
//        // ROS_INFO("old img step %d", semantic_object.color_image.step);
//        // ROS_INFO("old img encoding %s", semantic_object.color_image.encoding.c_str());
//
//        // compute coordinate of the grasp center in the new img
//        int grasp_image_row = grasp_image_index / semantic_object.color_image.width;
//        int grasp_image_col = grasp_image_index - (grasp_image_row * semantic_object.color_image.width);
//        grasp_image_col = grasp_image_col - col_min;
//        grasp_image_row = grasp_image_row - row_min;
//        // ROS_INFO("closest point img row %d, col %d", grasp_image_row, grasp_image_col);
//
//        // create the new img
//        for (int h = 0; h < msg.height; h++)
//        {
//            for (int w = 0; w < msg.width; w++)
//            {
//                int index = (msg.step * h) + (w * 3);
//                if (abs(h - grasp_image_row) < 5 and abs(w - grasp_image_col) < 5)
//                {
//                    msg.data[index] = 0;
//                    msg.data[index + 1] = 255;
//                    msg.data[index + 2] = 255;
//                } else
//                {
//                    // set RGB data
//                    int old_index = (semantic_object.color_image.step * (h + row_min)) + ((w + col_min) * 3);
//                    msg.data[index] = semantic_object.color_image.data[old_index + 2];
//                    msg.data[index + 1] = semantic_object.color_image.data[old_index + 1];
//                    msg.data[index + 2] = semantic_object.color_image.data[old_index];
//                }
//            }
//        }
//        debug_img_pub_.publish(msg);


        // use OpenCV to compute image features
        // convert to opencv img
        cv_bridge::CvImagePtr cv_ptr;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);


// The following block of code detect and compute descriptors for keypoints using SIFT
//        // use sift detector
//        int minHessian = 100;
//        cv::Ptr<cv::Feature2D> sift = cv::xfeatures2d::SIFT::create(minHessian);
//        std::vector<cv::KeyPoint> keypoints;
//        sift->detect(cv_ptr->image, keypoints);
//        cv::Mat img_keypoints;
//        drawKeypoints(cv_ptr->image, keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//        string OPENCV_WINDOW = "test";
//        cv::namedWindow(OPENCV_WINDOW);
//        cv::imshow(OPENCV_WINDOW, img_keypoints);
//        cv::waitKey(3);
//        // find key points that are close to the grasp point
//        int nearest_sift_k = 4;
//        std::vector<cv::KeyPoint> k_nearest_keypoints;
//        for (int si = 0; si < keypoints.size(); ++si)
//        {
//            if (sqrt(pow(keypoints[si].pt.x - grasp_image_col, 2) + pow(keypoints[si].pt.y - grasp_image_row, 2)) < 10)
//            {
//                k_nearest_keypoints.push_back(keypoints[si]);
//            }
//        }
//        // visualize keypoints
//        cv::Mat img_keypoints;
//        drawKeypoints(cv_ptr->image, k_nearest_keypoints, img_keypoints, cv::Scalar::all(-1), cv::DrawMatchesFlags::DEFAULT);
//        cv::imshow(OPENCV_WINDOW, img_keypoints);
//        cv::waitKey(3);

        // 2.3.1 Intensity, gradient, second-order gradient histogram
        // compute first order gradient of the image
        cv::Mat src_blur, src_gray, grad;
        int kernel_size = 3;
        int scale = 1;
        int delta = 0;
        int ddepth = CV_16S;
        // blur the image
        cv::GaussianBlur(cv_ptr->image, src_blur, cv::Size(3, 3), 0 ,0 , cv::BORDER_DEFAULT);
        cv::cvtColor(src_blur, src_gray, CV_BGR2GRAY);
        // compute gradient in x, y direction
        cv::Mat grad_x, grad_y, abs_grad_x, abs_grad_y;
        cv::Scharr(src_gray, grad_x, ddepth, 1, 0, scale, delta, cv::BORDER_DEFAULT);
        // Sobel is another option, just not as accurate
        // Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
        cv::Scharr(src_gray, grad_y, ddepth, 0, 1, scale, delta, cv::BORDER_DEFAULT);
        // Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
        // scale and normalize
        convertScaleAbs(grad_x, abs_grad_x);
        convertScaleAbs(grad_y, abs_grad_y);
        // total gradient
        addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
        // visualize gradient
        if (debug_)
        {
            string first_order_vis = "first_order_grad";
            cv::namedWindow(first_order_vis);
            cv::imshow(first_order_vis, grad);
            cv::waitKey(3);
        }

        // compute second order gradient (apply Laplace function)
        cv::Mat lap, abs_lap;
        cv::Laplacian(src_gray, lap, ddepth, kernel_size, scale, delta, cv::BORDER_DEFAULT);
        convertScaleAbs(lap, abs_lap);
        // visualize second gradient
        if (debug_)
        {
            string second_order_vis = "second_order_grad";
            cv::namedWindow(second_order_vis);
            cv::imshow(second_order_vis, abs_lap);
            cv::waitKey(3);
        }

        // convert image to grayscale
        cv::Mat gray;
        cv::cvtColor(cv_ptr->image, gray, CV_BGR2GRAY);


// handy function to determine the data type of Mat
//        string r;
//        uchar depth = abs_lap.type() & CV_MAT_DEPTH_MASK;
//        uchar chans = 1 + (abs_lap.type() >> CV_CN_SHIFT);
//        switch ( depth ) {
//            case CV_8U:  r = "8U"; break;
//            case CV_8S:  r = "8S"; break;
//            case CV_16U: r = "16U"; break;
//            case CV_16S: r = "16S"; break;
//            case CV_32S: r = "32S"; break;
//            case CV_32F: r = "32F"; break;
//            case CV_64F: r = "64F"; break;
//            default:     r = "User"; break;
//        }
//        r += "C";
//        r += (chans+'0');
//        ROS_INFO("Matrix: %s", r.c_str());


        // create histogram
        // number of bins
        int bins = 25;
        int histSize[] = {bins};
        float hranges[] = {0, 255}; // for CV_8U
        const float* ranges[] = {hranges};
        cv::Mat hist1, hist2, hist0;
        // calculate histogram for first order gradient
        calcHist(&grad, 1, 0, cv::Mat(), // do not use mask
                 hist1, 1, histSize, ranges,
                 true, // the histogram is uniform
                 false // do not accumulate
                 );
        // calculate histogram for second order gradient
        calcHist(&abs_lap, 1, 0, cv::Mat(), // do not use mask
                 hist2, 1, histSize, ranges,
                 true, // the histogram is uniform
                 false // do not accumulate
        );
        //
        // calculate histogram for gray scale
        calcHist(&gray, 1, 0, cv::Mat(), // do not use mask
                 hist0, 1, histSize, ranges,
                 true, // the histogram is uniform
                 false // do not accumulate
        );

        // visualize histogram
        if (debug_)
        {
            int hist_w = 512, hist_h = 400;
            int bin_w = cvRound((double) hist_w / histSize[0]);
            cv::Mat histImage(hist_h, hist_w, CV_8UC3, cv::Scalar(0,0,0));

            normalize(hist1, hist1, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
            normalize(hist2, hist2, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
            normalize(hist0, hist0, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());

            for( int i = 1; i < histSize[0]; i++ )
            {
                line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist1.at<float>(i-1)) ),
                      cv::Point( bin_w*(i), hist_h - cvRound(hist1.at<float>(i)) ),
                      cv::Scalar( 255, 0, 0), 2, 8, 0 );
                line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist2.at<float>(i-1)) ),
                      cv::Point( bin_w*(i), hist_h - cvRound(hist2.at<float>(i)) ),
                      cv::Scalar( 0, 255, 0), 2, 8, 0 );
                line( histImage, cv::Point( bin_w*(i-1), hist_h - cvRound(hist0.at<float>(i-1)) ),
                      cv::Point( bin_w*(i), hist_h - cvRound(hist0.at<float>(i)) ),
                      cv::Scalar( 0, 0, 255), 2, 8, 0 );
            }
            string hist_vis = "image gradient histogram";
            cv::namedWindow(hist_vis);
            cv::imshow(hist_vis, histImage);
            cv::waitKey(3);
        }

        // computed features
        vector<double> grasp_intensity_histogram, grasp_first_gradient_histogram, grasp_second_gradient_histogram;
        for (int mi =0; mi < hist0.rows; ++mi)
        {
            grasp_intensity_histogram.insert(grasp_intensity_histogram.end(), hist0.ptr<float>(mi), hist0.ptr<float>(mi)+hist0.cols);
            grasp_first_gradient_histogram.insert(grasp_first_gradient_histogram.end(), hist1.ptr<float>(mi), hist1.ptr<float>(mi)+hist1.cols);
            grasp_second_gradient_histogram.insert(grasp_second_gradient_histogram.end(), hist2.ptr<float>(mi), hist2.ptr<float>(mi)+hist2.cols);
        }

        // normalize
        double intensity_sum = std::accumulate(grasp_intensity_histogram.begin(), grasp_intensity_histogram.end(), 0);
        double first_gradient_sum = std::accumulate(grasp_first_gradient_histogram.begin(), grasp_first_gradient_histogram.end(), 0);
        double second_gradient_sum = std::accumulate(grasp_second_gradient_histogram.begin(), grasp_second_gradient_histogram.end(), 0);
        for (int hi = 0; hi < grasp_intensity_histogram.size(); ++hi)
        {
            grasp_intensity_histogram[hi] = grasp_intensity_histogram[hi] / intensity_sum;
            grasp_first_gradient_histogram[hi] = grasp_first_gradient_histogram[hi] / first_gradient_sum;
            grasp_second_gradient_histogram[hi] = grasp_second_gradient_histogram[hi] / second_gradient_sum;
        }

        // 2.3.2 Color histogram in CIELab space
        cv::Mat img_lab;
        cvtColor(cv_ptr->image, img_lab, CV_BGR2Lab);

        // split lab image to separate channels
        vector<cv::Mat> lab_planes;
        split(img_lab, lab_planes);
        // bin size
        int lab_histSize[] = {15};
        cv::Mat l_hist, a_hist, b_hist;
        calcHist(&lab_planes[0], 1, 0, cv::Mat(), l_hist, 1, lab_histSize, ranges, true, false);
        calcHist(&lab_planes[1], 1, 0, cv::Mat(), a_hist, 1, lab_histSize, ranges, true, false);
        calcHist(&lab_planes[2], 1, 0, cv::Mat(), b_hist, 1, lab_histSize, ranges, true, false);

        // computed features
        vector<double> grasp_color_histogram;
        // option 1: concatenation
//        for (int mi =0; mi < l_hist.rows; ++mi)
//        {
//            grasp_color_histogram.insert(grasp_color_histogram.end(), l_hist.ptr<float>(mi), l_hist.ptr<float>(mi)+l_hist.cols);
//            grasp_color_histogram.insert(grasp_color_histogram.end(), a_hist.ptr<float>(mi), a_hist.ptr<float>(mi)+a_hist.cols);
//            grasp_color_histogram.insert(grasp_color_histogram.end(), b_hist.ptr<float>(mi), b_hist.ptr<float>(mi)+b_hist.cols);
//        }
        // option 2: Euclidean distance
        for (int mi = 0; mi < l_hist.rows; ++mi)
        {
            float* l_ptr = l_hist.ptr<float>(mi);
            float* a_ptr = a_hist.ptr<float>(mi);
            float* b_ptr = b_hist.ptr<float>(mi);
            for (int mj = 0; mj < l_hist.cols; ++mj)
            {
                float value = sqrt(pow(l_ptr[mj],2) + pow(a_ptr[mj],2) + pow(b_ptr[mj],2));
                grasp_color_histogram.push_back(value);
            }
        }

        // normalize
        double lab_sum = std::accumulate(grasp_color_histogram.begin(), grasp_color_histogram.end(), 0);
        for (int hi = 0; hi < grasp_color_histogram.size(); ++hi)
        {
            grasp_color_histogram[hi] = grasp_color_histogram[hi] / lab_sum;
        }

        // 2.3.2 Color uniformity in CIELab space
        // computed features
        // mean
        double grasp_color_mean = 0;
        for (int ci = 0; ci < grasp_color_histogram.size(); ++ci)
        {
            grasp_color_mean = grasp_color_mean + grasp_color_histogram[ci] * (ci + 1);
        }
        // sample variance
        double grasp_color_variance = 0;
        int total_number = 0;
        for (int ci = 0; ci < grasp_color_histogram.size(); ++ci)
        {
            grasp_color_variance = grasp_color_variance + grasp_color_histogram[ci] * pow(ci + 1 - grasp_color_mean, 2);
        }
        // entropy
        double grasp_color_entropy = 0;
        for (int ci = 0; ci < grasp_color_histogram.size(); ++ci)
        {
            if (grasp_color_histogram[ci] == 0) continue;
            grasp_color_entropy = grasp_color_entropy - grasp_color_histogram[ci] * log(grasp_color_histogram[ci]);
        }
        ROS_INFO("Color mean: %f", grasp_color_mean);
        ROS_INFO("Color variance: %f", grasp_color_variance);
        ROS_INFO("Color entropy: %f", grasp_color_entropy);

        // Store all features in msg
        rail_semantic_grasping::BaseFeatures base_features;
        base_features.object_spherical_resemblance = object_spherical_resemblance;
        base_features.object_cylindrical_resemblance = object_cylindrical_resemblance;
        base_features.object_elongatedness = object_elongatedness;
        base_features.object_volume = object_volume;
        base_features.object_esf_descriptor = object_esf_descriptor;
        base_features.grasp_relative_position = grasp_relative_position;
        base_features.object_opening = object_opening;
        base_features.grasp_opening_angle = grasp_opening_angle;
        base_features.grasp_opening_distance = grasp_opening_distance;
        base_features.grasp_intensity_histogram = grasp_intensity_histogram;
        base_features.grasp_first_gradient_histogram = grasp_first_gradient_histogram;
        base_features.grasp_second_gradient_histogram = grasp_second_gradient_histogram;
        base_features.grasp_color_histogram = grasp_color_histogram;
        base_features.grasp_color_mean = grasp_color_mean;
        base_features.grasp_color_variance = grasp_color_variance;
        base_features.grasp_color_entropy = grasp_color_entropy;
        base_features.label = grasp.score;
        base_features.task = grasp.task;

        // add to service response
        res.base_features_list.push_back(base_features);
    }

    return true;
}

