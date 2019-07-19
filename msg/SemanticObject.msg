rail_semantic_grasping/SemanticPart[] parts             # The list of parts with different affordances
sensor_msgs/PointCloud2 point_cloud                     # Segmented point cloud
geometry_msgs/Point centroid                            # Centroid of the point cloud
geometry_msgs/Point center                              # Center of the point cloud
rail_manipulation_msgs/BoundingVolume bounding_volume   # minimum bounding rectangular prism
float64 width                                           # The width of the object in meters (x in point cloud frame)
float64 depth                                           # The depth of the object in meters (y in point cloud frame)
float64 height                                          # The height of the object in meters (z in point cloud frame)
float32[] rgb                                           # Average color in RGB color space
float32[] cielab                                        # Average color in CIELAB color space
geometry_msgs/Quaternion orientation                    # Orientation of the object (typically from object rec)
bool recognized                                         # True if the object is recognized
string name                                             # Object name (if recognized)
float64 confidence                                      # Recognition confidence value
visualization_msgs/Marker marker                        # The downsampled visualization of the object
rail_semantic_grasping/SemanticGrasp[] grasps           # List of grasps
rail_semantic_grasping/SemanticGrasp[] labeled_grasps   # List of grasps labeled with tasks and scores
int32[] image_indices                                   # Indices of the segmented points in the 2D image coordinate
sensor_msgs/Image color_image                           # RAW RGB image of the whole scene
sensor_msgs/Image depth_image                           # RAW RGB image of the whole scene