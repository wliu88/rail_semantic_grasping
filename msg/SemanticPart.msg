sensor_msgs/PointCloud2 point_cloud                     # Segmented point cloud
string affordance                                       # Affordance of the part
geometry_msgs/Point centroid                            # Centroid of the point cloud
visualization_msgs/Marker marker                        # The downsampled visualization of the part
visualization_msgs/Marker text_marker                   # Text marker showing the affordance of the part
string material                                         # Material of the part

#int32[] image_indices                                   # Indices of the segmented points in the 2D image coordinate
#sensor_msgs/Image image                                 # Segmented RGB image
#geometry_msgs/Point centroid                            # Centroid of the point cloud
#geometry_msgs/Point center                              # Center of the point cloud
#rail_manipulation_msgs/BoundingVolume bounding_volume   # minimum bounding rectangular prism
#float64 width                                           # The width of the object in meters (x in point cloud frame)
#float64 depth                                           # The depth of the object in meters (y in point cloud frame)
#float64 height                                          # The height of the object in meters (z in point cloud frame)
#float32[] rgb                                           # Average color in RGB color space
#float32[] cielab                                        # Average color in CIELAB color space
#geometry_msgs/Quaternion orientation                    # Orientation of the object (typically from object rec)

