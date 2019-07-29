# FEATURES
# Shape Primitives
float64 object_spherical_resemblance                    # How well the object resembles a sphere
float64 object_cylindrical_resemblance                  # How well the object resembles a cylinder
# Elongatedness
float64[] object_elongatedness                          # The ratio of the minor axes to the major axis
# Relative Free Volume
float64 object_volume                                   # Volume of the object
# Bag of Words Histogram
float64[] object_esf_descriptor                         # 3D Descriptor for object shape
# Relative Grasping Position
float64[] grasp_relative_position                       # The ratio of grasp position divided byb axis
# Opening
int32 object_opening                                    # Binary value indicating the existence of opening
float64 grasp_opening_angle
float64 grasp_opening_distance
# Intensity and Gradient Filters
float64[] grasp_intensity_histogram
float64[] grasp_first_gradient_histogram
float64[] grasp_second_gradient_histogram
# Color Quantization
float64[] grasp_color_histogram
# Color Uniformity
float64 grasp_color_mean
float64 grasp_color_variance
float64 grasp_color_entropy

# LABEL
int32 label
string task

# Size:
# object_spherical_resemblance: 1
# object_cylindrical_resemblance: 1
# object_elongatedness: 2
# object_volume: 1
# object_esf_descriptor: 640
# grasp_relative_position: 3
# object_opening: 1
# grasp_opening_angle: 1
# grasp_opening_distance: 1
# grasp_intensity_histogram: 25
# grasp_first_gradient_histogram: 25
# grasp_second_gradient_histogram: 25
# grasp_color_histogram: 25
# grasp_color_mean: 1
# grasp_color_variance: 1
# grasp_color_entropy: 1