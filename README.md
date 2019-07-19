# rail_semantic_grasping_offline

This package is the offline version of rail_semantic_grasping that does not require using the robot. The main 
functionality of this package is to collect semantic grasp labels from users and run different models on the labeled
data. 

## Note
This package is originally created for ROS melodic. However, it can still be compiled and used in ROS kinetic by adding
the line `add_compile_options(-std=c++11)` to the top-level `CMakeLists.txt` of the workspace.