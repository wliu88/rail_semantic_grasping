# rail_semantic_grasping_offline

This package is the offline version of rail_semantic_grasping that does not require using the robot. The main 
functionality of this package is to collect semantic grasp labels from users and run different models on the labeled
data. 

## Note
This package is originally created for ROS melodic. However, it can still be compiled and used in ROS kinetic by adding
the line `add_compile_options(-std=c++11)` to the top-level `CMakeLists.txt` of the workspace.

## Collect Grasp Labels
1. run `roscore`
2. run `rosrun rail_semantic_grasping grasp_collection_node.py`
3. run `rviz`
4. follow the prompt and visualization in rviz

## Compute Object and Grasp Base Features
1. run `roscore`
2. run `rosrun rail_semantic_graspingase_features_computation_node` to start the node for computing object base features
3. run `rosrun rail_semantic_grasping base_features_collection_node.py` to save base features to pickle files

