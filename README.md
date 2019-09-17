# rail_semantic_grasping_offline

This package contains the the semantic grasping dataset, **_SG14000_** and the code for the **_Semantic Grasp Network_** 
in the paper **_CAGE: Context-Aware Grasping Engine_**. 

The code can be used to collect semantic grasp labels from users, and test our proposed **_Semantic Grasp Network_** 
and other baselines on the labeled data. 

This package is a stand-alone package, so it is separated from the code that extracts semantic information from grasps 
and contexts, and executes semantic grasps on the Fetch robot. A separate package will be released that covers these
functions.

# DATA

In the `/data/labeled` folder contains the **_SG14000_** dataset. The folder has 44 pickle files for the 44 objects in
the dataset. Each file is a pickled ROS message of the `SemanticObjectList.msg` type (see `/msg` for more details). Each 
`SemanticObjectList` message only contains one semantic object defined in `SemanticObject.msg`. Furthermore, each 
`SemanticObject` message contains a list of grasps in the field `labeled_grasps`. These grasps are the labeled semantic
grasps of the SG14000 dataset. As defined in `SemanticGrasp.msg`, each semantic grasp has a pose, a grasp affordance and
a grasp material (semantic representation of grasps introduced in the paper), a task (which we use to describe both the 
manipulation task and the object state), and a label (which indicates whether this grasp is suitable for the context). 

Detailed description for all objects can be found in `/data/Object_Description.md`.

## Collect your own labels
To define your own semantic grasps, you can use the the data in `data/unlabled`. The folder also contains 44 pickle 
files for the 44 objects. The only difference from the data described above is each 
`SemanticObject` message now contains a list of unlabled grasps in the field `grasps`. To collect grasp labels with
provided code, please see the instructions in the CODE section. 

# CODE

## ROS Package
This package is a ROS package originally created for ROS melodic. However, it can be compiled and used in ROS kinetic 
by adding the line `add_compile_options(-std=c++11)` to the top-level `CMakeLists.txt` of the workspace.

## Collect Grasp Labels
1. run `roscore`
2. run `rosrun rail_semantic_grasping grasp_collection_node.py`
3. run `rviz`
4. follow the prompt and visualization in rviz

## Compute Object and Grasp Base Features
1. run `roscore`
2. run `rosrun rail_semantic_grasping base_features_computation_node` to start the node for computing object base features
3. run `rosrun rail_semantic_grasping base_features_collection_node.py` to save base features to pickle files

