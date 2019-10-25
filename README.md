# rail_semantic_grasping_offline

This package contains the the semantic grasping dataset **_SG14000_**, and the code for the **_Semantic Grasp Network_** 
in the paper **_CAGE: Context-Aware Grasping Engine_**, currently available [here](https://arxiv.org/abs/1909.11142). 

The code can be used to collect semantic grasp labels from users and test our proposed **_Semantic Grasp Network_** 
and other baselines on the labeled data. 

This package is a stand-alone package, so it is separated from the code that extracts semantic information from grasps 
and contexts and executes semantic grasps on the Fetch robot. A separate package will be released for these
functions.

# Data
All data is stored in `/data`. The structure of the folder is:
```text
-data
 -Objects_Description.md
 -labeled
  -YEAR_MONTH_DAY_HOUR_MINUTE
 -unlabeled
  -YEAR_MONTH_DAY_HOUR_MINUTE
 -base_features
  -YEAR_MONTH_DAY_HOUR_MINUTE
```
Folders with the same data and time as name in `labeled`, `unlabeled`, and `base_features` contain data for the same 
experiment. 

## SG14000 Dataset
The **_SG14000_** dataset is in the `/data/labeled/2019_08_19_11_08` folder. The folder has 44 pickle files for the 44 objects in
the dataset. Each file is a pickled ROS message described in `/msg/SemanticObjectList.msg`.
Each `SemanticObjectList` message only contains one semantic object defined in `/msg/SemanticObject.msg`. Furthermore, each 
`SemanticObject` message contains a list of grasps in the field named `labeled_grasps`. These grasps are the labeled 
semantic grasps of the SG14000 dataset. As defined in `SemanticGrasp.msg`, each semantic grasp has a pose, a grasp 
affordance and a grasp material (semantic representation of grasps introduced in the paper), a task (which we use to 
describe both the manipulation task and the object state), and a label (which indicates whether this grasp is suitable 
for the context). 

Detailed description for all objects can be found in `/data/Object_Description.md`.

## Unlabeled Data
To label semantic grasps yourself, you can use the the data in `data/unlabled/2019_08_19_11_08`. The folder also 
contains 44 pickle files for the 44 objects. The only difference from the data described above is each 
`SemanticObject` message now contains a list of unlabled grasps in the field `grasps`. To collect grasp labels with
provided code, please see the instructions in the Code section. Remember to rename the folder name to avoid rewriting
the SG14000 dataset.

## Base Features
One of the baselines of the paper uses base features such as image gradients and RGBD descriptors to ground semantic 
grasps. For each object in `/data/labeled/2019_08_19_11_08`, there is corresponding pickle file in 
`/data/base_features/2019_08_19_11_08`. Each pickle file contains a list of base features, defined in 
`/msg/BaseFeatures.msg`, for each grasp. 

# Code

## ROS Package
This package was originally developed with ROS melodic. However, it can be compiled and used in ROS kinetic 
by adding the line `add_compile_options(-std=c++11)` to the top-level `CMakeLists.txt` of the workspace.

## Collect Labels
The steps to collect your own labels for semantic grasps:
1. run `roscore`
2. run `rosrun rail_semantic_grasping grasp_collection_node.py`
3. run `rviz`
4. follow the prompt and visualization in rviz

## Compute Base Features
The steps to compute base features:
1. run `roscore`
2. run `rosrun rail_semantic_grasping base_features_computation_node` to start the node for computing object base features
3. run `rosrun rail_semantic_grasping base_features_collection_node.py` to save base features to pickle files

## Test CAGE and other baselines
Code for both our models and baselines is in `/main`. Scripts in this folder can all be run directly in python shell. 
No `rosrun` is needed. However, make sure you source your catkin space because the code still depends on message types
define in ros. 

To run various models on the experiments described in the paper, check out `/main/run_experiments.py`. 
 
The code mainly has three components:
`/main/DataReader.py` reads the grasps in pickle files and prepares data splits for different experiments; Code in 
`/main/algorithms` has implementations for different algorithms; `Metrics.py` scores algorithms based on their predictions.  