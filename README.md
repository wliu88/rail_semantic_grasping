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
2. run `rosrun rail_semantic_grasping base_features_computation_node` to start the node for computing object base features
3. run `rosrun rail_semantic_grasping base_features_collection_node.py` to save base features to pickle files

## ToDo:
### data collection
- [ ] Think about whether the mapping from 2d mask to 3d is correct
- [ ] Think about whether the part affordances of cup should match other objects for transfer
- [ ] Implement other features in base feature extraction. For example, the size of parts, the orientation of parts, ...
- [ ] Cup 25 red metal with handle needs to be recollected. Spatula wood circular 3 is classified as hammer need to 
be recollected
- [ ] If we are going to recollect all data. Then we should change the specification of msg types. For example, change
semantic grasp to include object state.
- [ ] Change has_stuff to filled
### model design
- [x] Assign weights to different classes since the data is inbalanced. This can be achieved in NLLLoss

## Progress

- 8.22: 
    - define all experiments
    - run all experiments on baselines
- 8.26:
    - implement algorithm
    
## Data Stats

* 19 ceramic bowl white display contain w_grasp
* 43 small ceramic bowl
* 0: glass bowl with dots contain w_grasp
* 20: glass bowl transparent contain w_grasp
* 17: metal bowl square contain w-grasp
* 30: metal bowl red contain
* 21: paper bowl display contain
* 3: white paper bowl
* 15: plastic bowl purple contain w_grasp
* 16: plastic bowl china-like contain
* 31: wood bowl large contain
* 40: shallow wood bowl 
---
* 10: glass bottle ikea grasp
* 18: glass bottle pasta sause contain grasp
* 8: metal bottle water bottle contain grasp
* 9: metal bottle sid's contain grasp
* 6: plastic bottle shaker contain grasp
* 7: plastic bottle shampoo contain grasp
---
* 1: metal spatula ||| display support grasp
* 2: metal spatula turner support grasp
* 32: plastic spatula scoop support grasp
* 33: plastic spatula turner ||| support grasp
* 34: plastic spatula narrow spoon grasp pound
* 35: plastic spatula black support grasp
* 5: wood spatula turner support grasp
* 4: wood spatula turner ||| support grasp
---
* 38: ceramic pan yellow contain grasp
* 36: ceramic pan small black contain w_grasp grasp
* 11: stone pan black contain grasp
* 41: small stone pan
* 12: metal pan silver contain w_grasp grasp
* 37: metal pan large contain grasp
---
* 22: ceramic cup large white opening handle body
* 23: ceramic cup black opening body handle
* 13: glass cup with handle opening handle body
* 29: glass cup transparent beer opening body
* 25: metal cup red opening body
* 27: metal cup green openign body handle
* 28: paper cup with coffee opening body
* 39: white paper cup 
* 14: plastic cup transparent opening body
* 26: plastic cup small blue opening body
* 24: wood cup with handle opening handle body
* 42: wood cup without handle

