FROM osrf/ros:kinetic-desktop-full

# Set the root directory as HOME
ENV HOME=/root

# Install the apt dependencies
RUN DEBIAN_FRONTEND=noninteractive apt-get update && \
    apt-get install -y -q \
        git \
        curl \
        nano && \
    curl https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python /tmp/get-pip.py && \
    pip install -U \
        numpy && \
    mkdir -p $HOME/catkin_ws/src

# Clone the repo, initialize & build the workspaces, and then update the
# entrypoint into the container to source the right workspace
WORKDIR $HOME/catkin_ws
# COPY ./scripts/rosinstall/active.rosinstall $HOME/ros/active/src/.rosinstall
# COPY ./scripts/rosinstall/stable.rosinstall $HOME/ros/stable/src/.rosinstall

RUN . /opt/ros/kinetic/setup.sh && \
    cd $HOME/catkin_ws/src && \
	cd $HOME/catkin_ws/ && \
        catkin_make && \
    cd $HOME/catkin_ws/src && \
    git clone https://github.com/wliu88/rail_semantic_grasping.git && \
    git clone https://github.com/GT-RAIL/rail_manipulation_msgs.git && \
    git clone https://github.com/wliu88/rail_part_affordance_detection.git && \
    sed -i "3i add_compile_options(-std=c++11)" $HOME/catkin_ws/src/CMakeLists.txt && \
    cd $HOME/catkin_ws && \
    catkin_make && \
    sed -i 's/\/opt\/ros\/\$ROS_DISTRO/\$HOME\/catkin_ws\/devel/' /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]
