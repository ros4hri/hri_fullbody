cmake_minimum_required(VERSION 3.0.2)
project(hri_fullbody)

find_package(catkin REQUIRED COMPONENTS
  std_msgs
  rospy
  hri_msgs
  human_description
  std_msgs
  geometry_msgs
  image_geometry
)

catkin_python_setup()

catkin_package(
    CATKIN_DEPENDS hri_msgs)

catkin_install_python(PROGRAMS nodes/detect
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})

install(DIRECTORY
  launch
  rviz
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

