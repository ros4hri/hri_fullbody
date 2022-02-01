# hri_fullbody: ROS4HRI holistic model estimation

`hri_fullbody` introduces a ROS node performing the estimation of a single
person holistic model over your camera video stream, via *Google Mediapipe*
holistic model estimation API. Since this project is still under development,
some of the concepts we are going to highlight in this document are already
thought for the future implementation of multi-people holistic model estimation.   

## Input Topics

- `RGB_IMAGE_TOPIC` => Topic for the incoming RGB video stream. The node performs
  fullbody model estimation on this stream. The code is thought to use
  compressed image (using the `CompressedImage` message from `sensor_msgs`).  

- `DEPTH_IMAGE_TOPIC` => Topic for the incoming depth video stream. This images
  are involved in the human model positioning in RViz. 

- `RGB_CAMERA_INFO_TOPIC` => Topic for the RGB camera information messages. It is
  used to access the RGB camera parameters for human model positioning in RViz. 

- `DEPTH_CAMERA_INFO_TOPIC` => Topic for the depth camera information messages.
  It is used to access the RGB camera parameters for human model positioning in
  RViz.

All of these topics must be defined before running the nodes, as environment
variables (e.g. `$ export RGB_IMAGE_TOPIC=/camera/color/image_raw/compressed`). 

## Output Topics

- `/humans/faces/*/roi` => The topic where the ROI of the face with id `*` gets
  published. The message type is `sensor_msgs/RegionOfInterest`.

- `/humans/bodies/*/roi` => The topic where the ROI of the body with id `*` gets
  published. The message type is `sensor_msgs/RegionOfInterest`.

- `/fullbody/visual` => Here gets published the video stream after some
  post-holistic-estimation image processing for debugging. It is possible to
  check which faces and which bodies the system detects. 

- `/fullbody/skeleton` => Here gets published the 2D skeleton message. The
  message type is Skeleton2D from ROS4HRI ROS messages package (`hri_msgs`). 

- `/humans/faces/tracked` => The topic where the list of the detected faces gets
  published. The message type is IdsList from ROS4HRI ROS messages package
  (`hri_msgs`).

- `/humans/bodies/tracked` => The topic where the list of the detected bodies
  gets published. The message type is IdsList from ROS4HRI ROS messages package
  (`hri_msgs`).

- `/test_joint_states` => The topic where the node publishes the human model
  jointstate for RViz visualization.

## Launch File

There is one simple `.launch` file that needs to be run to get everything from
this package working: `hri_fullbody.launch`. Once all the input topics have been
specified as environment variables, launching the file will also start RViz with
configured for human model visualization. 

## Video Example

Here an example of the expected result for the human model visualization on
RViz: ![Human Model Visualization](doc/human_model_visualization.webm)


