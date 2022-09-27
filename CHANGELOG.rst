^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_fullbody
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Forthcoming
-----------
* add dependency on ikpy 3.2.2
* Contributors: Séverin Lemaignan

0.1.2 (2022-09-02)
------------------
* update to hri_msgs-0.8.0
* Contributors: Séverin Lemaignan

0.1.0 (2022-08-02)
------------------
* Body position estimation without accessing depth information
* clean up launch files
* multibody depth detection
* Depth information for multipeople 3D pose estimation
* remove RegionOfInterestStamped + remove unused imports
* Multi-people detection
  We are currently working on the implementation of multi-people
  detection, usign a two stage approach: first detecting the
  people in the scene through a face detector, then estimating
  the holistic model of each one of the people detected. Work
  in progress!
* Killing robot_state_publishers with rosnode kill
  Using the robot_state_publisher_body_id.terminate method was not
  updating the rosnode list (however, it was actually terminating
  the process).
* Defined a consistency check for the face bounding box
  Testing the ROS4HRI plugin we are developing, it came out
  that there was a consistency problem in the published
  face bounding box. A consistency check on the published
  bounding box avoids this problem, verifying that its size
  matches the requirements imposed by the current image
  height and width.
* More debug parameters in launch file
  Updated the launch file to include two more debug parameters:
  stickman, to visualize simple frame representation using 3D
  Mediapipe estimation without inverse kinematic, and textual,
  to let the node print in the terminal information about the
  detected faces and bodies.
* One urdf per detected body
  hri_fullbody node now creating and managing one urdf file for each detected body.
* install as well the rviz config
* completed launch file to also start the webcam
* do not use CompressedImage directly
* import protobuf_to_dict
  The library is in the public domain: https://github.com/benhodgson/protobuf-to-dict
* Added README.md and video example of the expected result for human model visualization on RViz
* Now publishing Faces ROI, Bodies ROI, human model jointstate and 2D Skeleton message from ros4hri_msgs. Human model visualization via RViz included
* Code refined, HolistiDtector class defined. Face Detection message publishing according to ROS4HRI protocol
* Holistic node publishing skeleton message with chance of visual and textual debugging
* Contributors: Séverin Lemaignan, lorenzoferrini
