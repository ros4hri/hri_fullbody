^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_fullbody
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

0.1.10 (2023-07-14)
-------------------
* reset mediapipe backend in case of matrix inversion error
* Contributors: Luka Juricic

0.1.9 (2023-07-05)
------------------
* fix bug in left ear tragion publish
* change RoI message type to hri_msgs/NormalizedRegionOfInterest2D
* Contributors: Luka Juricic

0.1.8 (2023-05-23)
------------------
* disable publishing of face_XXX and gaze_XXX frames
  See comment in code for rationale
* Contributors: Séverin Lemaignan

0.1.7 (2023-05-18)
------------------
* diagnostic_msgs: exec_depend -> depend
* Contributors: Séverin Lemaignan

0.1.6 (2023-05-18)
------------------
* add diagnostics
* Contributors: lukajuricic

0.1.5 (2023-03-08)
------------------
* ensure mediapipe can not be called from 2 threads in parallel
  This would cause internal mediapipe errors related to non-monotonic
  timestamps
* Contributors: Séverin Lemaignan

0.1.4 (2023-02-03)
------------------
* Merge branch 'devel' into 'master'
  Revert "have threshold params"
  See merge request ros4hri/hri_fullbody!7
* Merge branch 'master' into 'devel'
  # Conflicts:
  #   src/hri_fullbody/fullbody_detector.py
* Revert "have threshold params"
  This reverts commit 78264248652879a1b572397e8edeee6202685bc6
  and commit 5cc5c34f7a93f8030623ff78945095778374cd36.
* Merge branch 'devel' into 'master'
  Introduced requirements.txt
  See merge request ros4hri/hri_fullbody!6
* Merge branch 'uncalibrated_monocular_camera' into 'master'
  Correctly managing uncalibrated camera
  See merge request ros4hri/hri_fullbody!5
* Introduced requirements.txt
  Introduced requirements.txt. Currently, the only requirement
  specified is ikpy. Using the requirements.txt file syntax,
  it was possible to specify the requested version, that is,
  3.2.2
* Enabled human jointstate estimation for uncalibrated RGB cameras
  It was previously not possible to estimate human joint angles
  when the RGB camera was not calibrated. Additionally, the node
  is now displaying a warning message when the received camera
  intrinsic matrix has all zero values, which usually means that
  it needs to be calibrated.
* [Doc] Specified camera calibration requirement
  It is now specified in the README.md file that camera calibration
  is required to estimate body depth without using a depth sensor.
* REP-155 compliance: skeleton must be published on /skeleton2d topic
* Merge branch 'master' of gitlab:ros4hri/hri_fullbody
* have threshold params
* have threshold params
* generate only letter-based IDs (for consistency with other nodes)
* code formatting
* Contributors: Séverin Lemaignan, lorenzoferrini, saracooper

0.1.3 (2022-09-27)
------------------
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
