import rospy
import math
import numpy as np
import time
import sys
import copy

from cv_bridge import CvBridge 
import cv2 

from sensor_msgs.msg import Image, CompressedImage 
from sensor_msgs.msg import JointState
from hri_msgs.msg import Skeleton2D, PointOfInterest2D

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

from protobuf_to_dict import protobuf_to_dict

VISUAL_DEBUG = True
TEXTUAL_DEBUG = True

pub = rospy.Publisher('/holistic/visual', Image, queue_size=1)
skel_pub = rospy.Publisher('/holistic/skeleton', Skeleton2D, queue_size=1)

def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int):

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def _make_2d_skeleton_msg(original_image, pose_2d):
  skel = Skeleton2D()
  skel.header = copy.copy(original_image.header)

  # OpenVINO 2D keypoint order:
  # ['neck', 'nose',
  #   'l_sho', 'l_elb', 'l_wri', 'l_hip', 'l_knee', 'l_ank',
  #   'r_sho', 'r_elb', 'r_wri', 'r_hip', 'r_knee', 'r_ank',
  #   'r_eye', 'l_eye',
  #   'r_ear', 'l_ear'] # TO UDPDATE WITH MEDIAPIPE ORDER ASAP!!! 

  skel.skeleton = [None] * 18 #What does this do?

  #mediapipe_ordered_kpt = [PointOfInterest2D(x=x, y=y, c=v) for x,y,z,v in pose_2d]
  #print(mediapipe_ordered_kpt[0].x)

  skel.skeleton[Skeleton2D.NOSE] =           PointOfInterest2D(x=pose_2d[0].get('x'), y=pose_2d[0].get('y'), c=pose_2d[0].get('visibility'))#mediapipe_ordered_kpt[0]
  skel.skeleton[Skeleton2D.LEFT_SHOULDER] =  PointOfInterest2D(x=pose_2d[11].get('x'), y=pose_2d[11].get('y'), c=pose_2d[11].get('visibility'))#mediapipe_ordered_kpt[11]
  skel.skeleton[Skeleton2D.LEFT_ELBOW] =     PointOfInterest2D(x=pose_2d[13].get('x'), y=pose_2d[13].get('y'), c=pose_2d[13].get('visibility'))#mediapipe_ordered_kpt[13]
  skel.skeleton[Skeleton2D.LEFT_WRIST] =     PointOfInterest2D(x=pose_2d[15].get('x'), y=pose_2d[15].get('y'), c=pose_2d[15].get('visibility'))#mediapipe_ordered_kpt[15]
  skel.skeleton[Skeleton2D.LEFT_HIP] =       PointOfInterest2D(x=pose_2d[23].get('x'), y=pose_2d[23].get('y'), c=pose_2d[23].get('visibility'))#mediapipe_ordered_kpt[23]
  skel.skeleton[Skeleton2D.LEFT_KNEE] =      PointOfInterest2D(x=pose_2d[25].get('x'), y=pose_2d[25].get('y'), c=pose_2d[25].get('visibility'))#mediapipe_ordered_kpt[25]
  skel.skeleton[Skeleton2D.LEFT_ANKLE] =     PointOfInterest2D(x=pose_2d[27].get('x'), y=pose_2d[27].get('y'), c=pose_2d[27].get('visibility'))#mediapipe_ordered_kpt[27]
  skel.skeleton[Skeleton2D.RIGHT_SHOULDER] = PointOfInterest2D(x=pose_2d[12].get('x'), y=pose_2d[12].get('y'), c=pose_2d[12].get('visibility'))#mediapipe_ordered_kpt[12]
  skel.skeleton[Skeleton2D.RIGHT_ELBOW] =    PointOfInterest2D(x=pose_2d[14].get('x'), y=pose_2d[14].get('y'), c=pose_2d[14].get('visibility'))#mediapipe_ordered_kpt[14]
  skel.skeleton[Skeleton2D.RIGHT_WRIST] =    PointOfInterest2D(x=pose_2d[16].get('x'), y=pose_2d[16].get('y'), c=pose_2d[16].get('visibility'))#mediapipe_ordered_kpt[16]
  skel.skeleton[Skeleton2D.RIGHT_HIP] =      PointOfInterest2D(x=pose_2d[24].get('x'), y=pose_2d[24].get('y'), c=pose_2d[24].get('visibility'))#mediapipe_ordered_kpt[24]
  skel.skeleton[Skeleton2D.RIGHT_KNEE] =     PointOfInterest2D(x=pose_2d[26].get('x'), y=pose_2d[26].get('y'), c=pose_2d[26].get('visibility'))#mediapipe_ordered_kpt[26]
  skel.skeleton[Skeleton2D.RIGHT_ANKLE] =    PointOfInterest2D(x=pose_2d[28].get('x'), y=pose_2d[28].get('y'), c=pose_2d[28].get('visibility'))#mediapipe_ordered_kpt[28]
  skel.skeleton[Skeleton2D.RIGHT_EYE] =      PointOfInterest2D(x=pose_2d[5].get('x'), y=pose_2d[5].get('y'), c=pose_2d[5].get('visibility'))#mediapipe_ordered_kpt[5]
  skel.skeleton[Skeleton2D.LEFT_EYE] =       PointOfInterest2D(x=pose_2d[2].get('x'), y=pose_2d[2].get('y'), c=pose_2d[2].get('visibility'))#mediapipe_ordered_kpt[2]
  skel.skeleton[Skeleton2D.RIGHT_EAR] =      PointOfInterest2D(x=pose_2d[8].get('x'), y=pose_2d[8].get('y'), c=pose_2d[8].get('visibility'))#mediapipe_ordered_kpt[8]
  skel.skeleton[Skeleton2D.LEFT_EAR] =       PointOfInterest2D(x=pose_2d[7].get('x'), y=pose_2d[7].get('y'), c=pose_2d[7].get('visibility'))#mediapipe_ordered_kpt[7]
  skel.skeleton[Skeleton2D.NECK] =           PointOfInterest2D((skel.skeleton[Skeleton2D.LEFT_SHOULDER].x+skel.skeleton[Skeleton2D.RIGHT_SHOULDER].x)/2,\
                                                                (skel.skeleton[Skeleton2D.LEFT_SHOULDER].y+skel.skeleton[Skeleton2D.RIGHT_SHOULDER].y)/2,\
                                                                 min(skel.skeleton[Skeleton2D.LEFT_SHOULDER].c, skel.skeleton[Skeleton2D.RIGHT_SHOULDER].c))  #mediapipe_ordered_kpt[0] #I have to update with the avg between left and right shoulder

  return skel

def _get_bounding_box_limits(face_landmarks, image_width, image_height):
  x_max = 0.0
  y_max = 0.0
  x_min = 1.0
  y_min = 1.0
  #for result in results:
  for data_point in face_landmarks:
    if x_max < data_point.x:
      x_max = data_point.x
    if y_max < data_point.y:
      y_max = data_point.y
    if x_min > data_point.x:
      x_min = data_point.x
    if y_min > data_point.y:
      y_min = data_point.y
  x_min, y_min = _normalized_to_pixel_coordinates(x_min, y_min, image_width, image_height)
  x_max, y_max = _normalized_to_pixel_coordinates(x_max, y_max, image_width, image_height)
  return x_min, y_min, x_max, y_max

def callback(data):
  br = CvBridge()
  image_rgb = br.compressed_imgmsg_to_cv2(data)
  img_height, img_width, _ = image_rgb.shape
  image_rgb.flags.writeable = False
  image_rgb = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB)
  results = holistic.process(image_rgb)
  image_rgb.flags.writeable = True
  mp_drawing.draw_landmarks(image_rgb, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
  mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
  i = 0
  if hasattr(results.face_landmarks, 'landmark'):
    x_min, y_min, x_max, y_max = _get_bounding_box_limits(results.face_landmarks.landmark, img_width, img_height)
    if TEXTUAL_DEBUG:
      print('FACE: bb limits ==> x_min =', x_min, 'y_min = ', y_min, 'x_max = ', x_max, 'y_max = ', y_max)
    if VISUAL_DEBUG:
      image_rgb = cv2.rectangle(image_rgb, (max(x_min, 0), max(y_min, 0)), (min(x_max, img_width-1), min(y_max, img_height)), (0, 0, 255), 3)
  if hasattr(results.pose_landmarks, 'landmark'):
    pose_keypoints = protobuf_to_dict(results.pose_landmarks)
    pose_kpt = pose_keypoints.get('landmark')
    skel_msg = _make_2d_skeleton_msg(data, pose_kpt)
    skel_pub.publish(skel_msg)
    x_min, y_min, x_max, y_max = _get_bounding_box_limits(skel_msg.skeleton, img_width, img_height)
    if TEXTUAL_DEBUG:
      print('BODY: bb limits ==> x_min =', x_min, 'y_min = ', y_min, 'x_max = ', x_max, 'y_max = ', y_max)
    if VISUAL_DEBUG:
      image_rgb = cv2.rectangle(image_rgb, (max(x_min, 0), max(y_min, 0)), (min(x_max, img_width-1), min(y_max, img_height)), (0, 255, 0), 3)
  pub.publish(CvBridge.cv2_to_imgmsg(br, image_rgb, "rgb8"))

def receive_message():
  rospy.init_node('video_sub_py', anonymous=True)
  if rospy.has_param('~rgbImageTopic'):
    print('********** Found rgb image topic parameter: ', rospy.get_param('~rgbImageTopic'), ' *****************')
  rgbImageTopic = rospy.get_param('~rgbImageTopic')
  rospy.Subscriber(rgbImageTopic, CompressedImage, callback, queue_size=1, buff_size = 2**24) # check name by rostopic list
  rospy.spin()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
      receive_message()





