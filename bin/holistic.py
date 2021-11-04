import math
import uuid
import numpy as np
import time
import sys
import copy
from collections import deque

import rospy
from sensor_msgs.msg import Image, CompressedImage, CameraInfo, RegionOfInterest 
from sensor_msgs.msg import JointState
from hri_msgs.msg import Skeleton2D, PointOfInterest2D, IdsList, RegionOfInterestStamped

from cv_bridge import CvBridge 
import cv2 

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

from protobuf_to_dict import protobuf_to_dict

VISUAL_DEBUG = True
TEXTUAL_DEBUG = False
PIXEL_THRESHOLD = 10000.0 #Maybe we could set this basing the evaluation on some ground truth
CACHE_SIZE = 30 #Maybe we could set this basing the evaluation on some ground truth

# if set to true, face IDs will be generated as a sequence of integers,
# starting at 00001.
# Otherwise, face IDs will be a random set of 5 characters in [0-9a-f]
DETERMINISTIC_ID = True

PREALLOCATE_PUBLISHERS = DETERMINISTIC_ID
PREALLOCATION_SIZE = 150

# nb of pixels between the centers of to successive regions of interest to
# consider they belong to the same person
MAX_ROIS_DISTANCE = 100

# max scale factor between two successive regions of interest to consider they
# belong to the same person
MAX_SCALING_ROIS = 1.2


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int):

    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px

def _make_2d_skeleton_msg(header, pose_2d):
  skel = Skeleton2D()
  skel.header = header

  # Mediapipe 2D keypoint order:
  # ['nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
  #   'right_eye_inner', 'right_eye', 'right_eye_outer',
  #   'left_ear', 'right_ear', 'mouth_left', 'mouth_right',
  #   'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
  #   'left_wrist', 'right_wrist', 'left_pinky', 'right_pinky',
  #   'left_index', 'right_index', 'left_thumb', 'right_thumb',
  #   'left_hip', 'right_hip', 'left_knee', 'right_knee',
  #   'left_ankle', 'right_ankle', 'left_heel', 'right_heel',
  #   'left_foot_index', 'right_foot_index']  

  skel.skeleton = [None] * 18 #What does this do?

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

def _distance_rois(bb1, bb2):
  x1, y1 = bb1.x_offset + bb1.width/2, bb1.y_offset + bb1.height/2
  x2, y2 = bb2.x_offset + bb2.width/2, bb2.y_offset + bb2.height/2

  return (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2)

class HolisticDetector:
  def __init__(self):

    self.detector = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    self.x_min_face = 1.00
    self.y_min_face = 1.00
    self.x_max_face = 0.00
    self.y_max_face = 0.00

    self.x_min_body = 1.00
    self.y_min_body = 1.00
    self.x_max_body = 0.00
    self.y_max_body = 0.00

    # ID -> (publisher, RoI, nb_frames_visible)
    self.detectedFaces = {}

    self.visual_pub = rospy.Publisher('/holistic/visual', Image, queue_size=1)
    self.skel_pub = rospy.Publisher('/holistic/skeleton', Skeleton2D, queue_size=1)

    self.br = CvBridge()

    self.last_id = 1

    if PREALLOCATE_PUBLISHERS:
        self.face_pubs = {"%05d" % i : rospy.Publisher("/humans/faces/%05d/roi" % i, RegionOfInterestStamped, queue_size=1) for i in range(1,PREALLOCATION_SIZE)}

    self.faces_pub = rospy.Publisher("/humans/faces/tracked", IdsList, queue_size=1)

  def find_previous_match(self, bb):
    for id, value in self.detectedFaces.items():
        _, prev_bb, _ = value
        rois_distance = _distance_rois(prev_bb, bb)
        #print('ROIs distance: ', rois_distance, '\tWidth Scaling: ', prev_bb.width/bb.width, '\tHeight Scaling: ', prev_bb.height/bb.height)
        if rois_distance < MAX_ROIS_DISTANCE * MAX_ROIS_DISTANCE \
        and 1/MAX_SCALING_ROIS < prev_bb.width/bb.width < MAX_SCALING_ROIS \
        and 1/MAX_SCALING_ROIS < prev_bb.height/bb.height < MAX_SCALING_ROIS:
            return id
    return None

  def generate_face_id(self):
    if DETERMINISTIC_ID:
        id_str = "%05d" % self.last_id
        self.last_id = (self.last_id + 1) % 10000
        return id_str

    else:
        return str(uuid.uuid4())[:5] # for a 5 char long ID

  def generate_tmp_face_id(self):
    return str(uuid.uuid4())[:5] # for a 5 char long ID

  def detect(self, image_rgb, header):

    img_height, img_width, _ = image_rgb.shape

    image_rgb.flags.writeable = False
    image_rgb = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB) 
    results = self.detector.process(image_rgb)
    image_rgb.flags.writeable = True

    currentFaces = {}

    if hasattr(results.face_landmarks, 'landmark'):
      self.x_min_face, self.y_min_face, self.x_max_face, self.y_max_face = _get_bounding_box_limits(results.face_landmarks.landmark, img_width, img_height)
      bb = RegionOfInterest(self.x_min_face, self.y_min_face, self.x_max_face-self.x_min_face, self.y_max_face-self.y_min_face, True)
      id = self.find_previous_match(bb)
      if id:
        # we re-detect a face: if it is a 2nd frame, we create a publisher for it.
        if not self.detectedFaces[id][0]:
          final_id = self.generate_face_id()
          rospy.loginfo("New face [face_%s]" % final_id)

          if PREALLOCATE_PUBLISHERS:
            pub = self.face_pubs[final_id]
          else:
            pub = rospy.Publisher("/humans/faces/%s/roi" % final_id, RegionOfInterestStamped, queue_size=1)

          currentFaces[final_id] = (pub, bb, self.detectedFaces[id][2] + 1)
        else:
          currentFaces[id] = (self.detectedFaces[id][0], bb, self.detectedFaces[id][2] + 1)
      else:
        # we 'provisionally' store the face - we'll create a publisher and start publishing only if we see the
        # face a second time
        id = self.generate_tmp_face_id()
        currentFaces[id] = (None, bb, 1)
        print('New face! Id:', id)
      for id, value in self.detectedFaces.items():
        if id not in currentFaces:
          pub, _, nb_frames = value
          if pub:
            rospy.loginfo("Face [face_%s] lost. It remained visible for %s frames" % (id, nb_frames))
            pub.unregister()

      self.detectedFaces = currentFaces

      list_ids = []

      for id, value in self.detectedFaces.items():
        pub, bb, _ = value
        if pub:
          list_ids.append(str(id))
          pub.publish(RegionOfInterestStamped(header, bb))

      self.faces_pub.publish(IdsList(header, list_ids))

      if TEXTUAL_DEBUG:
        print('FACE: bb limits ==> x_min =', self.x_min_face, 'y_min = ', self.y_min_face, 'x_max = ', self.x_max_face, 'y_max = ', self.y_max_face)        
      if VISUAL_DEBUG:
        image_rgb = cv2.rectangle(image_rgb, (max(self.x_min_face, 0), max(self.y_min_face, 0)), (min(self.x_max_face, img_width-1), min(self.y_max_face, img_height-1)), (0, 0, 255), 3)

    if hasattr(results.pose_landmarks, 'landmark'):
      pose_keypoints = protobuf_to_dict(results.pose_landmarks)
      pose_kpt = pose_keypoints.get('landmark')
      skel_msg = _make_2d_skeleton_msg(header, pose_kpt)
      self.skel_pub.publish(skel_msg)
      self.x_min_body, self.y_min_body, self.x_max_body, self.y_max_body = _get_bounding_box_limits(skel_msg.skeleton, img_width, img_height)
      if hasattr(results.face_landmarks, 'landmark'):
        self.x_min_body = min(self.x_min_body, self.x_min_face)
        self.y_min_body = min(self.y_min_body, self.y_min_face)
        self.x_max_body = max(self.x_max_body, self.x_max_face)
        self.y_max_body = max(self.y_max_body, self.y_max_face)
      if TEXTUAL_DEBUG:
        print('BODY: bb limits ==> x_min =', self.x_min_body, 'y_min = ', self.y_min_body, 'x_max = ', self.x_max_body, 'y_max = ', self.y_max_body)
      if VISUAL_DEBUG:
        image_rgb = cv2.rectangle(image_rgb, (max(self.x_min_body, 0), max(self.y_min_body, 0)), (min(self.x_max_body, img_width-1), min(self.y_max_body, img_height-1)), (255, 0, 0), 3)

    if VISUAL_DEBUG:  
      self.visual_pub.publish(CvBridge.cv2_to_imgmsg(self.br, image_rgb, "rgb8"))

  def image_callback(self, data):

    br = CvBridge()
    image_rgb = br.compressed_imgmsg_to_cv2(data)
    header = copy.copy(data.header)
    self.detect(image_rgb, header)

  def init_node(self, camera_topic):

    rospy.Subscriber(camera_topic, CompressedImage, self.image_callback, queue_size=1, buff_size = 2**24)


if __name__ == '__main__':
  rospy.init_node('holistic_estimation', anonymous=True)
  if rospy.has_param('~rgbImageTopic'):
    print('********** Found rgb image topic parameter: ', rospy.get_param('~rgbImageTopic'), ' *****************')
  imageTopic = rospy.get_param('~rgbImageTopic')
  dtct = HolisticDetector()
  dtct.init_node(imageTopic)
  rospy.spin()
  cv2.destroyAllWindows()





