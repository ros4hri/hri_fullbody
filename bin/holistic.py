import rospy
from sensor_msgs.msg import Image 
from cv_bridge import CvBridge 
import cv2 
import numpy as np
import time
import sys
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

pub = rospy.Publisher('/holistic', Image, queue_size=1)

def callback(data):
  with mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as holistic:
      br = CvBridge()
      image_rgb = br.imgmsg_to_cv2(data)
      image_rgb.flags.writeable = False
      image_rgb = cv2.cvtColor(image_rgb,cv2.COLOR_BGR2RGB)
      results = holistic.process(image_rgb)
      image_rgb.flags.writeable = True
      mp_drawing.draw_landmarks(image_rgb, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
      mp_drawing.draw_landmarks(image_rgb, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
      start_time = time.time()
      pub.publish(CvBridge.cv2_to_imgmsg(br, image_rgb, "rgb8"))

def receive_message():
  rospy.init_node('video_sub_py', anonymous=True)
  rospy.Subscriber('/camera/color/image_raw', Image, callback, queue_size=1, buff_size = 2**24) # check name by rostopic list
  rospy.spin()
  cv2.destroyAllWindows()

if __name__ == '__main__':
  receive_message()