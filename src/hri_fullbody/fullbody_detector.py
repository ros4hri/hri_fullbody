import io
import os
from ikpy import chain
from hri_fullbody.jointstate import compute_jointstate, \
    HUMAN_JOINT_NAMES, compute_jointstate
from hri_fullbody.rs_to_depth import rgb_to_xyz  # SITW
from hri_fullbody.urdf_generator import make_urdf_human
from hri_fullbody.protobuf_to_dict import protobuf_to_dict
from hri_fullbody.one_euro_filter import OneEuroFilter
from hri_fullbody.face_pose_estimation import face_pose_estimation
import math
import numpy as np
import sys
import copy
import subprocess

import rosparam
import rospy
import tf

from sensor_msgs.msg import Image, CameraInfo, RegionOfInterest
from sensor_msgs.msg import JointState
from hri_msgs.msg import Skeleton2D, PointOfInterest2D, IdsList
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Float64
from geometry_msgs.msg import TwistStamped, Point

from cv_bridge import CvBridge
import cv2

import mediapipe as mp
mp_face_detection = mp.solutions.face_detection
mp_holistic = mp.solutions.holistic
mp_pose = mp.solutions.pose

# if set to true, face IDs will be generated as a sequence of integers,
# starting at 00001.
# Otherwise, face IDs will be a random set of 5 characters in [0-9a-f]
DETERMINISTIC_ID = False

PREALLOCATE_PUBLISHERS = DETERMINISTIC_ID
PREALLOCATION_SIZE = 150

# nb of pixels between the centers of to successive regions of interest to
# consider they belong to the same person
MAX_ROIS_DISTANCE = 100

# max scale factor between two successive regions of interest to consider they
# belong to the same person
MAX_SCALING_ROIS = 1.2

# Parameter definition for the bounding box resizer
BB_MULT = 0.0
FACE_BB_MULT = 0.5

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

ros4hri_to_mediapipe = [None] * 18

ros4hri_to_mediapipe[Skeleton2D.NOSE] = 0
ros4hri_to_mediapipe[Skeleton2D.LEFT_SHOULDER] = 11
ros4hri_to_mediapipe[Skeleton2D.LEFT_ELBOW] = 13
ros4hri_to_mediapipe[Skeleton2D.LEFT_WRIST] = 15
ros4hri_to_mediapipe[Skeleton2D.RIGHT_SHOULDER] = 12
ros4hri_to_mediapipe[Skeleton2D.RIGHT_ELBOW] = 14
ros4hri_to_mediapipe[Skeleton2D.RIGHT_WRIST] = 16
ros4hri_to_mediapipe[Skeleton2D.LEFT_HIP] = 23
ros4hri_to_mediapipe[Skeleton2D.LEFT_KNEE] = 25
ros4hri_to_mediapipe[Skeleton2D.LEFT_ANKLE] = 27
ros4hri_to_mediapipe[Skeleton2D.RIGHT_HIP] = 24
ros4hri_to_mediapipe[Skeleton2D.RIGHT_KNEE] = 26
ros4hri_to_mediapipe[Skeleton2D.RIGHT_ANKLE] = 28
ros4hri_to_mediapipe[Skeleton2D.RIGHT_EYE] = 5
ros4hri_to_mediapipe[Skeleton2D.LEFT_EYE] = 2
ros4hri_to_mediapipe[Skeleton2D.LEFT_EAR] = 7
ros4hri_to_mediapipe[Skeleton2D.RIGHT_EAR] = 8


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

    skel.skeleton = [None] * 18

    for idx, human_joint in enumerate(ros4hri_to_mediapipe):
        if human_joint is not None:
            skel.skeleton[idx] = PointOfInterest2D(
                x=pose_2d[human_joint].get('x'),
                y=pose_2d[human_joint].get('y'),
                c=pose_2d[human_joint].get('visibility'))

    # There is no Neck landmark in Mediapipe pose estimation
    # However, we can think of the neck point as the average
    # point between left and right shoulder.
    skel.skeleton[Skeleton2D.NECK] = PointOfInterest2D(
        (
            skel.skeleton[Skeleton2D.LEFT_SHOULDER].x
            + skel.skeleton[Skeleton2D.RIGHT_SHOULDER].x
        )
        / 2,
        (
            skel.skeleton[Skeleton2D.LEFT_SHOULDER].y
            + skel.skeleton[Skeleton2D.RIGHT_SHOULDER].y
        )
        / 2,
        min(skel.skeleton[Skeleton2D.LEFT_SHOULDER].c,
            skel.skeleton[Skeleton2D.RIGHT_SHOULDER].c))

    return skel


def _get_bounding_box_limits(landmarks, image_width, image_height):
    x_max = 0.0
    y_max = 0.0
    x_min = 1.0
    y_min = 1.0
    # for result in results:
    for data_point in landmarks:
        if x_max < data_point.x:
            x_max = data_point.x
        if y_max < data_point.y:
            y_max = data_point.y
        if x_min > data_point.x:
            x_min = data_point.x
        if y_min > data_point.y:
            y_min = data_point.y

    #delta_x = x_max - x_min
    #delta_y = y_max - y_min

    #x_min -= BB_MULT*delta_x
    #y_min -= BB_MULT*delta_y
    #x_max += BB_MULT*delta_x
    #y_max += BB_MULT*delta_y
    x_min, y_min = _normalized_to_pixel_coordinates(
        x_min, y_min, image_width, image_height)
    x_max, y_max = _normalized_to_pixel_coordinates(
        x_max, y_max, image_width, image_height)
    return x_min, y_min, x_max, y_max


def _distance_rois(bb1, bb2):
    x1, y1 = bb1.x_offset + bb1.width/2, bb1.y_offset + bb1.height/2
    x2, y2 = bb2.x_offset + bb2.width/2, bb2.y_offset + bb2.height/2

    return (x1-x2) * (x1-x2) + (y1-y2) * (y1-y2)


class FullbodyDetector:

    def __init__(self,
                 use_depth,
                 textual_debug,
                 stickman_debug,
                 body_id,
                 single_body = False):

        self.use_depth = use_depth
        self.textual_debug = textual_debug
        self.stickman_debug = stickman_debug
        self.single_body = single_body

        self.detector = mp_holistic.Holistic(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.from_depth_image = False

        self.x_min_face = 1.00
        self.y_min_face = 1.00
        self.x_max_face = 0.00
        self.y_max_face = 0.00

        self.x_min_body = 1.00
        self.y_min_body = 1.00
        self.x_max_body = 0.00
        self.y_max_body = 0.00

        self.body_position_estimation = [None] * 3
        self.trans_vec = [None] * 3

        js_topic = '/joint_states_' + body_id
        skel_topic = '/humans/bodies/' + body_id + '/skeleton'

        self.skel_pub = rospy.Publisher(
            skel_topic, Skeleton2D, queue_size=1)
        self.js_pub = rospy.Publisher(
            js_topic, JointState, queue_size=1)

        self.br = CvBridge()

        # 1. generate a URDF model for this body, and set it on the
        #    ROS parameter server
        self.body_id = body_id
        self.urdf = make_urdf_human(self.body_id)
        rospy.loginfo("Setting URDF description for body"
                      "<%s> (param name: human_description_%s)" % (
                          self.body_id, self.body_id))
        self.human_description = "human_description_%s" % self.body_id
        rosparam.set_param_raw(self.human_description, self.urdf)

        self.urdf_file = io.StringIO(self.urdf)
        self.r_arm_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "r_y_shoulder_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])
        self.urdf_file.seek(0)
        self.l_arm_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "l_y_shoulder_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])
        self.urdf_file.seek(0)
        self.r_leg_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "r_y_hip_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])
        self.urdf_file.seek(0)
        self.l_leg_chain = chain.Chain.from_urdf_file(
            self.urdf_file,
            base_elements=[
                "l_y_hip_%s" % self.body_id],
            base_element_type="joint",
            active_links_mask=[False, True, True, True, True, False])

        self.ik_chains = {}  # maps a body id to the IKpy chains
        self.ik_chains[self.body_id] = [
            self.r_arm_chain,
            self.l_arm_chain,
            self.r_leg_chain,
            self.l_leg_chain
        ]

        # 2. spawn a new robot_state_publisher process, which will publish the
        #    TF frames for this body
        rospy.loginfo(
            "Spawning a instance of robot_state_publisher for this body...")
        cmd = ["rosrun", "robot_state_publisher", "robot_state_publisher",
               "__name:=robot_state_publisher_body_%s" % self.body_id,
               "joint_states:=%s" % js_topic,
               "robot_description:=%s" % self.human_description,
               "_publish_frequency:=25",
               "_use_tf_static:=false"
               ]
        rospy.loginfo("Executing: %s" % " ".join(cmd))
        self.proc = subprocess.Popen(cmd,
                                     stdout=sys.stdout,
                                     stderr=subprocess.STDOUT)

        self.tb = tf.TransformBroadcaster()
        self.one_euro_filter = [None] * 3
        self.one_euro_filter_dot = [None] * 3

        if not single_body:
            self.image_subscriber = Subscriber(
                                        "/humans/bodies/"+self.body_id+"/crop",
                                        Image,
                                        queue_size=1,
                                        buff_size=2**24)
        else:
            self.image_subscriber = Subscriber(
                                        "/image",
                                        Image,
                                        queue_size=1,
                                        buff_size=2**24)

        if self.use_depth and not single_body:
            self.tss = ApproximateTimeSynchronizer(
                [
                    self.image_subscriber,
                    Subscriber(
                        "/camera_info",
                        CameraInfo,
                        queue_size=1),
                    Subscriber(
                        "/humans/bodies/"+self.body_id+"/roi",
                        RegionOfInterest,
                        queue_size=1),
                    Subscriber(
                        "/depth_image",
                        Image,
                        queue_size=1,
                        buff_size=2**24),
                    Subscriber(
                        "/depth_info",
                        CameraInfo,
                        queue_size=1)
                ],
                10,
                0.1,
                allow_headerless=True
            )
            self.tss.registerCallback(self.image_callback_depth)
        elif not self.use_depth and not single_body:
            self.tss = ApproximateTimeSynchronizer(
                [
                    self.image_subscriber,
                    Subscriber(
                        "/camera_info",
                        CameraInfo,
                        queue_size=5)
                ],
                10,
                0.2
            )
            self.tss.registerCallback(self.image_callback_rgb)
        elif self.use_depth and single_body:
            # Here the code to detect one person only with depth information
            self.tss = ApproximateTimeSynchronizer(
                [
                    self.image_subscriber,
                    Subscriber(
                        "/camera_info",
                        CameraInfo,
                        queue_size=1),
                    Subscriber(
                        "/depth_image",
                        Image,
                        queue_size=1,
                        buff_size=2**24),
                    Subscriber(
                        "/depth_info",
                        CameraInfo,
                        queue_size=1)
                ],
                10,
                0.1,
                allow_headerless=True
            )
            self.tss.registerCallback(self.image_callback_depth_single_person)
        else:
            self.tss = ApproximateTimeSynchronizer(
                [
                    self.image_subscriber,
                    Subscriber(
                        "/camera_info",
                        CameraInfo,
                        queue_size=5)
                ],
                10,
                0.2
            )
            self.tss.registerCallback(self.image_callback_rgb)

        if single_body:
            self.ids_pub = rospy.Publisher(
                "/humans/bodies/tracked",
                IdsList,
                queue_size=1)
            self.roi_pub = rospy.Publisher(
                "/humans/bodies/"+body_id+"/roi",
                RegionOfInterest,
                queue_size=1)

        self.body_filtered_position = [None] * 3
        self.body_filtered_position_prev = [None] * 3
        self.body_unfiltered_position = [None] * 3 # Debugging purpose
        self.body_vel_estimation = [None] * 3
        self.body_vel_estimation_filtered = [None] * 3

        self.position_msg = [Point(), Point()]
        # [0] = filtered, [1] = unfiltered
        
        filtered_position_topic = "/humans/bodies/"+body_id+"/f_position"
        self.body_filtered_position_pub = rospy.Publisher( 
            filtered_position_topic,
            Point,
            queue_size=1) # Debugging purpose
        unfiltered_position_topic = "/humans/bodies/"+body_id+"/uf_position"
        self.body_unfiltered_position_pub = rospy.Publisher( 
            unfiltered_position_topic,
            Point,
            queue_size=1) # Debugging purpose
        self.speed_msg = TwistStamped()
        self.speed_msg.header.frame_id = "body_"+body_id
        twist_topic = "/humans/bodies/"+body_id+"/velocity"
        self.speed_pub = rospy.Publisher(
            twist_topic,
            TwistStamped,
            queue_size=1)

        self.image_info_sub = rospy.Subscriber(
            "camera_info", CameraInfo, self.info_callback
        )

    def unregister(self):
        if rospy.has_param(self.human_description):
            rospy.delete_param(self.human_description)
            rospy.loginfo('Deleted parameter %s', self.human_description)
        os.system("rosnode kill /robot_state_publisher_body_"+self.body_id)
        rospy.logwarn('unregistered %s', self.body_id)

    def info_callback(self, cameraInfo):

        if not hasattr(self, 'cameraInfo'):
            self.cameraInfo = cameraInfo

            self.K = np.zeros((3, 3), np.float32)
            self.K[0][0:3] = self.cameraInfo.K[0:3]
            self.K[1][0:3] = self.cameraInfo.K[3:6]
            self.K[2][0:3] = self.cameraInfo.K[6:9]

            self.f_x = self.K[0][0]
            self.f_y = self.K[1][1]
            self.c_x = self.K[0][2]
            self.c_y = self.K[1][2]

    def face_to_body_position_estimation(self, skel_msg):
        body_px = [(skel_msg.skeleton[Skeleton2D.LEFT_HIP].x \
                    + skel_msg.skeleton[Skeleton2D.RIGHT_HIP].x) \
                    / 2,
                   (skel_msg.skeleton[Skeleton2D.LEFT_HIP].y \
                    + skel_msg.skeleton[Skeleton2D.RIGHT_HIP].y) \
                    / 2]
        body_px = _normalized_to_pixel_coordinates(body_px[0], 
                                                   body_px[1], 
                                                   self.img_width,
                                                   self.img_height)

        d_x = np.sqrt((self.trans_vec[0]/1000)**2 \
                      +(self.trans_vec[1]/1000)**2 \
                      +(self.trans_vec[2]/1000)**2)

        x = body_px[0]-self.c_x
        y = body_px[1]-self.c_y

        Z = self.f_x*d_x/(np.sqrt(x**2 + self.f_x**2))
        X = x*Z/self.f_x
        Y = y*Z/self.f_y
        return [X, Y, Z]


    def make_jointstate(
            self,
            body_id,
            pose_3d,
            pose_2d,
            header):

        js = JointState()
        js.header = copy.copy(header)
        js.name = [jn + "_%s" % body_id for jn in HUMAN_JOINT_NAMES]

        torso = np.array([
            -(
                pose_3d[23].get('z')
                + pose_3d[24].get('z')
            )
            / 2,
            (
                pose_3d[23].get('x')
                + pose_3d[24].get('x')
            )
            / 2,
            -(
                pose_3d[23].get('y')
                + pose_3d[24].get('y')
            )
            / 2
        ])
        l_shoulder = np.array([
            -pose_3d[11].get('z'),
            pose_3d[11].get('x'),
            -pose_3d[11].get('y')-0.605
        ])
        l_elbow = np.array([
            -pose_3d[13].get('z'),
            pose_3d[13].get('x'),
            -pose_3d[13].get('y')-0.605
        ])
        l_wrist = np.array([
            -pose_3d[15].get('z'),
            pose_3d[15].get('x'),
            -pose_3d[15].get('y')-0.605
        ])
        l_ankle = np.array([
            -pose_3d[27].get('z'),
            pose_3d[27].get('x'),
            -pose_3d[27].get('y')
        ])
        r_shoulder = np.array([
            -pose_3d[12].get('z'),
            pose_3d[12].get('x'),
            -pose_3d[12].get('y')-0.605
        ])
        r_elbow = np.array([
            -pose_3d[14].get('z'),
            pose_3d[14].get('x'),
            -pose_3d[14].get('y')-0.605
        ])
        r_wrist = np.array([
            -pose_3d[16].get('z'),
            pose_3d[16].get('x'),
            -pose_3d[16].get('y')-0.605
        ])
        r_ankle = np.array([
            -pose_3d[28].get('z'),
            pose_3d[28].get('x'),
            -pose_3d[28].get('y')
        ])
        nose = np.array([
            -pose_3d[0].get('z'),
            pose_3d[0].get('x'),
            -pose_3d[0].get('y')
        ])
        feet = np.array([
            -(
                pose_3d[32].get('z')
                + pose_3d[31].get('z')
            )
            / 2,
            (
                pose_3d[32].get('x')
                + pose_3d[31].get('x')
            )
            / 2,
            -(
                pose_3d[32].get('y')
                + pose_3d[31].get('y')
            )
            / 2
        ])

        ### depth and rotation ###

        torso_px = _normalized_to_pixel_coordinates(
            (pose_2d[23].get('x')+pose_2d[24].get('x'))/2,
            (pose_2d[23].get('y')+pose_2d[24].get('y'))/2,
            self.img_width,
            self.img_height)
        theta = np.arctan2(pose_3d[24].get('x'), -pose_3d[24].get('z'))
        if self.use_depth:
            torso_res = rgb_to_xyz(
                torso_px[0],
                torso_px[1],
                self.rgb_info,
                self.depth_info,
                self.image_depth,
                self.roi.x_offset,
                self.roi.y_offset
            )
        elif self.body_position_estimation[0]:
            torso_res = self.body_position_estimation
        else:
            torso_res = np.array([0, 0, 0])

        ### Publishing tf transformations ###

        t = header.stamp.to_sec()

        if not self.one_euro_filter[0] and self.use_depth:
            self.one_euro_filter[0] = OneEuroFilter(
                t, 
                torso_res[2], 
                beta=0.05, 
                d_cutoff=0.5, 
                min_cutoff=0.3)
            self.one_euro_filter[1] = OneEuroFilter(
                t, 
                torso_res[0], 
                beta=0.05, 
                d_cutoff=0.5, 
                min_cutoff=0.3)
            self.body_filtered_position[0] = torso_res[2]
            self.body_filtered_position[1] = torso_res[0]
        elif self.use_depth:
            self.body_filtered_position_prev[0] = \
                self.body_filtered_position[0]
            self.body_filtered_position_prev[1] = \
                self.body_filtered_position[1]
            self.body_filtered_position[0], t_e = \
                self.one_euro_filter[0](t, torso_res[2])
            self.body_filtered_position[1], _ = \
                self.one_euro_filter[1](t, torso_res[0])

            self.position_msg[0].x = self.body_filtered_position[0]
            self.position_msg[0].y = self.body_filtered_position[1]
            self.position_msg[1].x = torso_res[2]
            self.position_msg[1].y = torso_res[0]
            self.body_filtered_position_pub.publish(self.position_msg[0])
            self.body_unfiltered_position_pub.publish(self.position_msg[1])

            self.body_vel_estimation[0] = \
                (self.body_filtered_position[0] \
                 - self.body_filtered_position_prev[0])/t_e
            self.body_vel_estimation[1] = \
                (self.body_filtered_position[1] \
                 - self.body_filtered_position_prev[1])/t_e

            if not self.one_euro_filter_dot[0]:
                self.one_euro_filter_dot[0] = OneEuroFilter(
                    t, 
                    self.body_vel_estimation[0], 
                    beta=0.2, 
                    d_cutoff=0.2, 
                    min_cutoff=0.5)
                self.one_euro_filter_dot[1] = OneEuroFilter(
                    t, 
                    self.body_vel_estimation[1], 
                    beta=0.2, 
                    d_cutoff=0.2, 
                    min_cutoff=0.5)
            else:
                self.body_vel_estimation_filtered[0], _ = \
                    self.one_euro_filter_dot[0](t, self.body_vel_estimation[0])
                self.body_vel_estimation_filtered[1], _ = \
                    self.one_euro_filter_dot[1](t, self.body_vel_estimation[1])
                self.speed_msg.twist.linear.x = \
                    -self.body_vel_estimation_filtered[0]
                self.speed_msg.twist.linear.y = \
                    self.body_vel_estimation_filtered[1]
                self.speed_pub.publish(self.speed_msg)

        if not self.use_depth:
            self.tb.sendTransform(
                (torso_res[0], 0.0, torso_res[2]),
                tf.transformations.quaternion_from_euler(
                    np.pi/2,
                    -theta,
                    0
                ),
                header.stamp,
                "body_%s" % body_id,
                header.frame_id
            )
        else:
            self.tb.sendTransform(
                (self.body_filtered_position[1], 
                    0.0, 
                    self.body_filtered_position[0]),
                tf.transformations.quaternion_from_euler(
                    np.pi/2,
                    -theta,
                    0
                ),
                header.stamp,
                "body_%s" % body_id,
                header.frame_id
            )

        if self.stickman_debug:
            self.tb.sendTransform(
                (-torso[1]+torso_res[0], torso[2], torso[0]+torso_res[2]),
                tf.transformations.quaternion_from_euler(
                    np.pi/2,
                    -theta,
                    0
                ),
                header.stamp,
                "mediapipe_torso_"+self.body_id,
                header.frame_id
            )
            self.tb.sendTransform(
                (0.0, 0.0, 0.605),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "our_torso_"+self.body_id,
                "mediapipe_torso_"+self.body_id
            )
            self.tb.sendTransform(
                (l_shoulder[0], l_shoulder[1], l_shoulder[2]),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "left_shoulder_"+self.body_id,
                "our_torso_"+self.body_id
            )
            self.tb.sendTransform(
                (r_shoulder[0], r_shoulder[1], r_shoulder[2]),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_shoulder_"+self.body_id,
                "our_torso_"+self.body_id
            )
            self.tb.sendTransform(
                (l_elbow[0]-l_shoulder[0],
                 l_elbow[1]-l_shoulder[1],
                 l_elbow[2]-l_shoulder[2]
                 ),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "left_elbow_"+self.body_id,
                "left_shoulder_"+self.body_id
            )
            self.tb.sendTransform(
                (r_elbow[0]-r_shoulder[0],
                 r_elbow[1]-r_shoulder[1],
                 r_elbow[2]-r_shoulder[2]
                 ),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_elbow_"+self.body_id,
                "right_shoulder_"+self.body_id
            )
            self.tb.sendTransform(
                (l_wrist[0]-l_elbow[0],
                 l_wrist[1]-l_elbow[1],
                 l_wrist[2]-l_elbow[2]),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "left_wrist_"+self.body_id,
                "left_elbow_"+self.body_id
            )
            self.tb.sendTransform(
                (r_wrist[0]-r_elbow[0],
                 r_wrist[1]-r_elbow[1],
                 r_wrist[2]-r_elbow[2]
                 ),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_wrist_"+self.body_id,
                "right_elbow_"+self.body_id
            )
            self.tb.sendTransform(
                (l_ankle[0],
                 l_ankle[1],
                 l_ankle[2]
                 ),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0),
                header.stamp,
                "left_ankle_"+self.body_id,
                "mediapipe_torso_"+self.body_id
            )
            self.tb.sendTransform(
                (r_ankle[0],
                 r_ankle[1],
                 r_ankle[2]
                 ),
                tf.transformations.quaternion_from_euler(
                    0,
                    0,
                    0
                ),
                header.stamp,
                "right_ankle_"+self.body_id,
                "mediapipe_torso_"+self.body_id
            )
        js.position = compute_jointstate(
            self.ik_chains[body_id], 
            torso,
            l_wrist,
            l_ankle,
            r_wrist,
            r_ankle
        )

        return js

    def check_bounding_box_consistency(self, bb):
        return bb.x_offset >= 0 \
            and bb.y_offset >= 0 \
            and bb.width > 0 \
            and bb.height > 0 \
            and (bb.x_offset + bb.width < self.img_width) \
            and (bb.y_offset + bb.height < self.img_height)

    def detect(self, image_rgb, header):

        img_height, img_width, _ = image_rgb.shape
        self.img_height, self.img_width = img_height, img_width

        image_rgb.flags.writeable = False
        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_BGR2RGB)
        results = self.detector.process(image_rgb)
        image_rgb.flags.writeable = True
        self.image = image_rgb

        self.x_min_person = img_width
        self.y_min_person = img_height
        self.x_max_person = 0
        self.y_max_person = 0

        ######## Face Detection Process ########

        if hasattr(results.face_landmarks, 'landmark'):
            (self.x_min_face,
             self.y_min_face,
             self.x_max_face,
             self.y_max_face) = _get_bounding_box_limits(
                results.face_landmarks.landmark,
                img_width,
                img_height
            )

            self.x_min_person = int(min(
                self.x_min_person, 
                self.x_min_face))
            self.y_min_person = int(min(
                self.y_min_person, 
                self.y_min_face))
            self.x_max_person = int(max(
                self.x_max_person, 
                self.x_max_face))
            self.y_max_person = int(max(
                self.y_max_person, 
                self.y_max_face))

            if not self.use_depth:
                for idx, landmark in enumerate(results.face_landmarks.landmark):
                    if idx == 1:
                        nose_tip = [landmark.x, landmark.y]
                    if idx == 13:
                        mouth_center = [landmark.x, landmark.y]
                    if idx == 159:
                        right_eye = [landmark.x, landmark.y]
                    if idx == 234:
                        right_ear_tragion = [landmark.x, landmark.y]
                    if idx == 386:
                        left_eye = [landmark.x, landmark.y]
                    if idx == 454:
                        left_ear_tragion = [landmark.x, landmark.y]

                if hasattr(self, "K"):
                    points_2D = np.array([
                        _normalized_to_pixel_coordinates(
                            nose_tip[0],
                            nose_tip[1],
                            self.img_width,
                            self.img_height),
                        _normalized_to_pixel_coordinates(
                            right_eye[0],
                            right_eye[1],
                            self.img_width,
                            self.img_height),
                        _normalized_to_pixel_coordinates(
                            left_eye[0],
                            left_eye[1],
                            self.img_width,
                            self.img_height),
                        _normalized_to_pixel_coordinates(
                            mouth_center[0],
                            mouth_center[1],
                            self.img_width,
                            self.img_height),
                        _normalized_to_pixel_coordinates(
                            right_ear_tragion[0],
                            right_ear_tragion[1],
                            self.img_width,
                            self.img_height),
                        _normalized_to_pixel_coordinates(
                            left_eye[0],
                            left_eye[1],
                            self.img_width,
                            self.img_height)], 
                        dtype="double")

                    self.trans_vec, self.angles = \
                        face_pose_estimation(points_2D, self.K)

                    self.tb.sendTransform(
                        (self.trans_vec[0]/1000,
                         self.trans_vec[1]/1000,
                         self.trans_vec[2]/1000),
                        tf.transformations.quaternion_from_euler(
                            self.angles[0]/180*np.pi,
                            self.angles[1]/180*np.pi,
                            self.angles[2]/180*np.pi),
                        rospy.Time.now(),
                        "face_"+self.body_id,
                        header.frame_id)
                    self.tb.sendTransform(
                        (0, 0, 0),
                        tf.transformations.quaternion_from_euler(
                            -np.pi/2,
                            0,
                            -np.pi/2),
                        rospy.Time.now(),
                        "gaze_"+self.body_id,
                        "face_"+self.body_id)

            # Since I create this structure starting from a face,
            # then I can just publish the body if I need.
            # I won't publish the face since I am
            # already doing it in the main FaceDetector class.
            # What is really important is then body parts and
            # Jointstate message

        ########################################

        ######## Introducing Hand Landmarks ########

        if hasattr(results.left_hand_landmarks, 'landmark'):
            pose_keypoints = protobuf_to_dict(results.pose_landmarks)
            pose_kpt = pose_keypoints.get('landmark')
            landmarks = [None] * 21
            for i in range(0, 21):
                landmarks[i] = PointOfInterest2D(
                    pose_kpt[i].get('x'),
                    pose_kpt[i].get('y'),
                    pose_kpt[i].get('visibility')
                )
            (self.x_min_hand_left,
             self.y_min_hand_left,
             self.x_max_hand_left,
             self.y_max_hand_left) = _get_bounding_box_limits(landmarks,
                                                              img_width,
                                                              img_height)
            self.x_min_person = int(min(
                self.x_min_person, 
                self.x_min_hand_left))
            self.y_min_person = int(min(
                self.y_min_person, 
                self.y_min_hand_left))
            self.x_max_person = int(max(
                self.x_max_person, 
                self.x_max_hand_left))
            self.y_max_person = int(max(
                self.y_max_person, 
                self.y_max_hand_left))

        if hasattr(results.right_hand_landmarks, 'landmark'):
            pose_keypoints = protobuf_to_dict(results.pose_landmarks)
            pose_kpt = pose_keypoints.get('landmark')
            landmarks = [None] * 21
            for i in range(0, 21):
                landmarks[i] = PointOfInterest2D(
                    pose_kpt[i].get('x'),
                    pose_kpt[i].get('y'),
                    pose_kpt[i].get('visibility')
                )
            (self.x_min_hand_right,
             self.y_min_hand_right,
             self.x_max_hand_right,
             self.y_max_hand_right) = _get_bounding_box_limits(landmarks,
                                                               img_width,
                                                               img_height)
            self.x_min_person = int(min(
                self.x_min_person, 
                self.x_min_hand_right))
            self.y_min_person = int(min(
                self.y_min_person, 
                self.y_min_hand_right))
            self.x_max_person = int(max(
                self.x_max_person, 
                self.x_max_hand_right))
            self.y_max_person = int(max(
                self.y_max_person, 
                self.y_max_hand_right))

        ############################################

        ######## Body Detection Process ########

        if hasattr(results.pose_landmarks, 'landmark'):
            pose_keypoints = protobuf_to_dict(results.pose_landmarks)
            pose_world_keypoints = protobuf_to_dict(
                results.pose_world_landmarks)
            pose_kpt = pose_keypoints.get('landmark')
            pose_world_kpt = pose_world_keypoints.get('landmark')
            skel_msg = _make_2d_skeleton_msg(header, pose_kpt)
            if self.trans_vec[0] and not self.use_depth:
                self.body_position_estimation = \
                    self.face_to_body_position_estimation(skel_msg)
            js = self.make_jointstate(
                self.body_id,
                pose_world_kpt,
                pose_kpt,
                header
            )
            self.js_pub.publish(js)
            self.skel_pub.publish(skel_msg)
            if self.single_body:
                landmarks = [None]*32
                for i in range(0, 32):
                    landmarks[i] = PointOfInterest2D(
                    pose_kpt[i].get('x'),
                    pose_kpt[i].get('y'),
                    pose_kpt[i].get('visibility')
                )
                (self.x_min_body,
                 self.y_min_body,
                 self.x_max_body,
                 self.y_max_body) = _get_bounding_box_limits(landmarks,
                                                                img_width,
                                                                img_height)
                self.x_min_person = int(min(
                    self.x_min_person, 
                    self.x_min_body))
                self.y_min_person = int(min(
                    self.y_min_person, 
                    self.y_min_body))
                self.x_max_person = int(max(
                    self.x_max_person, 
                    self.x_max_body))
                self.y_max_person = int(max(
                    self.y_max_person, 
                    self.y_max_body))

        if self.single_body:
            ids_list = IdsList()
            if(self.textual_debug):
                rospy.loginfo("Detected body_%s, ROI = [%s, %s, %s, %s]", \
                    self.body_id, 
                    str(self.x_min_person), 
                    str(self.y_min_person),
                    str(self.x_max_person-self.x_min_person),
                    str(self.y_max_person-self.y_min_person))
            if self.x_min_person < self.x_max_person \
                and self.y_min_person < self.y_max_person:
                self.x_min_person = max(0, self.x_min_person)
                self.y_min_person = max(0, self.y_min_person)
                self.x_max_person = min(img_width, self.x_max_person)
                self.y_max_person = min(img_height, self.y_max_person)
                ids_list.ids = [self.body_id]
                roi = RegionOfInterest()
                roi.x_offset = self.x_min_person
                roi.y_offset = self.y_min_person
                roi.width = self.x_max_person - self.x_min_person
                roi.height = self.y_max_person - self.y_min_person
                self.roi_pub.publish(roi)
            self.ids_pub.publish(ids_list)

        ########################################

    def image_callback_depth(self, 
                rgb_img, 
                rgb_info, 
                roi,
                depth_img, 
                depth_info):

        rgb_img = self.br.imgmsg_to_cv2(rgb_img)
        image_depth = self.br.imgmsg_to_cv2(depth_img, "16UC1")
        self.image_depth = image_depth
        if depth_info.header.stamp > rgb_info.header.stamp:
            header = copy.copy(depth_info.header)
            header.frame_id = rgb_info.header.frame_id # to check 
        else:
            header = copy.copy(rgb_info.header)
        self.depth_info = depth_info
        self.rgb_info = rgb_info
        self.x_offset = roi.x_offset
        self.y_offset = roi.y_offset
        self.roi = roi
        self.detect(rgb_img, header)

    def image_callback_depth_single_person(self, 
                rgb_img, 
                rgb_info,
                depth_img, 
                depth_info):

        rgb_img = self.br.imgmsg_to_cv2(rgb_img)
        image_depth = self.br.imgmsg_to_cv2(depth_img, "16UC1")
        self.image_depth = image_depth
        if depth_info.header.stamp > rgb_info.header.stamp:
            header = copy.copy(depth_info.header)
            header.frame_id = rgb_info.header.frame_id # to check 
        else:
            header = copy.copy(rgb_info.header)
        self.depth_info = depth_info
        self.rgb_info = rgb_info
        self.x_offset = 0
        self.y_offset = 0
        self.roi = RegionOfInterest()
        self.roi.x_offset = 0
        self.roi.y_offset = 0
        self.detect(rgb_img, header)

    def image_callback_rgb(self, rgb_img, rgb_info):

        if not rospy.has_param(self.human_description):
            rospy.logerr('URDF file NOT found', self.human_description)
            rosparam.set_param_raw(self.human_description, self.urdf)
            return
        
        rgb_img = self.br.imgmsg_to_cv2(rgb_img)        

        header = copy.copy(rgb_info.header)
        self.rgb_info = rgb_info
        self.detect(rgb_img, header)

    def get_image_topic(self):
        return self.image_subscriber.sub.resolved_name
