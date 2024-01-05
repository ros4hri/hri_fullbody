# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    rgb_camera_arg = DeclareLaunchArgument(
        'rgb_camera',
        default_value='',
        description='The input rgb camera namespace')
    rgb_camera_topic_arg = DeclareLaunchArgument(
        'rgb_camera_topic',
        default_value=[LaunchConfiguration('rgb_camera'),
                       '/image_raw'],
        description='The input rgb camera image topic')
    rgb_camera_info_arg = DeclareLaunchArgument(
        'rgb_camera_info',
        default_value=[LaunchConfiguration('rgb_camera'),
                       '/camera_info'],
        description='The input rgb camera info topic')
    use_depth_arg = DeclareLaunchArgument(
        'use_depth',
        default_value=['False'],
        description='Whether or not using the depth information')
    depth_camera_arg = DeclareLaunchArgument(
        'depth_camera',
        default_value='',
        description='The input depth camera namespace')
    depth_camera_topic_arg = DeclareLaunchArgument(
        'depth_camera_topic',
        default_value=[LaunchConfiguration('depth_camera'),
                       '/image_rect_raw'],
        description='The input depth camera image topic')
    depth_camera_info_arg = DeclareLaunchArgument(
        'depth_camera_info',
        default_value=[LaunchConfiguration('depth_camera'),
                       '/camera_info'],
        description='The input depth camera info topic')
    stickman_debug_arg = DeclareLaunchArgument(
        'stickman_debug',
        default_value=['False'],
        description='When true, publishes the trasforms of some\
                     3D skeleton keypoints as they are output\
                     from the holistic pose estimation process')
    single_body_arg = DeclareLaunchArgument(
        'single_body',
        default_value=['True'],
        description='When false, registers to the list of\
                     detected bodies and performs\
                     pose estimation for each one of them')

    fullbody_detect_node = Node(
        package='hri_fullbody',
        executable='node_pose_detect',
        namespace='',
        name='fullbody_detect',
        parameters=[{'use_depth': LaunchConfiguration('use_depth'),
                     'single_body': LaunchConfiguration('single_body'),
                     'stickman_debug': LaunchConfiguration('stickman_debug')}],
        remappings=[
            ('image', LaunchConfiguration('rgb_camera_topic')),
            ('camera_info', LaunchConfiguration('rgb_camera_info')),
            ('depth_image', LaunchConfiguration('depth_camera_topic')),
            ('depth_info', LaunchConfiguration('depth_camera_info'))])

    return LaunchDescription([
        rgb_camera_arg,
        rgb_camera_topic_arg,
        rgb_camera_info_arg,
        depth_camera_arg,
        depth_camera_topic_arg,
        depth_camera_info_arg,
        use_depth_arg,
        single_body_arg,
        stickman_debug_arg,
        fullbody_detect_node])
