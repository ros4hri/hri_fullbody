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

HUMAN_JOINT_NAMES = ["waist",
                     "r_head",
                     "y_head",
                     "p_head",
                     "l_y_shoulder",
                     "l_p_shoulder",
                     "l_r_shoulder",
                     "l_elbow",
                     "r_y_shoulder",
                     "r_p_shoulder",
                     "r_r_shoulder",
                     "r_elbow",
                     "l_y_hip",
                     "l_p_hip",
                     "l_r_hip",
                     "l_knee",
                     "r_y_hip",
                     "r_p_hip",
                     "r_r_hip",
                     "r_knee"]


def compute_jointstate(ik_chains, torso, l_wrist, l_ankle, r_wrist, r_ankle):

    r_arm, l_arm, r_leg, l_leg = ik_chains

    # k_args = {"optimizer": "scalar"}

    # TODO: INCORRECT! need to *transform* r_wrist in the torso's reference frame,
    # eg account for the torso's rotation!!
    r_arm_target = r_wrist - torso
    r_arm_joints = r_arm.inverse_kinematics(r_arm_target, optimizer="scalar")

    l_arm_target = l_wrist - torso
    l_arm_joints = l_arm.inverse_kinematics(l_arm_target, optimizer="scalar")

    r_leg_target = r_ankle - torso
    r_leg_joints = r_leg.inverse_kinematics(r_leg_target, optimizer="scalar")

    l_leg_target = l_ankle - torso
    l_leg_joints = l_leg.inverse_kinematics(l_leg_target, optimizer="scalar")

    return [0.0, 0.0, 0.0, 0.0]\
        + list(l_arm_joints)[1:-1]\
        + list(r_arm_joints)[1:-1]\
        + list(l_leg_joints)[1:-1]\
        + list(r_leg_joints)[1:-1]
