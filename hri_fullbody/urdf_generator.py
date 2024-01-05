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

import xacro
from ament_index_python.packages import get_package_share_directory
from pathlib import Path

TPL = Path(get_package_share_directory(
    "human_description")) / "urdf/human-tpl.xacro"


def make_urdf_human(
    body_id,
    head_radius=None,
    neck_shoulder_length=None,
    upperarm_length=None,
    forearm_length=None,
    torso_height=None,
    waist_length=None,
    tight_length=None,
    tibia_length=None,
):
    params = {"id": body_id}

    if head_radius:
        params["head_radius"] = str(head_radius)

    if neck_shoulder_length:
        params["neck_shoulder_length"] = str(neck_shoulder_length)

    if upperarm_length:
        params["upperarm_length"] = str(upperarm_length)

    if forearm_length:
        params["forearm_length"] = str(forearm_length)

    if torso_height:
        params["torso_height"] = str(torso_height)

    if waist_length:
        params["waist_length"] = str(waist_length)

    if tight_length:
        params["tight_length"] = str(tight_length)

    if tibia_length:
        params["tibia_length"] = str(tibia_length)

    return xacro.process_file(TPL, mappings=params).toxml()
