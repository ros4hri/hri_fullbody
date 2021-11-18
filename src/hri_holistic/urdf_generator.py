import xacro
import rospkg
from pathlib import Path

TPL = Path(rospkg.RosPack().get_path("hri_holistic")) / "urdf/human-tpl.xacro"

def make_urdf_human(body_id,
               head_radius = None,
               neck_shoulder_length = None,
               upperarm_length = None, 
               forearm_length = None,
               torso_height = None,
               waist_length = None,
               tight_length = None,
               tibia_length = None
               ):
    params = {'id': body_id}

    if head_radius:
        params['head_radius'] = str(head_radius)

    if neck_shoulder_length:
        params['neck_shoulder_length'] = str(neck_shoulder_length)

    if upperarm_length:
        params['upperarm_length'] = str(upperarm_length)

    if forearm_length:
        params['forearm_length'] = str(forearm_length)

    if torso_height:
        params['torso_height'] = str(torso_height)

    if waist_length:
        params['waist_length'] = str(waist_length)

    if tight_length:
        params['tight_length'] = str(tight_length)

    if tibia_length:
        params['tibia_length'] = str(tibia_length)

    return xacro.process_file(TPL, mappings=params).toxml()
