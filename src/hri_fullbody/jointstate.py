
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

    r_arm_target = r_wrist - torso # TODO: INCORRECT! need to *transform* r_wrist in the torso's reference frame, eg account for the torso's rotation!!
    r_arm_joints = r_arm.inverse_kinematics(r_arm_target)

    l_arm_target = l_wrist - torso
    l_arm_joints = l_arm.inverse_kinematics(l_arm_target)

    r_leg_target = r_ankle - torso
    r_leg_joints = r_leg.inverse_kinematics(r_leg_target)

    l_leg_target = l_ankle - torso
    l_leg_joints = l_leg.inverse_kinematics(l_leg_target)


    return [0.0, 0.0, 0.0, 0.0] + list(l_arm_joints)[1:-1] + list(r_arm_joints)[1:-1] + list(l_leg_joints)[1:-1] + list(r_leg_joints)[1:-1]


