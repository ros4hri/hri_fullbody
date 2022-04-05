import cv2
import numpy as np

# face key points
P3D_RIGHT_EYE = (-20., -65.5, -5.)
P3D_LEFT_EYE = (-20., 65.5, -5.)
P3D_RIGHT_EAR = (-100., -77.5, -6.)
P3D_LEFT_EAR = (-100., 77.5, -6.)
P3D_NOSE = (21.0, 0., -48.0)
P3D_STOMION = (10.0, 0., -75.0)


points_3D = np.array([
    P3D_NOSE,
    P3D_RIGHT_EYE,
    P3D_LEFT_EYE,
    P3D_STOMION,
    P3D_RIGHT_EAR,
    P3D_LEFT_EAR]
)

def face_pose_estimation(points_2D, K):
    success, rot_vec, trans_vec = cv2.solvePnP(
                                        points_3D,
                                        points_2D,
                                        K,
                                        None,
                                        tvec=np.array([0., 0., 1000.]),
                                        useExtrinsicGuess=True,
                                        flags=4)
    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    return trans_vec, angles