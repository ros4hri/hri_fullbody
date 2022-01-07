import numpy as np
from image_geometry import PinholeCameraModel


def rgb_to_xyz(
        x_rgb,
        y_rgb,
        rgb_camera_info,
        depth_camera_info,
        depth_data,
        x_offset = 0,
        y_offset = 0):
    depth_model = PinholeCameraModel()
    rgb_model = PinholeCameraModel()

    depth_model.fromCameraInfo(depth_camera_info)
    rgb_model.fromCameraInfo(rgb_camera_info)

    x_rgb = x_rgb + x_offset
    y_rgb = y_rgb + y_offset

    if x_rgb > rgb_model.width:
        x_rgb = rgb_model.width - 1
    if y_rgb > rgb_model.height:
        y_rgb = rgb_model.height - 1

    x_d = int(((x_rgb - rgb_model.cx())
               * depth_model.fx()
               / rgb_model.fx())
              + depth_model.cx())
    y_d = int(((y_rgb - rgb_model.cy())
               * depth_model.fy()
               / rgb_model.fy())
              + depth_model.cy())
    z = depth_data[y_d][x_d]/1000
    x = (x_d - depth_model.cx())*z/depth_model.fx()
    y = (y_d - depth_model.cy())*z/depth_model.fy()

    return np.array([x, y, z])


