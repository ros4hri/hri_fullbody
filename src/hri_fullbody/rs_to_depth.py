import numpy as np
from image_geometry import PinholeCameraModel


def rgb_to_xyz(
        x_rgb,
        y_rgb,
        rgb_camera_info,
        depth_camera_info,
        depth_data_encoding,
        depth_data,
        roi_xmin = 0.,
        roi_ymin = 0.):
    depth_model = PinholeCameraModel()
    rgb_model = PinholeCameraModel()

    depth_model.fromCameraInfo(depth_camera_info)
    rgb_model.fromCameraInfo(rgb_camera_info)

    x_rgb = x_rgb + (roi_xmin * rgb_model.width)
    y_rgb = y_rgb + (roi_ymin * rgb_model.height)

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
    
    if depth_data_encoding == '32FC1':
        # Get depth data encoded as 32bit/m
        z = depth_data[y_d][x_d]
    elif depth_data_encoding == '16UC1':
        # Convert depth data encoded as 16bit/mm to m
        z = depth_data[y_d][x_d]/1000
    else:
        raise ValueError('Unexpected encoding {}. '.format(depth_data_encoding) +\
                         'Depth encoding should be 16UC1 or `32FC1`.')
    
    x = (x_d - depth_model.cx())*z/depth_model.fx()
    y = (y_d - depth_model.cy())*z/depth_model.fy()

    return np.array([x, y, z])


