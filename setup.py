from setuptools import find_packages, setup

package_name = 'hri_fullbody'

setup(
    name=package_name,
    version='2.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hri_fullbody.launch.py'])
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lorenzo Ferrini',
    maintainer_email='lorenzo.ferrini@pal-robotics.com',
    description='ROS node implementing 2D/3D full-body pose estimation, using Google Mediapipe.\
                 Part of ROS4HRI.',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'node_pose_detect = hri_fullbody.node_pose_detect:main'
        ],
    },
)
