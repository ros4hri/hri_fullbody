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

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException

from hri_fullbody.fullbody_detector import FullbodyDetector

from hri_msgs.msg import IdsList
from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from rcl_interfaces.msg import ParameterDescriptor

from collections import OrderedDict
import random

# body detection processing time in ms triggering a diagnostic warning
BODY_DETECTION_PROC_TIME_WARN = 1000.


def generate_id():
    """Return a 5 chars ID."""
    return "".join(random.sample("abcdefghijklmnopqrstuvwxyz", 5))


class MultibodyManager(Node):
    """
    Manager for the FullbodyDetector objects assigned to each detected body.

    When started in single body mode, generates an ID
    for the to-be-dected body and starts a single FullbodyDetector.
    """

    def __init__(self):
        super().__init__("hri_fullbody")
        self.declare_parameter(
            "use_depth",
            False,
            ParameterDescriptor(
                description="Whether or not using depth to process the bodies position.")
        )
        self.declare_parameter(
            "stickman_debug",
            False,
            ParameterDescriptor(
                description="Whether or not using the stickman debugging visualization."))
        self.declare_parameter(
            "single_body",
            True,
            ParameterDescriptor(
                description="Whether performing single or multiple bodies detection"))
        self.declare_parameter(
            "diagnostic_period",
            1.,
            ParameterDescriptor(
                description="Diagnostic period"))

        self.single_body = self.get_parameter("single_body").value
        self.use_depth = self.get_parameter("use_depth").value
        self.stickman_debug = self.get_parameter("stickman_debug").value
        diag_period = self.get_parameter("diagnostic_period").value

        self.detected_bodies = OrderedDict()

        if self.single_body:
            self.get_logger().warning(
                "hri_fullbody running in single body mode:"
                + " only one skeleton will be detected.")
            id = generate_id()
            self.detected_bodies[id] = FullbodyDetector(self,
                                                        self.use_depth,
                                                        self.stickman_debug,
                                                        id,
                                                        self.single_body)
            self.get_logger().info("Generated single person detector for body_%s"
                                   % id)
            self.get_logger().info("Waiting for frames on topic %s"
                                   % self.detected_bodies[id].get_image_topic())
        else:
            self.get_logger().info("Waiting for ids on /humans/bodies/tracked")
            self.bodies_list_sub = self.create_subscription(IdsList,
                                                            "/humans/bodies/tracked",
                                                            self.ids_list_cb,
                                                            1)

        self.diag_timer = self.create_timer(diag_period, self.do_diagnostics)
        self.diag_pub = self.create_publisher(DiagnosticArray,
                                              "/diagnostics",
                                              1)

    def ids_list_cb(self, msg: IdsList):
        """Subscribe to the list of detected bodies."""
        for id in msg.ids:
            if id not in self.detected_bodies:
                self.detected_bodies[id] = FullbodyDetector(self,
                                                            self.use_depth,
                                                            self.stickman_debug,
                                                            id)
                self.get_logger().info(
                    "Generated single person detector for body_%s"
                    % id)
                self.get_logger().info(
                    "Waiting for frames on topic %s"
                    % self.detected_bodies[id].get_image_topic(),
                )

        for id in self.detected_bodies:
            if id not in msg.ids:
                self.detected_bodies[id].unregister()
                self.detected_bodies.pop(id)

    def do_diagnostics(self):
        """Perform diagnostic operations."""
        arr = DiagnosticArray()
        arr.header.stamp = self.get_clock().now().to_msg()

        proc_time = sum(v.get_proc_time().nanoseconds/1e9
                        for v in self.detected_bodies.values())

        msg = DiagnosticStatus(name="Social perception: Body analysis: Skeleton extraction",
                               hardware_id="none")

        if any(v.check_timeout() for v in self.detected_bodies.values()):
            msg.level = DiagnosticStatus.ERROR
            msg.message = "Body detection process not responding"
        elif proc_time > BODY_DETECTION_PROC_TIME_WARN:
            msg.level = DiagnosticStatus.WARN
            msg.message = "Body detection processing is slow"
        else:
            msg.level = DiagnosticStatus.OK

        msg.values = [
            KeyValue(key="Package name", value='hri_fullbody'),
            KeyValue(key="Single body detector mode",
                     value=str(self.single_body)),
            KeyValue(key="Currently detected bodies",
                     value=str(sum([v.is_body_detected for v in self.detected_bodies.values()]))),
            KeyValue(key="Last detected body ID", value=str(
                next(reversed(self.detected_bodies), ''))),
            KeyValue(key="Detection processing time",
                     value="{:.2f}".format(proc_time * 1000) + "ms"),
        ]

        arr.status = [msg]
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)

    node = MultibodyManager()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == "__main__":
    main()
