#!/usr/bin/env python3
"""
Boilerplate controller for the HE bot.

Fetches a shape from the get_shape service and builds a list of waypoints
to trace it. Fill in the control loop to drive the robot through them.


"""

import argparse
import math

import numpy as np
import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from shape_interface.srv import GetShape


WHEEL_RADIUS_M = 0.0255
CHASSIS_RADIUS_M = 0.06412
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0])

_BODY_TO_WHEEL_LIN = np.column_stack([
    np.cos(WHEEL_ANGLES_RAD),
    np.sin(WHEEL_ANGLES_RAD),
    np.full(3, CHASSIS_RADIUS_M),
])

# Wheel <-> body-velocity mapping, columns are [left, right, back] wheel
# speed (rad/s); rows are body frame [vx, vy, wz] per unit wheel speed.
_WHEEL_TO_BODY = WHEEL_RADIUS_M * np.linalg.inv(_BODY_TO_WHEEL_LIN)
_BODY_TO_WHEEL = np.linalg.inv(_WHEEL_TO_BODY)
_CTRL_LIMIT = 3.14     # rad/s, matches lekiwi.xml actuator ctrlrange

WAYPOINT_TOLERANCE  = 0.03 # metres
CIRCLE_SEGMENTS     = 36
POSITION_KP         = 1.5
YAW_HOLD_KP         = 2.0
CONTROL_PERIOD      = 0.02 # 50Hz

def body_to_wheels(vx, vy, wz):
    """Body-frame (vx, vy, wz) -> wheel angular velocities [left, right, back]."""
    # convert body velocity to wheel speeds using _BODY_TO_WHEEL,
    # clip each wheel to [-_CTRL_LIMIT, _CTRL_LIMIT], return as a list.
    w = _BODY_TO_WHEEL @ np.array([vx, vy, wz])
    w = np.clip(w, -_CTRL_LIMIT, _CTRL_LIMIT)
    return [float(x) for x in w]


def yaw_from_quat(w, x, y, z):
    # convert quaternion to yaw (radians).
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(a):
    return math.atan2(math.sin(a), math.cos(a))


def _regular_polygon(cx, cy, n_sides, side_length, start_angle=math.pi / 2):
    """Vertices of a regular polygon centred on (cx, cy), closed back to the
    first vertex so the last waypoint returns the robot to where it started
    drawing."""
    r = side_length / (2 * math.sin(math.pi / n_sides))
    pts = [
        (cx + r * math.cos(start_angle + 2 * math.pi * i / n_sides),
         cy + r * math.sin(start_angle + 2 * math.pi * i / n_sides))
        for i in range(n_sides)
    ]
    return pts + [pts[0]]


def build_waypoints(shape_name, data):
    """World-frame waypoints for `shape_name`, as returned by the get_shape
    service: data[0:2] is the shape's centre (x, y); the remaining entries
    are its size parameters (see shape_service.cpp's shape_map)."""
    cx, cy = data[0], data[1]

    if shape_name == "Circle":
        radius = data[2]
        return [
            (cx + radius * math.cos(2 * math.pi * i / CIRCLE_SEGMENTS),
             cy + radius * math.sin(2 * math.pi * i / CIRCLE_SEGMENTS))
            for i in range(1, CIRCLE_SEGMENTS + 1)
        ]

    if shape_name == "Square":
        return _regular_polygon(cx, cy, 4, data[2], start_angle=math.pi / 4)

    if shape_name == "Triangle":
        return _regular_polygon(cx, cy, 3, data[2])

    if shape_name == "Pentagon":
        return _regular_polygon(cx, cy, 5, data[2])

    if shape_name == "Rectangle":
        w, h = data[2], data[3]
        corners = [
            (cx - w / 2, cy - h / 2),
            (cx + w / 2, cy - h / 2),
            (cx + w / 2, cy + h / 2),
            (cx - w / 2, cy + h / 2),
        ]
        return corners + [corners[0]]

    raise ValueError(f"unknown shape '{shape_name}'")


class ShapeController(Node):
    def __init__(self, speed):
        super().__init__("shape_controller")
        self.speed = speed

        self.pose = None        # (x, y, yaw), latest ground truth
        self.start_pose = None  # (x, y, yaw), recorded on first odom message
        self.wp_index = 0
        self.done = False

        self.cmd_pub = self.create_publisher(Float64MultiArray, "/wheel_commands", 10) # publisher for wheel commands
        self.odom_sub = self.create_subscription(Odometry, "/odom", self._odom_cb, 10) # subscriber for odometry messages

        self.shape_name, self.waypoints = self._request_shape()
        self.control_timer = self.create_timer(CONTROL_PERIOD, self._control_step) # timer for periodic control steps
        self.get_logger().info(f"Shape: {self.shape_name}, waypoints: {self.waypoints}") 

    def _request_shape(self):
        client = self.create_client(GetShape, "get_shape")
        while not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("Waiting for get_shape service...")

        future = client.call_async(GetShape.Request())
        rclpy.spin_until_future_complete(self, future)
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(
                f"get_shape service call failed: {response and response.message}"
            )

        return response.shape_name, build_waypoints(response.shape_name, list(response.data))

    def _odom_cb(self, msg):
        # extract (x, y, yaw) from msg.pose.pose into self.pose,
        # and record self.start_pose on the first callback.
        p = msg.pose.pose.position       # current position
        o = msg.pose.pose.orientation    # current orientation
        yaw = yaw_from_quat(o.w, o.x, o.y, o.z)

        self.pose = (p.x, p.y, yaw)

        if self.start_pose is None:
            self.start_pose = self.pose

    def _publish(self, wheels):
        self.cmd_pub.publish(Float64MultiArray(data=wheels))

    def _control_step(self):
        if self.done or self.pose is None:
            return

        if not self.waypoints:
            self.done = True
            self._publish([0.0, 0.0, 0.0])
            self.get_logger().error("No waypoints were provided; stopping.")
            return

        x, y, yaw = self.pose

        # Skip any waypoints already inside the arrival tolerance.
        while self.wp_index < len(self.waypoints):
            tx, ty = self.waypoints[self.wp_index]
            dx = tx - x
            dy = ty - y
            distance = math.hypot(dx, dy)
            if distance > WAYPOINT_TOLERANCE:
                break
            self.wp_index += 1

        if self.wp_index >= len(self.waypoints):
            self.done = True
            self._publish([0.0, 0.0, 0.0])
            self.get_logger().info("Shape complete")
            return

        # Use proportional position control, limited to the requested speed.
        target_yaw = math.atan2(dy, dx)
        yaw_error = math.atan2(
            math.sin(target_yaw - yaw),
            math.cos(target_yaw - yaw),
        )

        vx_world = POSITION_KP * dx
        vy_world = POSITION_KP * dy
        speed = math.hypot(vx_world, vy_world)
        if speed > self.speed:
            scale = self.speed / speed
            vx_world *= scale
            vy_world *= scale

        # Convert the world-frame motion command into the robot's body frame.
        vx = math.cos(yaw) * vx_world + math.sin(yaw) * vy_world
        vy = -math.sin(yaw) * vx_world + math.cos(yaw) * vy_world
        wz = YAW_HOLD_KP * yaw_error

        self._publish(body_to_wheels(vx, vy, wz))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--speed", type=float, default=0.06,
                         help="max approach speed, m/s")
    args, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args)
    node = ShapeController(args.speed)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish([0.0, 0.0, 0.0])
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
