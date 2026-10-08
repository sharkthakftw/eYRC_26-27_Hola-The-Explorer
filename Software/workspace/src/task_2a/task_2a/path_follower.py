#!/usr/bin/env python3
# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

'''
*****************************************************************************************
*
*        =============================================
*           Hola The Explorer (HE) Theme (eYRC 2026-27)
*        =============================================
*
*  This script is to implement Task 2A of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ 3142 ]
# Author List:      [ Shourya Gupta, Sarthak Gupta]
# Filename:         path_follower.py
# Functions:        [ Add every extra helper function you write to this list ]
# Global variables: [ V_MAX, W_MAX, KP_POS, KI_POS, KD_POS, KP_YAW, KI_YAW, KD_YAW ]


############################ WHAT YOU HAVE TO DO ##############################
#
#  Drive r1, r2 and r3 AT THE SAME TIME, each in a straight line to its goal
#  in GOALS, holding its starting heading, and park it there.
#
#   1. Paste in your Task 1B / 1C code   (body_velocity_to_wheel_speeds, ...)
#   2. Write wrap(), to_body() and the GoToPoint PID controller
#   3. Complete RobotNode                (main() runs one per robot)
#
#  Frame (/rN/odom): origin at the arena's TOP-LEFT corner, x right, y DOWN,
#  metres. Yaw 0 along +x, growing clockwise; the robots start at -pi/2.
#
#  Run:  ros2 launch hb_description task2a.launch.py
#        ros2 run task_2a path_follower
#
###############################################################################

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray

################# ADD EXTRA IMPORTS / GLOBALS HERE ############
V_MAX = 0.6
W_MAX = 2.0
KP_POS, KI_POS, KD_POS = 2.5, 0.0, 0.15
KP_YAW, KI_YAW, KD_YAW = 3.0, 0.0, 0.1
###############################################################


# ----------------------------------------------------- arena (given, do not edit)
ARENA_SIZE = 2.4384           # m, the floor is square
ROBOT_RADIUS = 0.084          # m

START = {"r1": (0.8969, 2.2616),
         "r2": (1.2192, 2.2616),
         "r3": (1.5415, 2.2616)}
GOALS = {"r1": (0.6692, 0.8692),
         "r2": (1.2192, 0.6692),
         "r3": (1.7692, 0.8692)}
ORDER = ("r1", "r2", "r3")    # one node per robot

# ----------------------------------------------------- yours to set
CONTROL_HZ = 20.0              # how often tick() runs
GOAL_TOLERANCE = 0.02         # m, how close counts as "at the goal"
SETTLE_TICKS = 10            # ticks in a row inside GOAL_TOLERANCE = arrived

# ----------------------------------------------------- robot (same as Task 1B)
WHEEL_RADIUS_M = 0.0255       # m
CHASSIS_RADIUS_M = 0.06412    # m, chassis centre to each wheel's axle
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0]) # from Task 1B, [left, right, back]
IK_MATRIX = np.zeros((3, 3))                     # from Task 1B
_CTRL_LIMIT = 30.0            # rad/s, the wheels' ctrlrange in the robot's MJCF:
                              # faster commands are clamped by the simulation


##################### FROM TASK 1B / 1C #######################

def body_velocity_to_wheel_speeds(vx, vy, w):
    """Body twist (vx, vy, w) -> wheel speeds [left, right, back], rad/s."""
    # Task 1B
    angles = WHEEL_ANGLES_RAD
    linear_speed = vx * np.cos(angles) + vy * np.sin(angles)
    rotational_speed = w * CHASSIS_RADIUS_M
    return (linear_speed + rotational_speed) / WHEEL_RADIUS_M


def body_to_wheels(vx, vy, wz):
    """Like body_velocity_to_wheel_speeds(), but kept inside +/-_CTRL_LIMIT.
    Returns a list."""
    raw_wheel_speeds = body_velocity_to_wheel_speeds(vx, vy, wz)    # rad/s
    max_w = max(abs(w) for w in raw_wheel_speeds)                   
    if max_w > _CTRL_LIMIT:
        scaled_wheel_speeds = [w * (_CTRL_LIMIT / max_w) for w in raw_wheel_speeds]  # scale down to the limit
        return scaled_wheel_speeds
    return raw_wheel_speeds

def yaw_from_quat(w, x, y, z):
    """Quaternion -> yaw, radians."""
    # Task 1C
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


######################### NEW IN 2A ###########################

def wrap(a):
    """Angle -> the same angle in (-pi, pi]. 350 degrees becomes -10."""
    return math.atan2(math.sin(a), math.cos(a))


def to_body(vx_a, vy_a, yaw):
    """Arena-frame velocity -> body-frame velocity (vx, vy).
    Check: at yaw = -pi/2 (facing up), arena (0, -v) must give body (v, 0)."""
    vx = (vx_a * math.cos(yaw)) + (vy_a * math.sin(yaw))                       
    vy = (-vx_a * math.sin(yaw)) + (vy_a * math.cos(yaw))

    return (vx, vy)


def frame_check(x, y, size=ARENA_SIZE):
    """Given. A complaint if (x, y) is not in the /odom frame, else None."""
    if not (0.0 <= x <= size and 0.0 <= y <= size):
        return ("odom pose (%.3f, %.3f) is outside 0..%.4f: this code works "
                "in the top-left-origin arena frame, not MuJoCo's "
                "centre-origin one" % (x, y, size))
    return None


class GoToPoint:
    """PID controller: drive one robot straight at a point, holding a fixed yaw.
    One instance per robot -- the PID terms keep state between steps."""

    def __init__(self, v_max=0.0, w_max=0.0, kp=0.0, ki=0.0, kd=0.0,
                 kyaw=0.0, kiyaw=0.0, kdyaw=0.0, tol=0.0, dt=0.0):
        self.v_max, self.w_max = v_max, w_max            # m/s, rad/s caps
        self.kp, self.ki, self.kd = kp, ki, kd           # position PID
        self.kyaw, self.kiyaw, self.kdyaw = kyaw, kiyaw, kdyaw   # yaw PID
        self.tol = tol                                   # m, stop inside this
        self.dt = dt                                     # s, 1 / CONTROL_HZ

        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
        self.prev_error_yaw = 0.0
        self.integral_x = 0.0
        self.integral_y = 0.0
        self.integral_yaw = 0.0
        self.integral_limit = 1.0


    def step(self, pose, target, hold_yaw):
        """pose (x, y, yaw), target (x, y), hold_yaw -> (wheels, distance)."""
        #   1. error and distance from pose to target, in the arena frame
        #   2. arena velocity from a PID on that error: 0 inside tol, size
        #      capped at v_max, integral limited so it cannot wind up
        #   3. wz from a PID on wrap(hold_yaw - yaw), capped at w_max
        #   4. to_body(), then body_to_wheels()
        target_x, target_y = target
        pose_x, pose_y, pose_yaw = pose

        # compute the error in x and y directions
        error_x = target_x - pose_x
        error_y = target_y - pose_y
        dist = math.sqrt(error_x * error_x + error_y * error_y)

        # compute the PID control for x and y directions
        if dist <= self.tol:
            vx_a = 0
            vy_a = 0
            self.integral_y = 0.0
            self.integral_x = 0.0

        else:
            self.integral_x += error_x * self.dt
            self.integral_y += error_y * self.dt
            self.integral_x = max(-self.integral_limit, min(self.integral_limit, self.integral_x))
            self.integral_y = max(-self.integral_limit, min(self.integral_limit, self.integral_y))

            deriv_x = (error_x - self.prev_error_x) / self.dt
            deriv_y = (error_y - self.prev_error_y) / self.dt

            Px = self.kp * error_x
            Ix = self.ki * self.integral_x
            Dx = self.kd * deriv_x

            Py = self.kp * error_y
            Iy = self.ki * self.integral_y
            Dy = self.kd * deriv_y

            vx_a = Px + Ix + Dx
            vy_a = Py + Iy + Dy

            # scale the velocity to ensure it does not exceed v_max
            speed = math.hypot(vx_a, vy_a)
            if speed > self.v_max:
                scale = self.v_max / speed
                vx_a *= scale
                vy_a *= scale

        # update previous errors for the next iteration
        self.prev_error_x = error_x
        self.prev_error_y = error_y

        # compute the yaw error and PID control for yaw
        error_yaw = wrap(hold_yaw - pose_yaw)

        self.integral_yaw += error_yaw * self.dt
        self.integral_yaw = max(-self.integral_limit, min(self.integral_limit, self.integral_yaw))

        deriv_yaw = (error_yaw - self.prev_error_yaw) / self.dt
        self.prev_error_yaw = error_yaw

        Pwz = self.kyaw * error_yaw
        Iwz = self.kiyaw * self.integral_yaw
        Dwz = self.kdyaw * deriv_yaw

        wz = Pwz + Iwz + Dwz
        wz = max(-self.w_max, min(self.w_max, wz))       # cap the angular velocity to w_max

        # convert arena-frame velocity to body-frame velocity and then to wheel speeds
        vx_b, vy_b = to_body(vx_a, vy_a, pose_yaw)
        wheels = body_to_wheels(vx_b, vy_b, wz)

        return (wheels, dist) 


######################### ONE NODE PER ROBOT ##################

class RobotNode(Node):
    """Drives one robot to GOALS[robot]. Nodes: task_2a_r1, _r2, _r3.
    No robot moves until all three are in `ready`, so they start together."""

    def __init__(self, robot, ready, on_arrival):
        super().__init__(f"task_2a_{robot}")
        self.robot = robot
        self.ready = ready            # shared by the three nodes
        self.on_arrival = on_arrival  # call once: on_arrival(robot, seconds)

        self.pose = None              # (x, y, yaw) from /<robot>/odom
        self.hold_yaw = None          # yaw on the first odom message
        self.settled = 0
        self.arrived = False
        self.t_start = None

        self.odom_sub = self.create_subscription(Odometry, f"/{self.robot}/odom", self.odom_cb, 10) # subscribe to odometry topic
        self.pub = self.create_publisher(Float64MultiArray, f"/{self.robot}/wheel_commands", 10) # create publisher for wheel commands

        # initialize the GoToPoint controller with the specified parameters
        self.ctl = GoToPoint(
            v_max=V_MAX,
            w_max=W_MAX,
            kp=KP_POS,
            ki=KI_POS,
            kd=KD_POS,
            kyaw=KP_YAW,
            kiyaw=KI_YAW,
            kdyaw=KD_YAW,
            tol=GOAL_TOLERANCE,
            dt=1.0 / CONTROL_HZ
            )
        self.callback_timer = self.create_timer(1 / CONTROL_HZ, self.tick)   # create a timer to call the tick function at CONTROL_HZ frequency


    def odom_cb(self, msg):
        # self.pose = (x, y, yaw). First message only: set
        # self.hold_yaw, log frame_check(x, y) if it complains, and add
        # self.robot to self.ready.
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        yaw = yaw_from_quat(
            msg.pose.pose.orientation.w,
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z
            )

        self.pose = (x, y, yaw)

        if self.hold_yaw is None:
            self.hold_yaw = yaw
            error = frame_check(x, y)
            if error:
                self.get_logger().warning(error)
            self.ready.add(self.robot)

    def send(self, wheels):
        self.pub.publish(Float64MultiArray(data=[float(w) for w in wheels]))

    def tick(self):
        if len(self.ready) < len(ORDER):
            return                                   # wait for every robot's odom
        if self.t_start is None:
            self.t_start = self.get_clock().now()
        #   - arrived: send zeros and return
        #   - else: wheels from self.ctl.step(self.pose, GOALS[self.robot],
        #     self.hold_yaw), and send them
        #   - after SETTLE_TICKS in a row within GOAL_TOLERANCE: set
        #     self.arrived, log "<robot> reached its goal (... mm out)", and
        #     call self.on_arrival(self.robot, seconds since self.t_start)
        if self.arrived:
            self.send([0.0, 0.0, 0.0])   # send zeros to stop the robot
            return

        wheel, dist = self.ctl.step(self.pose, GOALS[self.robot], self.hold_yaw)  # get wheel speeds and distance to goal

        # check if the robot is within the goal tolerance and update settled count
        if dist < GOAL_TOLERANCE:
            self.settled += 1
        else:
            self.settled = 0
        if self.settled >= SETTLE_TICKS:
            self.arrived = True
            self.send([0.0, 0.0, 0.0])
            self.on_arrival(self.robot, (self.get_clock().now() - self.t_start).nanoseconds / 1e9)
            return
        self.send(wheel)


def main():
    """Given. One RobotNode per robot; logs "task 2A complete" at the end."""
    rclpy.init()
    ready = set()
    arrived = {}

    def on_arrival(robot, t):
        arrived[robot] = t
        if len(arrived) == len(ORDER):
            nodes[0].get_logger().info("task 2A complete in %.1fs" % max(arrived.values()))

    nodes = [RobotNode(r, ready, on_arrival) for r in ORDER]
    executor = MultiThreadedExecutor(num_threads=len(ORDER))
    for n in nodes:
        executor.add_node(n)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        for n in nodes:
            try:
                n.send((0.0, 0.0, 0.0))          # leave nothing driving
            except Exception:                    # no publisher yet, or Ctrl+C took the context down
                pass
            n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
