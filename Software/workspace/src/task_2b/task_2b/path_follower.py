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
*  This script is to implement Task 2B of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ 3142 ]
# Author List:      [ Shourya Gupta, Sarthak Gupta ]
# Filename:         path_follower.py
# Functions:        [ Add every extra helper function you write to this list ]
# Global variables: [ KP_POS, KI_POS, KD_POS, KP_YAW, KI_YAW, KD_YAW ]


############################ WHAT YOU HAVE TO DO ##############################
#
#  Drive r2 along the timed path on /r2/plan, stopping on each goal.
#
#   1. Paste in your Task 2A code      (the FROM TASK 2A section)
#   2. Write path_to_traj()            (the inverse of to_path_msg())
#
#  RobotNode is given: after the path's start time it aims your GoToPoint at
#  where the path says the robot should be, and holds it on each goal.
#
#  Run:  ros2 launch hb_description task2b.launch.py
#        ros2 run task_2b path_planner
#        ros2 run task_2b path_follower
#
###############################################################################

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray

################# ADD EXTRA IMPORTS / GLOBALS HERE ############
KP_POS, KI_POS, KD_POS = 2.5, 0.0, 0.15
KP_YAW, KI_YAW, KD_YAW = 3.0, 0.0, 0.1
###############################################################


# ----------------------------------------------------- follower (given)
ROBOTS = ("r2",)              # one node per robot
CONTROL_HZ = 50.0
GOAL_TOLERANCE = 0.020        # m, "on the goal"
LEAD = 0.12                   # s, aim this far ahead on the path
MIN_STOP = 0.5                # s, a pause this long in a path is a goal
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

# ----------------------------------------------------- robot (same as Task 1B)
WHEEL_RADIUS_M = 0.0255       # m
CHASSIS_RADIUS_M = 0.06412    # m, chassis centre to each wheel's axle
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0])
IK_MATRIX = np.zeros((3, 3))
_CTRL_LIMIT = 30.0            # rad/s, the wheels' ctrlrange in the robot's MJCF:
                              # faster commands are clamped by the simulation


######################### FROM TASK 2A ########################
## Paste your Task 2A functions and GoToPoint here, unchanged.##
###############################################################

##############  ADD YOUR CODE HERE  ##############

def body_velocity_to_wheel_speeds(vx, vy, w):
    """Body twist (vx, vy, w) -> wheel speeds [left, right, back], rad/s."""
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
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    """Angle -> the same angle in (-pi, pi]. 350 degrees becomes -10."""
    return math.atan2(math.sin(a), math.cos(a))


def to_body(vx_a, vy_a, yaw):
    """Arena-frame velocity -> body-frame velocity (vx, vy).
    Check: at yaw = -pi/2 (facing up), arena (0, -v) must give body (v, 0)."""
    vx = (vx_a * math.cos(yaw)) + (vy_a * math.sin(yaw))                       
    vy = (-vx_a * math.sin(yaw)) + (vy_a * math.cos(yaw))

    return (vx, vy)


class GoToPoint:
    """From Task 2A. RobotNode creates it as
    GoToPoint(v_max=0.30, tol=0.012, dt=1.0 / CONTROL_HZ), so make your TUNED
    gains the defaults -- zero gains do nothing."""

    def __init__(self, v_max=0.0, w_max=2.0, kp=KP_POS, ki=KI_POS, kd=KD_POS,
                 kyaw=KP_YAW, kiyaw=KI_YAW, kdyaw=KD_YAW, tol=0.0, dt=0.0):
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

##################################################


######################### PATH ################################

def path_to_traj(msg):
    """nav_msgs/Path -> [(t, x, y), ...], t = pose stamp - path stamp, in s."""
    # TODO: Time.from_msg(stamp) gives a Time; (Time - Time).nanoseconds * 1e-9
    pass


def stops_in(traj):
    """Given. Goals on a timed path: [(arrive, leave, x, y), ...], in s.
    A goal is a pause of at least MIN_STOP, or the end; the last is never left."""
    stops = [(t0, t1, x1, y1) for (t0, x0, y0), (t1, x1, y1) in zip(traj, traj[1:])
             if (x0, y0) == (x1, y1) and t1 - t0 >= MIN_STOP]
    if not stops or stops[-1][2:] != traj[-1][1:]:
        stops.append((traj[-1][0], traj[-1][0], *traj[-1][1:]))
    arrive, _, x, y = stops[-1]
    stops[-1] = (arrive, math.inf, x, y)
    return stops


def position_at(traj, t):
    """Given. Where the path says the robot should be at time t."""
    if t <= traj[0][0]:
        return traj[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(traj, traj[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return traj[-1][1:]


######################### NODE (given) ########################

class RobotNode(Node):
    """Drives one robot along its /rN/plan. Node: task_2b_r2."""

    def __init__(self, robot, on_arrival):
        super().__init__(f"task_2b_{robot}")
        self.robot, self.on_arrival = robot, on_arrival
        self.pose = self.hold_yaw = self.traj = self.start = None
        self.stops, self.reached, self.arrived = [], 0, False
        self.held = 0.0                         # s the path clock has been held on a goal
        self.create_subscription(Odometry, f"/{robot}/odom", self.on_odom, 10)
        self.create_subscription(Path, f"/{robot}/plan", self.on_path, LATCHED)
        self.pub = self.create_publisher(Float64MultiArray, f"/{robot}/wheel_commands", 10)
        self.ctl = GoToPoint(v_max=0.30, tol=0.012, dt=1.0 / CONTROL_HZ)
        self.create_timer(1.0 / CONTROL_HZ, self.tick)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(q.w, q.x, q.y, q.z))
        if self.hold_yaw is None:
            self.hold_yaw = self.pose[2]        # keep the starting heading throughout

    def on_path(self, msg):
        if self.traj is None and msg.poses:
            self.traj = path_to_traj(msg)
            self.stops = stops_in(self.traj)
            self.start = Time.from_msg(msg.header.stamp)
            self.get_logger().info("got my path: %d poses, arrive at t=%.1f s"
                                   % (len(msg.poses), self.traj[-1][0]))

    def send(self, wheels):
        self.pub.publish(Float64MultiArray(data=[float(w) for w in wheels]))

    def tick(self):
        if self.traj is None or self.pose is None or self.arrived:
            self.send((0.0, 0.0, 0.0))
            return
        now = (self.get_clock().now() - self.start).nanoseconds * 1e-9
        if now < 0:                             # not the start time yet
            self.send((0.0, 0.0, 0.0))
            return
        t = now - self.held                     # where on the path the robot should be
        arrive, leave, *goal = self.stops[self.reached]
        to_go = math.dist(self.pose[:2], goal)
        which = "its goal" if len(self.stops) == 1 else "goal %d of %d" % (self.reached + 1, len(self.stops))
        if t >= arrive:                         # the path is waiting on this goal
            if to_go < GOAL_TOLERANCE:
                self.get_logger().info("%s reached %s (%.0f mm out) at t=%.1fs"
                                       % (self.robot, which, to_go * 1000, now))
                self.reached += 1
                if self.reached == len(self.stops):
                    self.arrived = True
                    self.send((0.0, 0.0, 0.0))
                    self.on_arrival(self.robot, now)
                    return
            elif t > leave:                     # the stop is over but the robot is not on
                self.held += t - leave          # the goal yet: hold the path clock until it is
                t = leave
        # on a stop, aim at the goal itself: LEAD would aim down the next leg
        target = goal if t >= arrive else position_at(self.traj, t + LEAD)
        wheels, _ = self.ctl.step(self.pose, target, self.hold_yaw)
        self.send(wheels)


def main():
    rclpy.init()
    arrived = {}

    def on_arrival(robot, t):
        arrived[robot] = t
        if len(arrived) == len(ROBOTS):
            nodes[0].get_logger().info("task 2B complete in %.1fs" % max(arrived.values()))

    nodes = [RobotNode(r, on_arrival) for r in ROBOTS]
    executor = MultiThreadedExecutor(num_threads=len(ROBOTS))
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
            except Exception:
                pass
            n.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
