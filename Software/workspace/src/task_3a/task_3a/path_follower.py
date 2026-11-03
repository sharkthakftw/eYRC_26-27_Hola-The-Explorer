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
*  This script is to implement Task 3A of Hola The Explorer (HE) Theme (eYRC 2026-27).
*
*****************************************************************************************
'''

# Team ID:          [ 3142 ]
# Author List:      [ Shourya Gupta, Sarthak Gupta ]
# Filename:         path_follower.py
# Functions:        body_velocity_to_wheel_speeds(), body_to_wheels(), yaw_from_quat(),
#                   wrap(), to_body(), path_to_traj(), position_at(),
#                   RobotNode.request_attach(), RobotNode.request_detach(), ...
# Global variables: KP_POS, KI_POS, KD_POS, KP_YAW, KI_YAW, KD_YAW, HOLD_YAW, SLOT, ...

############################ WHAT THIS DOES ###################################
#
#   WAIT_PLAN -> APPROACH -> DOCK -> (attach) -> CARRY -> (detach) -> RETURN -> DONE
#
#  The robot faces right (yaw 0) for the whole task, so the crate stays in front
#  of the arrow while carried.
#
###############################################################################

import math

import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float64MultiArray
from hb_description.srv import AttachPayload, DetachPayload

################# ADD EXTRA IMPORTS / GLOBALS HERE ############
KP_POS, KI_POS, KD_POS = 2.5, 0.0, 0.15
KP_YAW, KI_YAW, KD_YAW = 3.0, 0.0, 0.1

ROBOT = "r2"
PAYLOAD = "payload_beta"
SLOT = (-0.1798, 0.0)          # robot centre minus crate centre when docked
HOLD_YAW = 0.0                 # face right (+x): the docking heading, kept all task

V_CRUISE = 0.30                # m/s caps
V_CARRY = 0.20
V_NEAR = 0.08                  # m/s, along the last SLOW_PATH metres of every path
SLOW_PATH = 0.30               # m of path left
V_SLOW = 0.06                  # m/s, final creep onto a goal / docking
KP_TRACK = 4.0                 # 1/s, pull back onto the path (on top of the path's own velocity)
ERR_HOLD = 0.05                # m, the path clock waits when the robot is this far behind it
CTL_TOL = 0.003                # m, GoToPoint stops commanding inside this

DOCK_TOL = 0.004               # m, "on the docking point"
CARRY_TOL = 0.005              # m, crate centre on the hexagon centre
RETURN_TOL = 0.010             # m, back on the start circle
SETTLE_TICKS = 15              # ticks in a row inside the tolerance

CREEP = 0.003                  # m, nudge forward after a refused attach
MAX_ATTACH_TRIES = 8
###############################################################


# ----------------------------------------------------- follower (given)
CONTROL_HZ = 50.0
LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)

# ----------------------------------------------------- robot (same as Task 1B)
WHEEL_RADIUS_M = 0.0255
CHASSIS_RADIUS_M = 0.06412
WHEEL_ANGLES_RAD = np.radians([30.0, 150.0, 270.0])
IK_MATRIX = np.zeros((3, 3))
_CTRL_LIMIT = 30.0             # rad/s


######################### FROM TASK 2A ########################

def body_velocity_to_wheel_speeds(vx, vy, w):
    """Body twist (vx, vy, w) -> wheel speeds [left, right, back], rad/s."""
    angles = WHEEL_ANGLES_RAD
    linear_speed = vx * np.cos(angles) + vy * np.sin(angles)
    rotational_speed = w * CHASSIS_RADIUS_M
    return (linear_speed + rotational_speed) / WHEEL_RADIUS_M


def body_to_wheels(vx, vy, wz):
    """Like body_velocity_to_wheel_speeds(), but kept inside +/-_CTRL_LIMIT."""
    raw = body_velocity_to_wheel_speeds(vx, vy, wz)
    max_w = max(abs(w) for w in raw)
    if max_w > _CTRL_LIMIT:
        return [w * (_CTRL_LIMIT / max_w) for w in raw]
    return raw


def yaw_from_quat(w, x, y, z):
    """Quaternion -> yaw, radians."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap(a):
    """Angle -> the same angle in (-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


def to_body(vx_a, vy_a, yaw):
    """Arena-frame velocity -> body-frame velocity (vx, vy)."""
    vx = (vx_a * math.cos(yaw)) + (vy_a * math.sin(yaw))
    vy = (-vx_a * math.sin(yaw)) + (vy_a * math.cos(yaw))
    return (vx, vy)


class GoToPoint:
    """PID controller: drive one robot straight at a point, holding a fixed yaw.
    step()    -> go to a point (used to hold, dock and finish a path)
    step_ff() -> follow a moving reference: its own velocity + a pull onto it"""

    def __init__(self, v_max=V_CRUISE, w_max=2.0, kp=KP_POS, ki=KI_POS, kd=KD_POS,
                 kyaw=KP_YAW, kiyaw=KI_YAW, kdyaw=KD_YAW, tol=CTL_TOL, dt=1.0 / CONTROL_HZ):
        self.v_max, self.w_max = v_max, w_max
        self.kp, self.ki, self.kd = kp, ki, kd
        self.kyaw, self.kiyaw, self.kdyaw = kyaw, kiyaw, kdyaw
        self.tol = tol
        self.dt = dt
        self.integral_limit = 1.0
        self.reset()

    def reset(self):
        """Forget PID history (new leg / new target) so the first derivative
        term does not kick."""
        self.prev_error_x = self.prev_error_y = self.prev_error_yaw = 0.0
        self.integral_x = self.integral_y = self.integral_yaw = 0.0
        self.primed = False
        self.yaw_primed = False

    def _wz(self, pose_yaw, hold_yaw):
        error_yaw = wrap(hold_yaw - pose_yaw)
        if not self.yaw_primed:
            self.prev_error_yaw = error_yaw
            self.yaw_primed = True
        self.integral_yaw += error_yaw * self.dt
        self.integral_yaw = max(-self.integral_limit, min(self.integral_limit, self.integral_yaw))
        deriv_yaw = (error_yaw - self.prev_error_yaw) / self.dt
        self.prev_error_yaw = error_yaw
        wz = (self.kyaw * error_yaw + self.kiyaw * self.integral_yaw
              + self.kdyaw * deriv_yaw)
        return max(-self.w_max, min(self.w_max, wz))

    def step(self, pose, target, hold_yaw):
        """pose (x, y, yaw), target (x, y), hold_yaw -> (wheels, distance)."""
        target_x, target_y = target
        pose_x, pose_y, pose_yaw = pose

        error_x = target_x - pose_x
        error_y = target_y - pose_y
        dist = math.hypot(error_x, error_y)

        if not self.primed:
            self.prev_error_x, self.prev_error_y = error_x, error_y
            self.primed = True

        if dist <= self.tol:
            vx_a = vy_a = 0.0
            self.integral_x = self.integral_y = 0.0
        else:
            self.integral_x += error_x * self.dt
            self.integral_y += error_y * self.dt
            lim = self.integral_limit
            self.integral_x = max(-lim, min(lim, self.integral_x))
            self.integral_y = max(-lim, min(lim, self.integral_y))

            deriv_x = (error_x - self.prev_error_x) / self.dt
            deriv_y = (error_y - self.prev_error_y) / self.dt

            vx_a = self.kp * error_x + self.ki * self.integral_x + self.kd * deriv_x
            vy_a = self.kp * error_y + self.ki * self.integral_y + self.kd * deriv_y

            speed = math.hypot(vx_a, vy_a)
            if speed > self.v_max:
                vx_a *= self.v_max / speed
                vy_a *= self.v_max / speed

        self.prev_error_x, self.prev_error_y = error_x, error_y

        wz = self._wz(pose_yaw, hold_yaw)
        vx_b, vy_b = to_body(vx_a, vy_a, pose_yaw)
        return (body_to_wheels(vx_b, vy_b, wz), dist)

    def step_ff(self, pose, ref_xy, v_ff, hold_yaw):
        """Follow a moving reference point: arena velocity = the path's own
        velocity (v_ff) + KP_TRACK * (reference - pose). Because the path's velocity
        is fed forward, the robot does not trail behind it and cut corners."""
        vx_a = v_ff[0] + KP_TRACK * (ref_xy[0] - pose[0])
        vy_a = v_ff[1] + KP_TRACK * (ref_xy[1] - pose[1])
        speed = math.hypot(vx_a, vy_a)
        if speed > self.v_max:
            vx_a *= self.v_max / speed
            vy_a *= self.v_max / speed
        self.primed = False                       # so a later step() does not inherit stale errors
        wz = self._wz(pose[2], hold_yaw)
        vx_b, vy_b = to_body(vx_a, vy_a, pose[2])
        return body_to_wheels(vx_b, vy_b, wz)


######################### FROM TASK 2B ########################

def path_to_traj(msg):
    """nav_msgs/Path -> [(t, x, y), ...], t = pose stamp - path stamp, in s."""
    path_time = Time.from_msg(msg.header.stamp)
    traj = []
    for ps in msg.poses:
        t = (Time.from_msg(ps.header.stamp) - path_time).nanoseconds * 1e-9
        traj.append((t, ps.pose.position.x, ps.pose.position.y))
    return traj


def position_at(traj, t):
    """Where the path says the robot should be at time t."""
    if t <= traj[0][0]:
        return traj[0][1:]
    for (t0, x0, y0), (t1, x1, y1) in zip(traj, traj[1:]):
        if t <= t1:
            u = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
    return traj[-1][1:]


######################### ONE LEG OF THE JOB ##################

class Leg:
    """A timed path being followed. `offset` is added to every path point
    (SLOT while carrying: the path is the crate's centre, the robot is SLOT away).

    The path is followed with its OWN clock `s`, not the wall clock: `s` only
    advances while the robot keeps up with the path. A late start, a slow turn
    or a lagging robot therefore never makes the reference run ahead and drag the
    robot in a straight line across obstacles to the goal."""

    def __init__(self, name, traj, start, offset, v_max, tol):
        self.name, self.traj, self.start = name, traj, start
        self.offset, self.v_max, self.tol = offset, v_max, tol
        self.end = traj[-1][0]
        self.goal = (traj[-1][1] + offset[0], traj[-1][2] + offset[1])
        self.s = traj[0][0]                       # path clock
        self.cum = [0.0]                          # arc length at each waypoint
        for (t0, x0, y0), (t1, x1, y1) in zip(traj, traj[1:]):
            self.cum.append(self.cum[-1] + math.hypot(x1 - x0, y1 - y0))
        self.length = self.cum[-1]

    def ref(self, s):
        """Path state at clock s -> (x, y, vx, vy, arc length still to go)."""
        ox, oy = self.offset
        traj = self.traj
        if s <= traj[0][0]:
            return (traj[0][1] + ox, traj[0][2] + oy, 0.0, 0.0, self.length)
        for i, ((t0, x0, y0), (t1, x1, y1)) in enumerate(zip(traj, traj[1:])):
            if s <= t1:
                seg = t1 - t0
                u = (s - t0) / seg if seg > 0 else 1.0
                x, y = x0 + (x1 - x0) * u, y0 + (y1 - y0) * u
                vx = (x1 - x0) / seg if seg > 0 else 0.0
                vy = (y1 - y0) / seg if seg > 0 else 0.0
                rem = self.length - (self.cum[i] + math.hypot(x - x0, y - y0))
                return (x + ox, y + oy, vx, vy, max(rem, 0.0))
        return (traj[-1][1] + ox, traj[-1][2] + oy, 0.0, 0.0, 0.0)


######################### NODE ################################

class RobotNode(Node):
    """Approaches, docks, attaches, carries, lets go and drives back. Node: task_3a_r2."""

    def __init__(self):
        super().__init__("task_3a_r2")
        self.pose = None
        self.state = "WAIT_PLAN"
        self.leg = None
        self.plan_traj = self.plan_start = None
        self.carry_traj = self.return_traj = None
        self.settled = 0
        self.pending = False
        self.attempts = 0
        self.dock_goal = None
        self.hold_goal = None
        self.t0 = None

        self.create_subscription(Odometry, f"/{ROBOT}/odom", self.on_odom, 10)
        self.create_subscription(Path, f"/{ROBOT}/plan", self.on_plan, LATCHED)
        self.create_subscription(Path, "/carry_plan", self.on_carry, LATCHED)
        self.create_subscription(Path, f"/{ROBOT}/return_plan", self.on_return, LATCHED)
        self.pub = self.create_publisher(Float64MultiArray, f"/{ROBOT}/wheel_commands", 10)

        self.attach_cli = self.create_client(AttachPayload, f"/{ROBOT}/attach_payload")
        self.detach_cli = self.create_client(DetachPayload, f"/{ROBOT}/detach_payload")

        self.ctl = GoToPoint(v_max=V_CRUISE, tol=CTL_TOL, dt=1.0 / CONTROL_HZ)
        self.create_timer(1.0 / CONTROL_HZ, self.tick)

    # ------------------------------------------------ subscriptions
    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.pose = (p.x, p.y, yaw_from_quat(q.w, q.x, q.y, q.z))

    def on_plan(self, msg):
        if self.plan_traj is None and msg.poses:
            self.plan_traj = path_to_traj(msg)
            self.plan_start = Time.from_msg(msg.header.stamp)
            self.get_logger().info("got /r2/plan: %d poses" % len(msg.poses))

    def on_carry(self, msg):
        if self.carry_traj is None and msg.poses:
            self.carry_traj = path_to_traj(msg)
            self.get_logger().info("got /carry_plan: %d poses" % len(msg.poses))

    def on_return(self, msg):
        if self.return_traj is None and msg.poses:
            self.return_traj = path_to_traj(msg)
            self.get_logger().info("got /r2/return_plan: %d poses" % len(msg.poses))

    def send(self, wheels):
        self.pub.publish(Float64MultiArray(data=[float(w) for w in wheels]))

    # ------------------------------------------------ services (never block)
    def request_attach(self):
        """Send AttachPayload with call_async; the reply is handled in a done-callback."""
        if not self.attach_cli.service_is_ready():
            self.get_logger().warn("attach service not ready yet")
            return
        req = AttachPayload.Request()
        req.payload_name = PAYLOAD
        self.pending = True
        self.attach_cli.call_async(req).add_done_callback(self.on_attach_reply)

    def on_attach_reply(self, future):
        self.pending = False
        try:
            reply = future.result()
        except Exception as e:                                  # call itself failed
            reply = None
            self.get_logger().warn("attach call failed: %s" % e)

        if reply is not None and reply.success:
            self.get_logger().info("attached %s: carrying" % PAYLOAD)
            self.ctl.reset()
            self.settled = 0
            self.state = "WAIT_CARRY"                           # tick() starts the leg
            return

        msg = reply.message if reply is not None else "no reply"
        self.attempts += 1
        self.get_logger().warn("attach refused (%d/%d): %s" % (self.attempts, MAX_ATTACH_TRIES, msg))
        if self.attempts >= MAX_ATTACH_TRIES:
            self.get_logger().error("giving up on attaching")
            self.state = "FAILED"
            return
        if "fac" not in msg.lower():                            # not a heading problem: close the gap
            self.dock_goal = (self.dock_goal[0] + CREEP, self.dock_goal[1])
        self.settled = 0

    def request_detach(self):
        """Send DetachPayload with call_async; the reply is handled in a done-callback."""
        if not self.detach_cli.service_is_ready():
            self.get_logger().warn("detach service not ready yet")
            return
        req = DetachPayload.Request()
        req.payload_name = PAYLOAD
        self.pending = True
        self.detach_cli.call_async(req).add_done_callback(self.on_detach_reply)

    def on_detach_reply(self, future):
        self.pending = False
        try:
            reply = future.result()
        except Exception as e:
            reply = None
            self.get_logger().warn("detach call failed: %s" % e)

        if reply is not None and getattr(reply, "success", True):
            self.get_logger().info("let go of %s" % PAYLOAD)
            self.ctl.reset()
            self.settled = 0
            self.state = "WAIT_RETURN"
        else:
            self.get_logger().warn("detach refused: %s"
                                   % (getattr(reply, "message", "") if reply else "no reply"))
            self.settled = 0

    # ------------------------------------------------ driving helpers
    def hold(self, target=None):
        """Stay put (or on `target`) and keep the heading."""
        target = target if target is not None else self.pose[:2]
        self.ctl.v_max = V_SLOW
        self.send(self.ctl.step(self.pose, target, HOLD_YAW)[0])

    def start_leg(self, leg):
        self.leg = leg
        self.settled = 0
        self.ctl.reset()
        late = (self.get_clock().now() - leg.start).nanoseconds * 1e-9
        self.get_logger().info("%s leg: %.1f s of path, %.2f m%s"
                               % (leg.name, leg.end, leg.length,
                                  " (plan stamp is %.0f s old: following it from the start anyway)" % late
                                  if late > 1.0 else ""))

    def follow_leg(self):
        leg = self.leg
        if (self.get_clock().now() - leg.start).nanoseconds < 0:     # not time yet: just turn
            self.hold()
            return
        dt = 1.0 / CONTROL_HZ

        rx, ry, vx, vy, rem = leg.ref(leg.s)
        err = math.hypot(rx - self.pose[0], ry - self.pose[1])

        # slow along the last SLOW_PATH metres; wait for the robot if it falls behind
        v_path = math.hypot(vx, vy)
        cap = V_NEAR if rem < SLOW_PATH else leg.v_max
        k = min(1.0, cap / v_path) if v_path > 1e-9 else 1.0
        k *= max(0.0, 1.0 - err / ERR_HOLD)
        leg.s = min(leg.s + dt * k, leg.end)
        rx, ry, vx, vy, rem = leg.ref(leg.s)

        to_go = math.dist(self.pose[:2], leg.goal)
        if leg.s >= leg.end:                         # path finished: creep onto the goal
            self.ctl.v_max = V_SLOW
            self.send(self.ctl.step(self.pose, leg.goal, HOLD_YAW)[0])
        else:
            self.ctl.v_max = leg.v_max
            self.send(self.ctl.step_ff(self.pose, (rx, ry), (vx * k, vy * k), HOLD_YAW))

        self.settled = self.settled + 1 if (leg.s >= leg.end and to_go < leg.tol) else 0
        if self.settled >= SETTLE_TICKS:
            self.settled = 0
            self.leg_done()

    def leg_done(self):
        name = self.leg.name
        self.send((0.0, 0.0, 0.0))
        self.ctl.reset()
        if name == "approach":
            self.dock_goal = self.leg.goal
            self.attempts = 0
            self.state = "DOCK"
        elif name == "carry":
            self.hold_goal = self.leg.goal
            self.state = "RELEASE"
        elif name == "return":
            self.state = "DONE"
            self.get_logger().info("task 3A complete in %.1fs"
                                   % ((self.get_clock().now() - self.t0).nanoseconds * 1e-9))

    def dock(self):
        """Creep onto the docking point, stop, then ask for the attach."""
        self.ctl.v_max = V_SLOW
        wheels, dist = self.ctl.step(self.pose, self.dock_goal, HOLD_YAW)
        self.send(wheels)
        facing = abs(wrap(HOLD_YAW - self.pose[2])) < 0.03
        if dist < DOCK_TOL and facing and not self.pending:
            self.settled += 1
        else:
            self.settled = 0
        if self.settled >= SETTLE_TICKS and not self.pending:
            self.settled = 0
            self.request_attach()

    def release(self):
        """On the hexagon's centre: stand still, then ask for the detach."""
        self.hold(self.hold_goal)
        if not self.pending and math.dist(self.pose[:2], self.hold_goal) < CARRY_TOL:
            self.settled += 1
        else:
            self.settled = 0
        if self.settled >= SETTLE_TICKS and not self.pending:
            self.settled = 0
            self.request_detach()

    # ------------------------------------------------ main loop
    def tick(self):
        if self.pose is None:
            self.send((0.0, 0.0, 0.0))
            return
        self.get_logger().info("state=%s yaw=%.2f (want %.2f)" % (self.state, self.pose[2], HOLD_YAW),
                               throttle_duration_sec=1.0)
        s = self.state

        if s == "WAIT_PLAN":
            self.hold()                                  # turn to face right while waiting
            if self.plan_traj is not None:
                self.t0 = self.plan_start
                self.state = "APPROACH"
                self.start_leg(Leg("approach", self.plan_traj, self.plan_start,
                                   (0.0, 0.0), V_CRUISE, DOCK_TOL))
        elif s == "APPROACH" or s == "CARRY" or s == "RETURN":
            self.follow_leg()
        elif s == "DOCK":
            self.dock()
        elif s == "WAIT_CARRY":
            self.hold(self.dock_goal)
            if self.carry_traj is not None:
                self.state = "CARRY"
                # the path is the crate's centre; the robot sits SLOT away from it
                self.start_leg(Leg("carry", self.carry_traj, self.get_clock().now(),
                                   SLOT, V_CARRY, CARRY_TOL))
        elif s == "RELEASE":
            self.release()
        elif s == "WAIT_RETURN":
            self.hold(self.hold_goal)
            if self.return_traj is not None:
                self.state = "RETURN"
                self.start_leg(Leg("return", self.return_traj, self.get_clock().now(),
                                   (0.0, 0.0), V_CRUISE, RETURN_TOL))
        else:                                            # DONE / FAILED
            self.send((0.0, 0.0, 0.0))


def main():
    rclpy.init()
    node = RobotNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.send((0.0, 0.0, 0.0))                   # leave nothing driving
        except Exception:
            pass
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()