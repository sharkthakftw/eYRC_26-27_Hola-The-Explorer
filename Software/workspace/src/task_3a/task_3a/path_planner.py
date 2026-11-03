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
# Filename:         path_planner.py
# Functions:        find_payload(), debug_payload(), mark_square(), remove_drivable(),
#                   carry_grid(), a_star(), is_line_clear(), smooth(), plan_leg(),
#                   plan_route(), plan_carry(), PathPlanner.to_path_msg()
# Global variables: CRATE_HALF, SLOT, DESTINATION, START, CARRY_SPEED, TEAL_LOW,
#                   TEAL_HIGH, MIN_TOP_AREA, PARALLAX


############################ WHAT YOU HAVE TO DO ##############################
#
#  r2 docks on the left of payload_beta, carries it to the hexagon's centre and
#  returns to its start circle. This node plans all three legs ONCE:
#
#    /r2/plan         robot: start circle -> docking point (crate = obstacle)
#    /carry_plan      crate centre: where it is -> DESTINATION (carry grid)
#    /r2/return_plan  robot: next to the hexagon -> start circle
#
###############################################################################

import heapq
import math

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from rclpy.time import Time
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path

################# ADD EXTRA IMPORTS / GLOBALS HERE ############
CRATE_HALF = 0.1089                  # m, half the crate's footprint
DESTINATION = (1.2192, 1.2192)       # hexagon centre
SLOT = (-0.1798, 0.0)                # robot centre minus crate centre when docked
START = (1.2192, 2.2616)             # r2's start circle

CARRY_SPEED = 0.15                   # m/s, the crate path is timed at this

# teal (verdigris) top of payload_beta, OpenCV HSV (H 0-179). TUNE with debug_payload()
TEAL_LOW, TEAL_HIGH = (78, 120, 140), (95, 255, 255)
MIN_TOP_AREA = 400                   # px^2, ignore smaller teal blobs

# The camera sees the raised top pushed outwards from the point under it
# (taken as the arena centre). 1.0 = no correction. TUNE: calibrate until the
# robot docks with the solenoid touching.
PARALLAX = 0.95

PAD_RADIUS = 0.19                    # m, the crate's round pad (its dark rim is NOT an obstacle)
TIGHT_MARGIN = 0.010                 # m, clearance kept from the crate itself on the approach
APPROACH_MARGIN = 0.008              # m, pillar/wall clearance allowed ONLY where the dock needs it
SOFT_COST = 8.0                      # extra cost per metre inside the normal-margin zone
###############################################################


# ----------------------------------------------------- arena (given, do not edit)
ROBOTS = ("r2",)
START_CIRCLES = [(0.8969, 2.2616), (1.2192, 2.2616), (1.5415, 2.2616)]
ARENA_SIZE = 2.4384                  # m
ROBOT_RADIUS = 0.084                 # m
SAFETY_MARGIN = 0.030                # m, kept clear on top of the robot radius

# ----------------------------------------------------- camera (given)
CAMERA_URL = "http://127.0.0.1:8080/stream"
ARENA_LEFT, ARENA_TOP, ARENA_PIXELS = 304, 24, 671       # floor square in the image
GRID_CELLS = 244                     # occupancy grid is GRID_CELLS x GRID_CELLS
CELL_SIZE = ARENA_SIZE / GRID_CELLS  # m, about 1 cm
SAND_LOW, SAND_HIGH = (10, 40, 170), (30, 120, 255)      # sand colour, HSV

# ----------------------------------------------------- planning (tunable)
PLAN_CELL = 0.04                     # m, planning grid cell
PLAN_SPEED = 0.24                    # m/s, the speed the robot paths are timed at
START_DELAY = 2.0                    # s, from publishing to setting off

LATCHED = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)


######################### OCCUPANCY GRID ######################

def read_frame(url=CAMERA_URL):
    """Given. One BGR frame from the overhead camera."""
    cap = cv2.VideoCapture(url)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("no frame from %s -- is task3a.launch.py running?" % url)
    return frame


def crop_arena(frame):
    """Given. The arena floor only, ARENA_PIXELS square."""
    return frame[ARENA_TOP:ARENA_TOP + ARENA_PIXELS, ARENA_LEFT:ARENA_LEFT + ARENA_PIXELS]


def frame_to_occupancy(frame):
    """Given. Frame -> grid[row, col] = grid[y, x], True = obstacle."""
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    obstacles = cv2.bitwise_not(cv2.inRange(hsv, SAND_LOW, SAND_HIGH))
    obstacles = cv2.morphologyEx(obstacles, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(obstacles, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(obstacles, contours, -1, 255, cv2.FILLED)
    small = cv2.resize(obstacles, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA)
    return small > 0


def clear_circle(occupied, xy, radius):
    """Given. Free every cell within `radius` m of point xy (in place)."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    occupied[(x - xy[0]) ** 2 + (y - xy[1]) ** 2 <= radius ** 2] = False


########################## NEW IN 3A ##########################

def find_payload(frame):
    """Camera frame -> (x, y) of payload_beta's centre in the /odom frame (m),
    or None if its teal top is not seen."""
    crop = crop_arena(frame)
    hsv = cv2.cvtColor(cv2.medianBlur(crop, 5), cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, TEAL_LOW, TEAL_HIGH)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= MIN_TOP_AREA]
    if not contours:
        return None
    blob = max(contours, key=cv2.contourArea)
    m = cv2.moments(blob)
    if m["m00"] < 1e-6:
        return None
    cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]       # px in the arena crop

    # pixels -> metres (top-left origin, x right, y down)
    scale = ARENA_SIZE / ARENA_PIXELS
    x, y = cx * scale, cy * scale

    # the top is above the floor: pull it back towards the point under the camera
    mid = ARENA_SIZE / 2.0
    x = mid + (x - mid) * PARALLAX
    y = mid + (y - mid) * PARALLAX
    return (x, y)


def debug_payload(frame, path="/tmp/payload_mask.png"):
    """Helper: save the teal mask so TEAL_LOW / TEAL_HIGH can be tuned."""
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    cv2.imwrite(path, cv2.inRange(hsv, TEAL_LOW, TEAL_HIGH))


def find_tops(frame):
    """Centres (x, y) in m of every compact, saturated payload top in the frame
    (all four payloads: orange, pink, teal, purple). Used to keep the OTHER
    payloads as obstacles: their saturated tops would otherwise be freed as
    'painted floor' by remove_drivable()."""
    hsv = cv2.cvtColor(cv2.medianBlur(crop_arena(frame), 5), cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 1] > 110) & (hsv[..., 2] > 90)).astype(np.uint8) * 255
    m[(hsv[..., 0] >= 8) & (hsv[..., 0] <= 32) & (hsv[..., 1] < 130)] = 0     # sand
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, _, stats, cen = cv2.connectedComponentsWithStats(m)

    px = ARENA_SIZE / ARENA_PIXELS                   # m per pixel
    lo, hi = (0.08 / px) ** 2, (0.25 / px) ** 2      # a top is roughly 0.14 m square
    mid = ARENA_SIZE / 2.0
    tops = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if not (lo <= area <= hi) or not (0.6 <= w / h <= 1.6) or area / float(w * h) < 0.55:
            continue
        cx = mid + (cen[i][0] * px - mid) * PARALLAX
        cy = mid + (cen[i][1] * px - mid) * PARALLAX
        if math.dist((cx, cy), DESTINATION) < 0.25:  # the hexagon's gold centre is not a payload
            continue
        tops.append((cx, cy))
    return tops


def mark_square(occupied, xy, half, value):
    """Set every cell within a square of half-size `half` m around xy (in place)."""
    rows, cols = np.indices(occupied.shape)
    x, y = (cols + 0.5) * CELL_SIZE, (rows + 0.5) * CELL_SIZE
    occupied[(np.abs(x - xy[0]) <= half) & (np.abs(y - xy[1]) <= half)] = value


def remove_drivable(frame, occupied, robots, goals, payload=None):
    """Free what a robot may drive over; return the grid.
    Keeps the pillars and the crate (the crate's square is re-marked at the end,
    because its saturated teal top would otherwise be freed as 'painted floor')."""

    # free the painted floor: obstacles have low saturation
    hsv = cv2.cvtColor(crop_arena(frame), cv2.COLOR_BGR2HSV)
    saturation = cv2.split(hsv)[1]
    painted = (saturation > 30).astype(np.float32)
    frac = cv2.resize(painted, (GRID_CELLS, GRID_CELLS), interpolation=cv2.INTER_AREA)
    occupied[frac > 0.65] = False

    # remove lines and specks
    occupied = cv2.morphologyEx(occupied.astype(np.uint8), cv2.MORPH_OPEN,
                                np.ones((2, 2), np.uint8)).astype(bool)

    for xy in START_CIRCLES:
        clear_circle(occupied, xy, ROBOT_RADIUS + 0.02)
    for xy in robots:
        clear_circle(occupied, xy, ROBOT_RADIUS + 0.02)
    for xy in goals:
        clear_circle(occupied, xy, ROBOT_RADIUS + 0.02)

    if payload is None:
        payload = find_payload(frame)
    for top in find_tops(frame):                     # the other payloads stay obstacles
        if payload is None or math.dist(top, payload) > 0.15:
            mark_square(occupied, top, CRATE_HALF + 0.005, True)
    if payload is not None:
        clear_circle(occupied, payload, PAD_RADIUS)          # pad rim is drivable
        mark_square(occupied, payload, CRATE_HALF + 0.005, True)
    return occupied


######################### PLANNING GRID #######################

def planning_grid(occupied, margin=SAFETY_MARGIN):
    """Given. Grid -> blocked[i, j] = blocked[x, y] (NOTE: x first) of PLAN_CELL
    cells. Obstacles grow by ROBOT_RADIUS + SAFETY_MARGIN."""
    pad = ROBOT_RADIUS + margin
    k = int(math.ceil(pad / CELL_SIZE))
    disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    inflated = cv2.dilate(occupied.astype(np.uint8), disc) > 0

    n = int(round(ARENA_SIZE / PLAN_CELL))
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T        # [row=y, col=x] -> [i=x, j=y]
    near_wall = (centre < pad) | (centre > ARENA_SIZE - pad)
    blocked[near_wall, :] = True
    blocked[:, near_wall] = True
    return blocked


def carry_grid(occupied):
    """Grid for planning the crate's CENTRE as a point while the robot holds its
    heading (robot + crate = one fixed shape: the crate square plus the robot's
    disc at SLOT). Same output layout as planning_grid(): blocked[i=x, j=y].
    NOTE: if your Task 3A template already ships carry_grid(), use that one."""
    m = SAFETY_MARGIN
    h = int(math.ceil(0.32 / CELL_SIZE))
    ys, xs = np.mgrid[-h:h + 1, -h:h + 1] * CELL_SIZE
    robot = (xs - SLOT[0]) ** 2 + (ys - SLOT[1]) ** 2 <= (ROBOT_RADIUS + m) ** 2
    crate = (np.abs(xs) <= CRATE_HALF + m) & (np.abs(ys) <= CRATE_HALF + m)
    kernel = (robot | crate).astype(np.uint8)
    # cv2.dilate: dst(c) = max over kernel offsets k of src(c + k)  ->  kernel = shape itself
    inflated = cv2.dilate(occupied.astype(np.uint8), kernel) > 0

    n = int(round(ARENA_SIZE / PLAN_CELL))
    centre = (np.arange(n) + 0.5) * PLAN_CELL
    fine = np.minimum((centre / CELL_SIZE).astype(int), GRID_CELLS - 1)
    blocked = inflated[np.ix_(fine, fine)].T

    # walls: crate centre limits (robot sticks out on the left)
    left = -SLOT[0] + ROBOT_RADIUS + m
    right = CRATE_HALF + m
    edge = max(CRATE_HALF, ROBOT_RADIUS) + m
    blocked[(centre < left) | (centre > ARENA_SIZE - right), :] = True
    blocked[:, (centre < edge) | (centre > ARENA_SIZE - edge)] = True
    return blocked


def crate_blocked(payload, margin=TIGHT_MARGIN):
    """blocked[i=x, j=y] on the planning grid: robot centres closer than
    ROBOT_RADIUS + TIGHT_MARGIN to the crate square. The crate gets a thinner
    margin than pillars so the docking point (0.07 m off its face) stays reachable."""
    n = int(round(ARENA_SIZE / PLAN_CELL))
    c = (np.arange(n) + 0.5) * PLAN_CELL
    dx = np.maximum(np.abs(c - payload[0]) - CRATE_HALF, 0.0)[:, None]
    dy = np.maximum(np.abs(c - payload[1]) - CRATE_HALF, 0.0)[None, :]
    return np.hypot(dx, dy) < (ROBOT_RADIUS + margin)


def to_cell(p):
    """Given. Point (x, y) in m -> planning cell (i, j)."""
    n = int(round(ARENA_SIZE / PLAN_CELL))
    return (min(max(int(p[0] / PLAN_CELL), 0), n - 1), min(max(int(p[1] / PLAN_CELL), 0), n - 1))


def to_world(c):
    """Given. Planning cell (i, j) -> its centre (x, y) in m."""
    return ((c[0] + 0.5) * PLAN_CELL, (c[1] + 0.5) * PLAN_CELL)


def nearest_free(blocked, p, max_r=0.30):
    """Given. The free cell closest to point p, or None within max_r m."""
    c = to_cell(p)
    if not blocked[c]:
        return c
    free = [(i, j) for i in range(blocked.shape[0]) for j in range(blocked.shape[1])
            if not blocked[i, j] and abs(i - c[0]) * PLAN_CELL <= max_r
            and abs(j - c[1]) * PLAN_CELL <= max_r]
    return min(free, key=lambda q: math.dist(to_world(q), p)) if free else None


######################### PATH SEARCH (from Task 2B) ##########

def a_star(blocked, src, dst, soft=None):
    heuristic = lambda c: math.hypot(c[0] - dst[0], c[1] - dst[1]) * PLAN_CELL
    open_heap = [(heuristic(src), 0.0, src)]
    came_from = {src: None}
    g_score = {src: 0.0}

    while open_heap:
        _, cost, curr = heapq.heappop(open_heap)
        if curr == dst:
            path = []
            while curr is not None:
                path.append(curr)
                curr = came_from[curr]
            return path[::-1]
        if cost > g_score.get(curr, float("inf")):
            continue                                  # stale heap entry
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                nb = (curr[0] + di, curr[1] + dj)
                if not (0 <= nb[0] < blocked.shape[0] and 0 <= nb[1] < blocked.shape[1]):
                    continue
                if blocked[nb]:
                    continue
                step = math.hypot(di, dj) * PLAN_CELL
                if soft is not None and soft[nb]:
                    step *= 1.0 + SOFT_COST          # allowed, but only if there is no other way
                ng = cost + step
                if ng < g_score.get(nb, float("inf")):
                    g_score[nb] = ng
                    came_from[nb] = curr
                    heapq.heappush(open_heap, (ng + heuristic(nb), ng, nb))
    return None


def is_line_clear(blocked, c0, c1):
    pts = max(abs(c1[0] - c0[0]), abs(c1[1] - c0[1])) * 2 + 1
    for k in range(int(pts) + 1):
        u = k / pts
        ci = int(round(c0[0] + (c1[0] - c0[0]) * u))
        cj = int(round(c0[1] + (c1[1] - c0[1]) * u))
        ci = max(0, min(ci, blocked.shape[0] - 1))
        cj = max(0, min(cj, blocked.shape[1] - 1))
        if blocked[ci, cj]:
            return False
    return True


def smooth(cells, blocked):
    """Straighten the path wherever a straight line is clear."""
    if len(cells) <= 2:
        return cells
    result = [cells[0]]
    anchor = 0
    while anchor < len(cells) - 1:
        far = len(cells) - 1
        while far > anchor + 1:
            if is_line_clear(blocked, cells[anchor], cells[far]):
                break
            far -= 1
        result.append(cells[far])
        anchor = far
    return result


LAST_FAIL = ""                                    # why the last plan_leg() returned None


def plan_leg(blocked, start, goal, speed, free_start=0.0, soft=None):
    """One leg: start -> goal on `blocked`, timed at `speed`.
    Returns [(t, x, y), ...] beginning (0.0, *start) and ending at goal, or None.
    free_start > 0 un-blocks that radius (m) around start first (the carried
    shape's own footprint can make its start cell look blocked)."""
    blocked = blocked.copy()
    if free_start > 0.0:
        n = blocked.shape[0]
        centre = (np.arange(n) + 0.5) * PLAN_CELL
        near = ((centre[:, None] - start[0]) ** 2 + (centre[None, :] - start[1]) ** 2
                <= free_start ** 2)
        blocked[near] = False

    global LAST_FAIL
    src = nearest_free(blocked, start)
    dst = nearest_free(blocked, goal)
    if src is None:
        LAST_FAIL = "no free cell within 0.30 m of the START %s" % (start,)
        return None
    if dst is None:
        LAST_FAIL = "no free cell within 0.30 m of the GOAL %s" % (goal,)
        return None
    cells = a_star(blocked, src, dst, soft)
    if cells is None:
        LAST_FAIL = ("A* found no path from cell %s to %s (start boxed in, or the "
                     "goal is cut off)" % (src, dst))
        return None
    cells = smooth(cells, blocked if soft is None else (blocked | soft))   # never cut across the soft zone

    plan = [(0.0, start[0], start[1])]
    prev, t = tuple(start), 0.0
    for cell in cells:
        xy = to_world(cell)
        d = math.dist(prev, xy)
        if d < 1e-6:
            continue
        t += d / speed
        plan.append((t, *xy))
        prev = xy
    d = math.dist(prev, goal)
    if d > 1e-6:                                   # the exact goal, not its cell centre
        t += d / speed
        plan.append((t, goal[0], goal[1]))
    return plan


def plan_route(blocked, start, goal, soft=None):
    """Robot leg start -> goal (Task 2B's planner for ONE leg). Used for the
    approach (goal = docking point) and the return (goal = START)."""
    return plan_leg(blocked, start, goal, PLAN_SPEED, soft=soft)


def plan_carry(blocked, payload_xy):
    """Crate centre: where find_payload() saw it -> DESTINATION, on the carry
    grid, timed at CARRY_SPEED."""
    return plan_leg(blocked, payload_xy, DESTINATION, CARRY_SPEED, free_start=0.12)

    ##################################################


######################### NODE ################################

class PathPlanner(Node):
    """Waits for /r2/odom, plans the three legs once and publishes them (latched)."""

    def __init__(self):
        super().__init__("task_3a_path_planner")
        self.pose = {}
        for r in ROBOTS:
            self.create_subscription(Odometry, f"/{r}/odom", self._odom_cb(r), 10)
        self.plan_pub = self.create_publisher(Path, "/r2/plan", LATCHED)
        self.carry_pub = self.create_publisher(Path, "/carry_plan", LATCHED)
        self.return_pub = self.create_publisher(Path, "/r2/return_plan", LATCHED)
        self.planned = False
        self.warned = False
        self.create_timer(0.2, self.tick)
        self.get_logger().info("task 3A planner: carry payload_beta to %s" % (DESTINATION,))

    def _odom_cb(self, robot):
        def cb(msg):
            p = msg.pose.pose.position
            self.pose[robot] = (p.x, p.y)
        return cb

    def to_path_msg(self, plan, start):
        """[(t, x, y), ...] and start (rclpy Time) -> nav_msgs/Path."""
        path = Path()
        path.header.frame_id = "odom"
        path.header.stamp = start.to_msg()
        for t, x, y in plan:
            ps = PoseStamped()
            ps.header.frame_id = "odom"
            ps.header.stamp = (start + Duration(seconds=t)).to_msg()
            ps.pose.position.x = x
            ps.pose.position.y = y
            path.poses.append(ps)
        return path

    def save_debug(self, frame, occupied, carried, cblocked):
        """Write /home/shourya/Downloads/dbg_*.png: look at them to see what blocks the route."""
        def up(b):                                   # planning grid [x, y] -> image [y, x]
            img = np.where(b.T, 0, 255).astype(np.uint8)
            return cv2.resize(img, (488, 488), interpolation=cv2.INTER_NEAREST)
        cv2.imwrite("/home/shourya/Downloads/dbg_occupied.png", np.where(occupied, 0, 255).astype(np.uint8))
        cv2.imwrite("/home/shourya/Downloads/dbg_carry_blocked.png", up(cblocked))
        cv2.imwrite("/home/shourya/Downloads/dbg_robot_blocked.png", up(planning_grid(occupied)))
        self.get_logger().info("saved /home/shourya/Downloads/dbg_occupied.png, dbg_carry_blocked.png, "
                               "dbg_robot_blocked.png (black = blocked)")

    def tick(self):
        if self.planned or len(self.pose) < len(ROBOTS):
            return
        robot_xy = self.pose["r2"]

        frame = read_frame()
        payload = find_payload(frame)
        if payload is None:
            if not self.warned:
                self.get_logger().warn("payload_beta not seen yet -- retrying "
                                       "(tune TEAL_LOW/TEAL_HIGH with debug_payload)")
                self.warned = True
            return
        self.planned = True
        self.get_logger().info("payload_beta found at (%.3f, %.3f)" % payload)

        dock = (payload[0] + SLOT[0], payload[1] + SLOT[1])
        dock_end = (DESTINATION[0] + SLOT[0], DESTINATION[1] + SLOT[1])

        occupied = remove_drivable(frame, frame_to_occupancy(frame),
                                   robots=[robot_xy], goals=[DESTINATION, START],
                                   payload=payload)
        self.get_logger().info("occupancy grid: %.1f%% obstacle" % (100.0 * occupied.mean()))

        # the grid without the crate (it leaves its place when carried)
        carried = occupied.copy()
        mark_square(carried, payload, CRATE_HALF + 0.05, False)   # + the sides the camera sees

        # 1. approach: the crate is an obstacle with a thin margin, so its left side is reachable
        normal = planning_grid(carried) | crate_blocked(payload, SAFETY_MARGIN)
        tight = planning_grid(carried, APPROACH_MARGIN) | crate_blocked(payload, TIGHT_MARGIN)
        route = plan_route(tight, robot_xy, dock, soft=normal)

        # 2. carry: the crate moves as one shape with the robot
        carry = plan_carry(carry_grid(carried), payload)

        # 3. return: the crate now sits on the hexagon's centre
        after = carried.copy()
        mark_square(after, DESTINATION, CRATE_HALF + 0.005, True)
        back = plan_route(planning_grid(after), dock_end, START)

        for name, p in (("approach", route), ("carry", carry), ("return", back)):
            if p is None:
                self.get_logger().error("no %s route (dock %s, payload %s): %s"
                                        % (name, dock, payload, LAST_FAIL))
                self.save_debug(frame, occupied, carried, carry_grid(carried))
                self.planned = False
                return

        zero = Time(seconds=0)                      # carry / return times count from the event
        start = self.get_clock().now() + Duration(seconds=START_DELAY)
        self.plan_pub.publish(self.to_path_msg(route, start))
        self.get_logger().info("published /r2/plan, %d poses, finishes at t=%.1f s"
                               % (len(route), route[-1][0]))
        self.get_logger().info("approach waypoints: %s"
                               % [(round(x, 2), round(y, 2)) for _, x, y in route])
        self.save_debug(frame, occupied, carried, carry_grid(carried))   # always save images
        self.carry_pub.publish(self.to_path_msg(carry, zero))
        self.get_logger().info("published /carry_plan, %d poses, %.1f s" % (len(carry), carry[-1][0]))
        self.return_pub.publish(self.to_path_msg(back, zero))
        self.get_logger().info("published /r2/return_plan, %d poses, %.1f s" % (len(back), back[-1][0]))


def main():
    rclpy.init()
    node = PathPlanner()
    try:
        rclpy.spin(node)                        # keeps the latched paths available
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()