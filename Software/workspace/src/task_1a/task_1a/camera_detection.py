#!/usr/bin/env python3

# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

'''
*****************************************************************************************
*
*        =============================================
*          Hola The Explorer (HE) Theme (eYRC 2025-26)
*        =============================================
*
*  This script is to implement Task 1A of Hola The Explorer (HE) Theme (eYRC 2025-26).
*
*****************************************************************************************
'''

# Team ID:          [ eYRC#3142 ]
# Author List:      [ Shourya Gupta, Sarthak Gupta ]
# Filename:         camera_detection.py
# Functions:        centre_of_quad(), make_mask_lab(), close_mask(), build_scratch(), 
#                   angle_diff(), is_trapezoid(), find_quads(), find_trapezoids(), main()
#                   [ Add every extra helper function you write to this list ]
# Global variables: STREAM_URL, WINDOW, BINARY_WINDOW, FPS_WINDOW, ARENA_*, SAND_DISTANCE,
#                   HOUGH_*, MIN_TRAPEZOID_AREA, PARALLEL_TOLERANCE_DEG, REPORT_PERIOD_SEC
#                   [ Add every extra global variable you declare to this list ]
# Service Clients:  pixel_to_world  ->  shape_interface/srv/PixelToWorld


import math
import time
from collections import deque

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from shape_interface.srv import PixelToWorld

###############################################################

################# ADD EXTRA IMPORTS / GLOBALS HERE ############



###############################################################


############################ WHAT YOU HAVE TO DO ##############################
#
#  The overhead camera publishes an MJPEG stream of the arena. Three "station
#  funnels" are painted on the arena floor -- each one is a TRAPEZOID (exactly
#  ONE pair of parallel sides). Your job:
#
#   1. Detect the three trapezoids in every frame.
#   2. Show a second window in which ONLY the trapezoid borders are white and
#      everything else is black.
#   3. Find each trapezoid's centre in pixels, and call the `pixel_to_world`
#      ROS service to convert that pixel to arena coordinates in metres.
#
#  Other things on the floor WILL try to fool you: the arena wall, the central
#  hexagon, circular markers and the rectangular "safe" that sits next to every
#  funnel. Your filters have to reject all of them.
#
#  Run order:
#      ros2 launch hb_description task1a.launch.py     # terminal 1 (simulation)
#      ros2 run task_1a camera_feed                    # terminal 2 (this file)
#
#  The launch file already starts pixel_to_world_service for you. If you launch
#  with enable_pixel_service:=false, start it yourself in its own terminal with
#      ros2 run task_1a pixel_to_world_service
#
###############################################################################


##################### TUNABLE CONSTANTS #######################
## These are STARTING values, not final answers. Keep the    ##
## binary window open and tune them until only the three     ##
## funnel borders survive.                                   ##
###############################################################

STREAM_URL = "http://127.0.0.1:8080/stream"
WINDOW = "camera_feed"
BINARY_WINDOW = "trapezoid_borders"
FPS_WINDOW = 30                 # number of frames averaged for the FPS readout

# Arena floor in image pixels, at the default 1280x720 stream. Everything
# outside this rectangle is terrain -- the SAME sandy rock as the floor, so if
# you do not crop it away it floods your mask. Re-measure these if you change
# the camera or the stream resolution (open one frame in an image viewer and
# read off the corners of the floor).
ARENA_X0, ARENA_Y0, ARENA_X1, ARENA_Y1 = 304, 24, 975, 695

# How far a pixel's colour must be from the floor's OWN colour before you call
# it "a drawn feature". The floor texture is very uniform, so a modest
# threshold separates cleanly. Raise it if noise leaks in, lower it if the pale
# cyan funnel disappears.
#
# Lowered from 40 -> 24: the pale cyan funnel's LAB distance from the floor
# colour is small, and 40 was filtering its border out along with the sand.
# If sand-texture speckle starts showing up in the mask at this value, raise
# it back up in small steps (e.g. 28, 32) rather than jumping straight to 40 --
# the cyan funnel is the first thing to disappear as this climbs.
SAND_DISTANCE = 26

# Line-detection parameters. The funnel borders are thin outlines only a few
# pixels wide: a large enough minimum length keeps small icon detail out, and a
# generous maximum gap bridges the break where a safe overlaps a border.
#
# THRESHOLD and MIN_LENGTH lowered, MAX_GAP raised: the pale funnel's edges
# are weaker and more broken up than the saturated orange/green ones, so the
# defaults were dropping its sides before they ever reached the scratch image.
HOUGH_THRESHOLD = 30
HOUGH_MIN_LENGTH = 25
HOUGH_MAX_GAP = 30

MIN_TRAPEZOID_AREA = 1200       # px^2, throws away small enclosed blobs

# Raised from 7 -> 9 to give a perspective-skewed pale border a bit more
# slack. Do not push this much higher: the rectangular safe next to each
# funnel has TWO parallel pairs, and too generous a tolerance risks merging
# its near-parallel pair readings into a false single-pair match.
PARALLEL_TOLERANCE_DEG = 9

# Converting and printing at the full frame rate would be unreadable and would
# put ~90 service calls a second on the wire for no benefit.
REPORT_PERIOD_SEC = 0.5

# Flip this on while retuning to pop up a live window showing just the LAB
# distance mask (step 2 of the pipeline), before Canny/Hough ever run. It's
# the fastest way to see whether a missing funnel is a colour-threshold
# problem or a downstream geometry problem.
DEBUG_SHOW_MASK = False

###############################################################


##############################################################
def centre_of_quad(corners):
    """
    Purpose:
    ---
    Find the centre of a quadrilateral given its four corners.

    NOTE: the mean of the four corners is NOT the answer. A trapezoid's two
    parallel sides have different lengths, so the corner mean is pulled towards
    the shorter side and sits off the true centre of the shape. You want the
    AREA centroid of the polygon instead -- look up image moments in OpenCV
    (`cv2.moments` works on a contour, and the centroid is m10/m00, m01/m00).
    Remember to handle the degenerate case where m00 is (nearly) zero.

    Input Arguments:
    ---
    `corners` :  [ numpy array of shape (4, 2), dtype float ]
        the four corners of the quadrilateral, in pixel coordinates

    Returns:
    ---
    `cx` :  [ float ]  x coordinate of the centre, in pixels
    `cy` :  [ float ]  y coordinate of the centre, in pixels

    Example call:
    ---
    cx, cy = centre_of_quad(corners)
    """

    cx, cy = 0.0, 0.0

    # Extract coordinates for the 4 corners
    c1x, c1y = corners[0, 0], corners[0, 1]
    c2x, c2y = corners[1, 0], corners[1, 1]
    c3x, c3y = corners[2, 0], corners[2, 1]
    c4x, c4y = corners[3, 0], corners[3, 1]

    # Calculate cross products for area and centroid weighting
    d1 = c1x * c2y - c2x * c1y
    d2 = c2x * c3y - c3x * c2y
    d3 = c3x * c4y - c4x * c3y
    d4 = c4x * c1y - c1x * c4y

    # Calculate standard polygon area
    area = (d1 + d2 + d3 + d4) / 2

    # Degenerate case fallback: If area is practically zero, return corner mean.
    if abs(area) < 1e-6:
        cx = float(np.mean(corners[:, 0]))
        cy = float(np.mean(corners[:, 1]))
        return cx, cy

    # Compute weighted centroid coordinates mathematically
    cx = ((c1x + c2x) * d1 + (c2x + c3x) * d2 + (c3x + c4x) * d3 + (c4x + c1x) * d4) / (6 * area)
    cy = ((c1y + c2y) * d1 + (c2y + c3y) * d2 + (c3y + c4y) * d3 + (c4y + c1y) * d4) / (6 * area)

    return cx, cy


def make_mask_lab(img):
    """Return a binary mask of non-floor pixels in the arena crop using LAB colour distance."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    floor_colour = np.array(
        [np.bincount(lab[:, :, c].ravel(), minlength=256).argmax() for c in range(3)],
        dtype=np.float32,
    )
    dist = np.linalg.norm(lab.astype(np.float32) - floor_colour, axis=2)
    return ((dist > SAND_DISTANCE) * 255).astype(np.uint8)


def close_mask(mask):
    """Close the mask lightly to preserve the thin funnel borders."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def build_scratch(shape, segments):
    """Draw the Hough segments into a scratch image for contour-based quad finding."""
    h, w = shape[:2]
    scratch = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in segments:
        cv2.line(scratch, (x1, y1), (x2, y2), 255, 3)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    return cv2.morphologyEx(scratch, cv2.MORPH_CLOSE, kernel)


def angle_diff(a, b):
    """Smallest angular difference in degrees, handling wraparound."""
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def is_trapezoid(pts):
    """Reject any four-sided shape that does not have exactly one pair of parallel sides."""
    angles = []
    for idx in range(4):
        dx, dy = pts[(idx + 1) % 4] - pts[idx]
        angles.append(np.degrees(np.arctan2(dy, dx)) % 180.0)

    parallel_pairs = 0
    # Check opposite sides for parallelism (side 0 vs 2, and side 1 vs 3)
    for a, b in ((0, 2), (1, 3)):
        if angle_diff(angles[a], angles[b]) < PARALLEL_TOLERANCE_DEG:
            parallel_pairs += 1

    return parallel_pairs == 1


def find_quads(scratch):
    """Find convex four-vertex contours that could be trapezoids."""
    contours, hierarchy = cv2.findContours(scratch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []

    hierarchy = hierarchy[0]
    quads = []
    
    for idx, contour in enumerate(contours):
        # Ignore external contours
        if hierarchy[idx][3] == -1:
            continue
            
        # Ignore shapes that are too small
        if cv2.contourArea(contour) < MIN_TRAPEZOID_AREA:
            continue

        # Approximate the contour to a polygon
        epsilon = 0.03 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Accept if it resolves to exactly 4 points and is convex
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2))

    return quads


def find_trapezoids(frame):
    """
    Purpose:
    ---
    Locate the three station funnels (trapezoids) in one camera frame.

    Pipeline:

     1. CROP to the arena rectangle (ARENA_X0 .. ARENA_Y1). Work on the crop
        from here on and remember to add the offset back at the end.

     2. BUILD A MASK of "everything that is not floor" in LAB colour space:
        find the floor's own colour (the most common value in each channel),
        then flag pixels farther than SAND_DISTANCE from it.

     3. CLEAN the mask with a small morphological closing so the thin borders
        are not broken up.

     4. DETECT EDGES (Canny) and then LINE SEGMENTS with the probabilistic
        Hough transform.

     5. DRAW every detected segment onto a blank scratch image, thick enough
        that the four sides of one funnel join into a closed loop, then close
        that image morphologically too. This scratch image is throw-away
        geometry used only to FIND the funnels -- it is not what you display.

     6. FIND CONTOURS on the scratch image with RETR_CCOMP. A closed funnel
        shows up as a HOLE (inner contour) -- keep only contours with a
        parent in the hierarchy, and drop ones smaller than MIN_TRAPEZOID_AREA.

     7. APPROXIMATE each surviving contour to a polygon and keep only the
        ones that come out CONVEX with exactly FOUR vertices.

     8. REJECT non-trapezoids: keep only quads with exactly ONE pair of
        opposite sides parallel within PARALLEL_TOLERANCE_DEG.

     9. SHIFT the accepted corners back into full-frame coordinates and get
        the centre with centre_of_quad().

     10. BUILD THE BINARY IMAGE that gets displayed: a black image the size
         of the FULL frame with only the accepted trapezoid borders drawn
         white.

    Input Arguments:
    ---
    `frame` :  [ numpy array ]
        one BGR frame straight from the camera stream

    Returns:
    ---
    `binary` :  [ numpy array, uint8, same height/width as `frame` ]
        trapezoid borders in white (255) on black (0), nothing else
    `trapezoids` :  [ list of tuples ]
        one (centre_x, centre_y, corners) per detected trapezoid, where
        `corners` is a (4, 2) array in FULL-FRAME pixel coordinates

    Example call:
    ---
    binary, trapezoids = find_trapezoids(frame)
    """

    binary = np.zeros(frame.shape[:2], np.uint8)
    trapezoids = []

    # 1. Crop to the relevant arena area
    crop = frame[ARENA_Y0:ARENA_Y1, ARENA_X0:ARENA_X1]
    
    # 2 & 3. Generate and clean the colour mask
    raw_mask = make_mask_lab(crop)
    mask = close_mask(raw_mask)

    if DEBUG_SHOW_MASK:
        cv2.imshow("debug_mask", mask)

    # 4. Detect edges and line segments
    edges = cv2.Canny(mask, 50, 150)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )

    segments = []
    if lines is not None:
        segments = [tuple(int(v) for v in row) for row in lines.reshape(-1, 4)]

    # 5, 6 & 7. Build geometry and extract structural quadrilaterals
    scratch = build_scratch(crop.shape, segments)
    quads = find_quads(scratch)
    
    # Calculate offset to translate cropped coordinates back to original frame space
    offset = np.array([ARENA_X0, ARENA_Y0])

    # 8, 9 & 10. Validate shapes and store final coordinates
    for quad in quads:
        if not is_trapezoid(quad):
            continue

        corners = quad + offset
        cx, cy = centre_of_quad(corners)
        trapezoids.append((cx, cy, corners))
        cv2.polylines(binary, [corners.astype(np.int32).reshape(-1, 1, 2)], True, 255, 2)

    return binary, trapezoids


##############################################################
def main():
    """
    Purpose:
    ---
    Open the camera stream, run find_trapezoids() on every frame, display the
    result, and periodically convert each trapezoid centre to world
    coordinates through the `pixel_to_world` service.

    Input Arguments:
    ---
    None

    Returns:
    ---
    None

    Example call:
    ---
    Called automatically by the Python interpreter.
    """

    # Initialize ROS 2 and create the node and client
    rclpy.init()
    node = Node("camera_feed")
    client = node.create_client(PixelToWorld, "pixel_to_world")

    node.get_logger().info("waiting for the pixel_to_world service ...")
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(
            "pixel_to_world is not up. Start it first: ros2 run task_1a pixel_to_world_service")
        rclpy.shutdown()
        return

    # Initialize video capture
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        node.get_logger().error(f"could not open {STREAM_URL}")
        node.get_logger().error(
            "start the simulation first: ros2 launch hb_description task1a.launch.py")
        rclpy.shutdown()
        return

    # Setup variables for rolling FPS calculation and reporting intervals
    stamps = deque(maxlen=FPS_WINDOW)
    fps = 0.0
    last_report = 0.0

    # Main processing loop
    while rclpy.ok():
        ok, frame = cap.read()
        if not ok:
            break

        # Calculate a rolling average for frames-per-second
        stamps.append(time.monotonic())
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            fps = (len(stamps) - 1) / span if span > 0 else 0.0

        # Execute the main vision pipeline
        binary, trapezoids = find_trapezoids(frame)

        # Draw overlays on the main feed (red outline + yellow centre dot)
        for cx, cy, corners in trapezoids:
            cv2.polylines(frame, [np.round(corners).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1)

        # Display HUD information
        cv2.putText(frame, f"{fps:5.1f} FPS", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(trapezoids)} trapezoids", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        cv2.imshow(WINDOW, frame)
        cv2.imshow(BINARY_WINDOW, binary)

        now = time.monotonic()
        if trapezoids and now - last_report >= REPORT_PERIOD_SEC:
            last_report = now
            print(f"\n{len(trapezoids)} trapezoid(s):")

            # sorted top-to-bottom, then left-to-right, so the printed order is
            # stable from frame to frame
            for cx, cy, _ in sorted(trapezoids, key=lambda t: (t[1], t[0])):

                ##############  ADD YOUR CODE HERE  ##############
                #
                # Convert this one pixel centre to world coordinates:
                #
                #   a. Build a PixelToWorld.Request() and fill in its
                #      `pixel_x` and `pixel_y` fields (they are float64 --
                #      cast, or the service call will reject them).
                #   b. Send it with the ASYNCHRONOUS client call, then wait for
                #      the answer with a spin-until-complete helper and a
                #      timeout of about 1 second. Never use the blocking call
                #      here -- it deadlocks when you are already spinning.
                #   c. Read the result. Three cases to print, all of them:
                #        - result is None                 -> the call timed out
                #        - result.success is False        -> print .message
                #        - otherwise                      -> print .world_x and
                #                                            .world_y, in metres
                #
                # Suggested output format:
                #   print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  "
                #         f"world ({wx:6.3f}, {wy:6.3f}) m")
                #
                # The service definition is in
                #   src/shape_interface/srv/PixelToWorld.srv
                # Read it -- it tells you the exact field names.

                # a. Build a PixelToWorld.Request() and fill in fields (cast to float)
                req = PixelToWorld.Request()
                req.pixel_x = float(cx)
                req.pixel_y = float(cy)

                # b. Send it with the ASYNCHRONOUS client call, then wait for the answer
                future = client.call_async(req)

                # Spin until the service responds or 1.0 seconds pass to prevent deadlocks
                rclpy.spin_until_future_complete(node, future, timeout_sec=1.0)

                # c. Read the result and handle the three cases
                result = future.result()

                if result is None:
                    # Case 1: result is None -> the call timed out
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  call timed out")
                elif not result.success:
                    # Case 2: result.success is False -> print .message
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  Failed: {result.message}")
                else:
                    # Case 3: otherwise -> print .world_x and .world_y in metres
                    wx = result.world_x
                    wy = result.world_y
                    print(f"  pixel ({cx:7.2f}, {cy:7.2f})  ->  world ({wx:6.3f}, {wy:6.3f}) m")

                ##################################################

        # Quit if 'q' is pressed
        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


##############################################################
if __name__ == "__main__":
    main()