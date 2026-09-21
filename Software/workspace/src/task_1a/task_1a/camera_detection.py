# Copyright (c) 2026 e-Yantra, IIT Bombay. All rights reserved.
# These simulation files and source code are the intellectual property of e-Yantra,
# IIT Bombay, provided solely for eYRC 2026-27 (Theme: Hola The Explorer).
# Sharing or redistribution of this material, in whole or in part, is not permitted.

#!/usr/bin/env python3
'''
*****************************************************************************************
*
*        =============================================
*           Hola The Explorer (HE) Theme (eYRC 2025-26)
*        =============================================
*
*  This script is to implement Task 1A of Hola The Explorer (HE) Theme (eYRC 2025-26).
*
*****************************************************************************************
'''

# Team ID:          [ Team-ID ]
# Author List:      [ Names of team members who worked on this file, separated by comma ]
# Filename:         camera_detection.py
# Functions:        centre_of_quad(), find_trapezoids(), main()
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
SAND_DISTANCE = 40

# Line-detection parameters. The funnel borders are thin outlines only a few
# pixels wide: a large enough minimum length keeps small icon detail out, and a
# generous maximum gap bridges the break where a safe overlaps a border.
HOUGH_THRESHOLD = 40
HOUGH_MIN_LENGTH = 35
HOUGH_MAX_GAP = 25

MIN_TRAPEZOID_AREA = 1200       # px^2, throws away small enclosed blobs
PARALLEL_TOLERANCE_DEG = 7      # two sides count as parallel within this angle

# Converting and printing at the full frame rate would be unreadable and would
# put ~90 service calls a second on the wire for no benefit.
REPORT_PERIOD_SEC = 0.5

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

    # Corners of the quadrilateral, in pixel coordinates
    c1x = corners[0,0]
    c1y = corners[0,1]

    c2x = corners[1,0]
    c2y = corners[1,1]

    c3x = corners[2,0]
    c3y = corners[2,1]

    c4x = corners[3,0]
    c4y = corners[3,1]

    # Dot products of the corner coordinates, used to compute the area and centroid
    d1 = c1x*c2y - c2x*c1y
    d2 = c2x*c3y - c3x*c2y
    d3 = c3x*c4y - c4x*c3y
    d4 = c4x*c1y - c1x*c4y

    # Area of the quadrilateral, used to compute the centroid
    area = (d1 + d2 + d3 + d4) / 2

    # Centroid coordinates, computed as a weighted average of the corner coordinates
    cx = ((c1x+c2x)*d1 + (c2x+c3x)*d2 + (c3x+c4x)*d3 + (c4x+c1x)*d4 ) / (6*area)
    cy = ((c1y+c2y)*d1 + (c2y+c3y)*d2 + (c3y+c4y)*d3 + (c4y+c1y)*d4 ) / (6*area)

    return cx, cy


def find_trapezoids(frame):
    """
    Purpose:
    ---
    Locate the three station funnels (trapezoids) in one camera frame.

    Suggested pipeline -- you write every step:

      1. CROP to the arena rectangle (ARENA_X0 .. ARENA_Y1). Work on the crop
         from here on and remember to add the offset back at the end.

      2. BUILD A MASK of "everything that is not floor".
         Do NOT use a plain HSV hue range: the three funnels are cyan, green
         and orange, and the cyan one is so pale that any hue band wide enough
         to catch it also catches the sand.
         Instead: convert the crop to the LAB colour space, find the floor's
         own colour (the MOST COMMON value in each of the three channels -- a
         histogram/bincount gives you this), then measure how far every pixel
         is from that colour. Pixels farther than SAND_DISTANCE are features.

      3. CLEAN the mask with a small morphological closing so the thin borders
         are not broken up.

      4. DETECT EDGES (Canny) and then LINE SEGMENTS with the probabilistic
         Hough transform, using HOUGH_THRESHOLD / HOUGH_MIN_LENGTH /
         HOUGH_MAX_GAP. Keep the segments in a list -- you need them again.
         Careful: the Hough call returns None when it finds nothing.

      5. DRAW every detected segment onto a blank scratch image, thick enough
         (about 3 px) that the four sides of one funnel join into a closed
         loop, then close that image morphologically too.
         This scratch image is throw-away geometry used only to FIND the
         funnels -- it is not what you display.

      6. FIND CONTOURS on the scratch image with the RETR_CCOMP retrieval
         mode. A closed funnel shows up as a HOLE, i.e. an INNER contour, so
         keep only contours that have a parent in the hierarchy, and drop the
         ones smaller than MIN_TRAPEZOID_AREA.

      7. APPROXIMATE each surviving contour to a polygon (`cv2.approxPolyDP`,
         epsilon of roughly 3% of the arc length) and keep only the ones that
         come out CONVEX and with exactly FOUR vertices. Circular markers
         enclose nothing straight and the central hexagon has six sides, so
         both drop out right here.

      8. REJECT non-trapezoids. Measure the angle of each of the four sides,
         then count how many of the two OPPOSITE-SIDE pairs are parallel
         within PARALLEL_TOLERANCE_DEG. A trapezoid has exactly ONE such pair.
         The rectangular safe next to every funnel has TWO -- this test is what
         removes it. (Angles wrap: 179 deg and 1 deg are 2 deg apart, not 178.)

      9. SHIFT the accepted corners back into full-frame coordinates by adding
         (ARENA_X0, ARENA_Y0), and get the centre with centre_of_quad().

     10. BUILD THE BINARY IMAGE that gets displayed: a black image the size of
         the FULL frame with only the accepted trapezoid borders drawn white.
         Do not just return the mask from step 2 -- the wall, the hexagon and
         the safes all produce strong lines and none of them belong here.

    OPTIONAL (for better accuracy, attempt it after the above works):
    the corners from step 7 outline the region ENCLOSED by the detected
    segments, so every side sits INSIDE the real border by roughly the width
    the segments were drawn with. You can remove that bias by refitting a line
    (`cv2.fitLine`) through the endpoints of the Hough segments that actually
    belong to each side, then intersecting neighbouring sides to get the true
    corner. Deciding which segments "belong" to a side is the interesting part:
    a segment belongs if it runs nearly parallel to the side, lies within a few
    pixels of it, and lies ALONGSIDE it rather than running past the corner.

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

    ##############  ADD YOUR CODE HERE  ##############

    ##################################################

    return binary, trapezoids


##############################################################
def main():
    """
    Purpose:
    ---
    Open the camera stream, run find_trapezoids() on every frame, display the
    result, and periodically convert each trapezoid centre to world
    coordinates through the `pixel_to_world` service.
    """

    # 1. Initialize ROS 2 and create the node and client
    rclpy.init()
    node = Node("camera_feed")
    client = node.create_client(PixelToWorld, "pixel_to_world")

    node.get_logger().info("waiting for the pixel_to_world service ...")
    
    # Wait up to 10 seconds for the service to become available
    if not client.wait_for_service(timeout_sec=10.0):
        node.get_logger().error(
            "pixel_to_world is not up. Start it first: ros2 run task_1a pixel_to_world_service")
        rclpy.shutdown()
        return

    # 2. Open the video stream
    cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened():
        node.get_logger().error(f"could not open {STREAM_URL}")
        node.get_logger().error(
            "start the simulation first: ros2 launch hb_description task1a.launch.py")
        rclpy.shutdown()
        return

    # 3. Setup variables for rolling FPS calculation and reporting intervals
    stamps = deque(maxlen=FPS_WINDOW)
    fps = 0.0
    last_report = 0.0

    # 4. Main processing loop
    while rclpy.ok():
        ok, frame = cap.read()
        if not ok:
            break

        # Calculate rolling frame rate over the last FPS_WINDOW frames
        stamps.append(time.monotonic())
        if len(stamps) >= 2:
            span = stamps[-1] - stamps[0]
            fps = (len(stamps) - 1) / span if span > 0 else 0.0

        # Process the frame to find trapezoids
        binary, trapezoids = find_trapezoids(frame)

        # Overlay: red outline + yellow centre dot for every detection
        for cx, cy, corners in trapezoids:
            cv2.polylines(frame, [np.round(corners).astype(np.int32)], True, (0, 0, 255), 2)
            cv2.circle(frame, (int(round(cx)), int(round(cy))), 6, (0, 255, 255), -1)

        # Draw FPS and count text on the frame
        cv2.putText(frame, f"{fps:5.1f} FPS", (12, 34), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{len(trapezoids)} trapezoids", (12, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display the windows
        cv2.imshow(WINDOW, frame)
        cv2.imshow(BINARY_WINDOW, binary)

        # 5. Periodically send detected coordinates to the ROS 2 service
        now = time.monotonic()
        if trapezoids and now - last_report >= REPORT_PERIOD_SEC:
            last_report = now
            print(f"\n{len(trapezoids)} trapezoid(s):")

            # Sorted top-to-bottom, then left-to-right, so the printed order is
            # stable from frame to frame
            for cx, cy, _ in sorted(trapezoids, key=lambda t: (t[1], t[0])):

                ##############  YOUR CODE BLOCK  ##############
                
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

    # 6. Cleanup resources
    cap.release()
    cv2.destroyAllWindows()
    node.destroy_node()
    rclpy.shutdown()


##############################################################
if __name__ == "__main__":
    main()
