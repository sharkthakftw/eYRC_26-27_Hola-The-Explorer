import cv2
import numpy as np

# ---------------- constants ----------------
# Arena crop rectangle: set these to your arena bounds in the screenshot.
# None for X1/Y1 means "to the edge of the image".
ARENA_X0 = 28
ARENA_Y0 = 42
ARENA_X1 = 935
ARENA_Y1 = 940

SAND_DISTANCE = 20
CLOSE_KSIZE = 3               # step 3

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 40
HOUGH_MIN_LENGTH = 35
HOUGH_MAX_GAP = 25

SCRATCH_THICKNESS = 3
SCRATCH_CLOSE_KSIZE = 5

MIN_TRAPEZOID_AREA = 1500
PARALLEL_TOLERANCE_DEG = 6


# ---------------- helpers ----------------

def centre_of_quad(corners):
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

def make_mask_lab(img):
    """Step 2: 255 where the pixel is NOT floor, 0 where it is."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    floor = np.array(
        [np.bincount(lab[:, :, c].ravel(), minlength=256).argmax()
         for c in range(3)], np.float32)
    dist = np.linalg.norm(lab.astype(np.float32) - floor, axis=2)
    return ((dist > SAND_DISTANCE) * 255).astype(np.uint8)


def close_mask(mask):
    """Step 3: small closing so thin borders aren't broken."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_KSIZE, CLOSE_KSIZE))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)


def detect_segments(mask):
    """Step 4: Canny + probabilistic Hough. Returns (edges, list of segments)."""
    edges = cv2.Canny(mask, CANNY_LOW, CANNY_HIGH)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is None:
        return edges, []
    segments = [tuple(int(v) for v in row) for row in lines.reshape(-1, 4)]
    return edges, segments


def build_scratch(shape, segments):
    """Step 5: throw-away geometry image used only to find the funnels."""
    h, w = shape[:2]
    scratch = np.zeros((h, w), np.uint8)
    for x1, y1, x2, y2 in segments:
        cv2.line(scratch, (x1, y1), (x2, y2), 255, SCRATCH_THICKNESS)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (SCRATCH_CLOSE_KSIZE, SCRATCH_CLOSE_KSIZE))
    return cv2.morphologyEx(scratch, cv2.MORPH_CLOSE, kernel)


def find_quads(scratch):
    """Steps 6-7: inner contours -> convex polygons with exactly 4 vertices."""
    contours, hierarchy = cv2.findContours(
        scratch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]

    quads = []
    for i, cnt in enumerate(contours):
        if hierarchy[i][3] == -1:                 # no parent -> outer contour
            continue
        if cv2.contourArea(cnt) < MIN_TRAPEZOID_AREA:
            continue
        eps = 0.03 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            quads.append(approx.reshape(4, 2))
    return quads


def angle_diff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)          # 179 vs 1 -> 2, not 178


def is_trapezoid(pts):
    """Step 8: exactly one pair of parallel opposite sides."""
    ang = []
    for j in range(4):
        dx, dy = pts[(j + 1) % 4] - pts[j]
        ang.append(np.degrees(np.arctan2(dy, dx)) % 180.0)
    n = sum(angle_diff(ang[a], ang[b]) < PARALLEL_TOLERANCE_DEG
            for a, b in ((0, 2), (1, 3)))
    return n == 1


# ---------------- main function ----------------
def find_trapezoids(frame, debug=True):
    binary = np.zeros(frame.shape[:2], np.uint8)
    trapezoids = []

    # 1. crop to arena
    x1 = ARENA_X1 if ARENA_X1 is not None else frame.shape[1]
    y1 = ARENA_Y1 if ARENA_Y1 is not None else frame.shape[0]
    crop = frame[ARENA_Y0:y1, ARENA_X0:x1]

    # 2-3. mask + closing
    mask = close_mask(make_mask_lab(crop))

    # 4. edges + segments
    edges, segments = detect_segments(mask)

    # 5. scratch image
    scratch = build_scratch(crop.shape, segments)

    # 6-8. contours -> quads -> trapezoids
    quads = find_quads(scratch)
    accepted = [q for q in quads if is_trapezoid(q)]

    # 9-10. shift to full-frame coords, centres, draw binary
    offset = np.array([ARENA_X0, ARENA_Y0])
    for q in accepted:
        corners = q + offset
        cx, cy = centre_of_quad(corners)
        trapezoids.append((cx, cy, corners))
        cv2.polylines(binary, [corners.astype(np.int32).reshape(-1, 1, 2)],
                      True, 255, 2)

    if debug:
        cv2.imshow("EDGES", edges)
        vis = frame.copy()
        for q in quads:        # red: every 4-sided convex shape
            cv2.polylines(vis, [(q + offset).reshape(-1, 1, 2)],
                          True, (0, 0, 255), 2)
        for _, _, c in trapezoids:   # green: accepted trapezoids
            cv2.polylines(vis, [c.astype(np.int32).reshape(-1, 1, 2)],
                          True, (0, 255, 0), 2)
        print(f"{len(segments)} segments, {len(quads)} quads, "
              f"{len(trapezoids)} trapezoids")
        cv2.imshow("RESULT", vis)

    return binary, trapezoids

if __name__ == "__main__":
    img = cv2.imread("/home/sharkthak/Screenshots/arena.png")
    binary, trapezoids = find_trapezoids(img, debug=False)
    for cx, cy, corners in trapezoids:
        print((cx, cy), corners.tolist())

    # --- snippet goes here ---
    vis = img.copy()
    for cx, cy, corners in trapezoids:
        cv2.polylines(vis, [corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
        cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
    cv2.imshow("CHECK", vis)
    # -------------------------

    print(binary.shape, img.shape[:2])   # should match
    cv2.imshow("BINARY", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()