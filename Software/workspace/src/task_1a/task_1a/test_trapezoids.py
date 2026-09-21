import cv2
import numpy as np

CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 40
HOUGH_MIN_LENGTH = 35
HOUGH_MAX_GAP = 25
SCRATCH_THICKNESS = 3
SCRATCH_CLOSE_KSIZE = 5

def find_dominant_color(img, k=5, max_side=150):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)

    pixels = img.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    counts = np.bincount(labels.flatten())
    return centers[np.argmax(counts)]  # float array, (R, G, B)

def make_mask(img, dominant_rgb, tol):
    dominant_bgr = dominant_rgb[::-1]

    diff = img.astype(np.float32) - dominant_bgr.astype(np.float32)
    dist = np.linalg.norm(diff, axis=2)

    mask = (dist > tol).astype(np.uint8)
    return mask

# def clean_mask(mask, ksize, iterations):
#     kernel = np.ones((ksize, ksize), np.uint8)

#     # Step 1: opening (erode -> dilate) removes small white specks (noise)
#     opened = cv2.erode(mask, kernel, iterations=iterations)
#     opened = cv2.dilate(opened, kernel, iterations=iterations)

#     # Step 2: closing (dilate -> erode) fills small black holes inside shapes
#     closed = cv2.dilate(opened, kernel, iterations=iterations)
#     closed = cv2.erode(closed, kernel, iterations=iterations)

#     return closed

def detect_segments(mask):
    edges = cv2.Canny(mask * 255, CANNY_LOW, CANNY_HIGH)

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
    """Throw-away geometry image used only to find the funnels."""
    h, w = shape[:2]
    scratch = np.zeros((h, w), np.uint8)   # blank single-channel canvas

    for x1, y1, x2, y2 in segments:
        cv2.line(scratch, (x1, y1), (x2, y2), 255, SCRATCH_THICKNESS)

    # Close small gaps so each funnel's outline becomes a closed loop
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (SCRATCH_CLOSE_KSIZE, SCRATCH_CLOSE_KSIZE))
    scratch = cv2.morphologyEx(scratch, cv2.MORPH_CLOSE, kernel)
    return scratch

img = cv2.imread("/home/sharkthak/Screenshots/arena.png")

dominant = find_dominant_color(img)
mask = make_mask(img, dominant, 40)
# mask = clean_mask(mask, 3, 1)
edges, segments = detect_segments(mask)
scratch = build_scratch(img.shape, segments)

cv2.imshow("EDGES", edges)
cv2.imshow("SCRATCH", scratch)
cv2.waitKey(0)
cv2.destroyAllWindows()