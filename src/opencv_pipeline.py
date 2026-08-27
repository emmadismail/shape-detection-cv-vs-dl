import cv2
import numpy as np


def find_table_mask(gray_img):

    h, w = gray_img.shape
    _, bright_mask = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_CLOSE, kernel)
    bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    bottom_threshold = int(0.90 * h)
    touching_bottom = [c for c in contours if cv2.boundingRect(c)[1] + cv2.boundingRect(c)[3] >= bottom_threshold]
    if not touching_bottom:
        return None

    table_contour = max(touching_bottom, key=cv2.contourArea)
    hull = cv2.convexHull(table_contour)

    table_mask = np.zeros_like(gray_img)
    cv2.drawContours(table_mask, [hull], -1, 255, -1)

    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    table_mask = cv2.erode(table_mask, erode_kernel)

    top_limit = int(0.5 * h)
    table_mask[:top_limit, :] = 0

    return table_mask


def detect_and_classify(image_path, debug=False, debug_prefix="outputs/debug"):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    scale = 900 / img.shape[1]
    img = cv2.resize(img, (900, int(img.shape[0] * scale)))

    top_crop = int(0.16 * img.shape[0])
    left_crop = int(0.17 * img.shape[1])
    img = img[top_crop:, left_crop:]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    table_mask = find_table_mask(gray)
    if table_mask is None:
        raise RuntimeError(f"Could not find table region in {image_path}")

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 40, 120)
    edges = cv2.bitwise_and(edges, edges, mask=table_mask)

    boundary = cv2.Canny(table_mask, 50, 150)
    boundary = cv2.dilate(boundary, np.ones((15, 15), np.uint8))
    edges[boundary > 0] = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []
    output = img.copy()

    for c in contours:
        area = cv2.contourArea(c)
        if area < 800 or area > 0.5 * img.shape[0] * img.shape[1]:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * peri, True)
        vertices = len(approx)

        if vertices == 3:
            shape = "triangle"
        elif vertices == 4:
            shape = "rectangle"
        else:
            shape = "other"

        x, y, w, h = cv2.boundingRect(c)
        detections.append({"shape": shape, "area": area, "bbox": (x, y, w, h)})
        cv2.drawContours(output, [c], -1, (0, 0, 255), 2)
        cv2.putText(output, shape, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)

    if debug:
        cv2.imwrite(f"{debug_prefix}_gray.jpg", gray)
        cv2.imwrite(f"{debug_prefix}_table_mask.jpg", table_mask)
        cv2.imwrite(f"{debug_prefix}_edges.jpg", edges)
        cv2.imwrite(f"{debug_prefix}_closed.jpg", closed)
        cv2.imwrite(f"{debug_prefix}_output.jpg", output)

    return detections