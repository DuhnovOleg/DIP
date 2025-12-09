import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage as ndi

image_path = "image.jpg"
blackberry_hsv_lower = np.array([110, 40, 10], dtype=np.uint8)
blackberry_hsv_upper = np.array([170, 255, 90], dtype=np.uint8)
ws_min_distance = 15
min_contour_area = 150
circularity_threshold = 0.3

def create_enhanced_mask(hsv, lo, hi):
    mask = cv.inRange(hsv, lo, hi)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, np.ones((11,11),np.uint8), iterations=2)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, np.ones((7,7),np.uint8), iterations=2)
    cnts, _ = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv.contourArea(c) < 200:
            cv.drawContours(mask, [c], -1, 0, -1)
    return mask

def apply_watershed(binary_mask, min_distance):
    dist = ndi.distance_transform_edt(binary_mask)
    coords = peak_local_max(dist, min_distance=min_distance, labels=binary_mask.astype(bool))
    if coords is None or coords.size == 0:
        coords = peak_local_max(dist, min_distance=max(5, min_distance//2), labels=binary_mask.astype(bool))
    local_max = np.zeros_like(dist, dtype=bool)
    if coords is not None and coords.size > 0:
        local_max[tuple(coords.T)] = True
    markers = ndi.label(local_max)[0]
    labels = watershed(-dist, markers, mask=binary_mask.astype(bool))
    return labels, dist

def filter_watershed_labels(labels, min_area, circ_thresh):
    h,w = labels.shape
    out_mask = np.zeros((h,w), dtype=np.uint8)
    for lab in np.unique(labels):
        if lab == 0: continue
        m = (labels==lab).astype(np.uint8)*255
        cnts, _ = cv.findContours(m, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if not cnts: continue
        c = max(cnts, key=cv.contourArea)
        area = cv.contourArea(c)
        if area < min_area: continue
        per = cv.arcLength(c, True)
        circ = 4*np.pi*area/(per*per) if per>0 else 0
        if circ >= circ_thresh:
            cv.drawContours(out_mask, [c], -1, 255, -1)
    return out_mask


if __name__ == "__main__":
    img = cv.imread(image_path)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    color_mask = create_enhanced_mask(hsv, blackberry_hsv_lower, blackberry_hsv_upper)

    labels, dist_map = apply_watershed(color_mask, ws_min_distance)

    ws_mask = filter_watershed_labels(labels, min_contour_area, circularity_threshold)
    ws_mask = cv.morphologyEx(ws_mask, cv.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ws_mask = cv.morphologyEx(ws_mask, cv.MORPH_OPEN, np.ones((5, 5), np.uint8))

    ws_result = cv.bitwise_and(img, img, mask=ws_mask)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))
    axs[0].imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
    axs[0].set_title("Original")
    axs[0].axis('off')
    axs[1].imshow(color_mask, cmap='gray')
    axs[1].set_title("Color Mask (HSV)")
    axs[1].axis('off')
    axs[2].imshow(cv.cvtColor(ws_result, cv.COLOR_BGR2RGB))
    axs[2].set_title("Watershed Result")
    axs[2].axis('off')
    plt.tight_layout()
    plt.savefig("result_plot.png", dpi=300)
    plt.show()

