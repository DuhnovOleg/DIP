import cv2
import numpy as np
import matplotlib.pyplot as plt

img  = cv2.imread('image.jpg')
template = cv2.imread('template.jpg')
template = cv2.flip(template, 1)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
templ_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()

kp1, des1 = sift.detectAndCompute(img_gray, None)
kp2, des2 = sift.detectAndCompute(templ_gray, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m, n in matches:
    if m.distance < 0.6 * n.distance:
        good.append(m)

print("Количество хороших совпадений:", len(good))

src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)

h, w = templ_gray.shape

corners = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
]).reshape(-1, 1, 2)

projected = cv2.perspectiveTransform(corners, H)
proj_coords = projected.reshape(4, 2)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))

axes[0].imshow(cv2.cvtColor(template, cv2.COLOR_BGR2RGB))
axes[0].set_title("Шаблон")
axes[0].axis("off")

img_res = img.copy()
cv2.polylines(img_res, [np.int32(projected)], True, (0, 255, 0), 5)
axes[1].imshow(cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB))
axes[1].set_title("Найденный объект")
axes[1].axis("off")

plt.tight_layout()
plt.savefig("result.jpg")
plt.show()
