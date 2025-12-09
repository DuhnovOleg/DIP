# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 09:38:53 2023

@author: AM4
"""

import sys
sys.path.append('../')
import numpy as np
import cv2 as cv
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import KMeans
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage import data
from scipy import ndimage
import matplotlib.pyplot as plt
import segmentation_utils


image = cv.imread("image.jpg")
image_hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

# Цветовые диапазоны для ежевики в HSV
blackberry_hsv_lower = np.array([110, 40, 10], dtype=np.uint8)
blackberry_hsv_upper = np.array([170, 255, 90], dtype=np.uint8)

blackberry_color_mask = cv.inRange(image_hsv, blackberry_hsv_lower, blackberry_hsv_upper)
# Находим контуры в маске
contours, _ = cv.findContours(blackberry_color_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
seeds = []
for cnt in contours[:5]:
    if cv.contourArea(cnt) > 100:
        M = cv.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            seeds.append((cY, cX))

if not seeds:
    seeds = [(100, 100), (200, 150), (300, 200), (150, 300), (250, 350)]

# координаты для графика
x = list(map(lambda x: x[1], seeds))
y = list(map(lambda x: x[0], seeds))
# порог похожести цвета региона
threshold = 100
# находим сегментацию используя метод из segmentation_utils
segmented_region = segmentation_utils.region_growingHSV(image_hsv, seeds, threshold)
# накладываем маску - отображаем только участки попавшие в какой-либо сегмент
result = cv.bitwise_and(image, image, mask=segmented_region)
# отображаем полученное изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.scatter(x, y, marker="x", color="red", s=200)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение с начальными точками")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
plt.title("Region Growing - выделение ежевики")
plt.show()

# Создаем копию изображения и применяем цветовую маску
image_masked = image.copy()
blackberry_mask = cv.inRange(image_hsv, blackberry_hsv_lower, blackberry_hsv_upper)
image_masked[blackberry_mask == 0] = [0, 0, 0]

qt = segmentation_utils.QTree(stdThreshold = 0.25, minPixelSize = 4, img = image_masked.copy())
qt.subdivide()
tree_image = qt.render_img(thickness=1, color=(0,0,0))

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(cv.cvtColor(image_masked, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(tree_image, cv.COLOR_BGR2RGB))
plt.title("Quadtree - разделение областей")
plt.show()


color_mask = cv.inRange(image_hsv, blackberry_hsv_lower, blackberry_hsv_upper)

# Улучшаем маску морфологическими операциями
kernel = np.ones((5,5), np.uint8)
color_mask = cv.morphologyEx(color_mask, cv.MORPH_CLOSE, kernel, iterations=2)
color_mask = cv.morphologyEx(color_mask, cv.MORPH_OPEN, kernel, iterations=2)

distance_map = ndimage.distance_transform_edt(color_mask)

coords = peak_local_max(distance_map, min_distance=15, labels=color_mask)
local_max = np.zeros_like(distance_map, dtype=bool)
local_max[tuple(coords.T)] = True

markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
labels = watershed(-distance_map, markers, mask=color_mask)

plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(color_mask, cmap="gray")
plt.title("Цветовая маска ежевики")
plt.subplot(1, 3, 2)
plt.imshow(np.uint8(distance_map + 50), cmap="gray")
plt.title("Карта расстояний")
plt.subplot(1, 3, 3)
plt.imshow(np.uint8(labels))
plt.title("Метки Watershed")
plt.show()

mask1 = np.zeros(image.shape[0:2], dtype="uint8")
total_area = 0
result_image = image.copy()
for label in np.unique(labels):
    if label < 2:
        continue
    mask = np.zeros(gray.shape, dtype="uint8")
    mask[labels == label] = 255
    mask1 = mask1 + mask

    cnts = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    c = max(cnts, key=cv.contourArea)
    area = cv.contourArea(c)
    total_area += area
    cv.drawContours(result_image, [c], -1, (36,255,12), 2)

result = cv.bitwise_and(image, image, mask=mask1)

plt.figure(figsize=(15,20))
plt.subplot(1, 2, 1)
plt.imshow(mask1, cmap="gray")
plt.title("Финальная маска ежевики")
plt.subplot(1, 2, 2)
plt.imshow(cv.cvtColor(result_image, cv.COLOR_BGR2RGB))
plt.title("Контуры ежевики")
plt.show()

## Методы кластеризации. K-средних
# Работаем в HSV пространстве для лучшего выделения цвета
hsv_flat = image_hsv.reshape(-1, 3)

hsv_for_clustering = hsv_flat[:, [0, 2]]

# Задаем число кластеров для сегментации
K = 3
# Проводим кластеризацию
kmeans = KMeans(n_clusters=K, random_state=0)
labels = kmeans.fit_predict(hsv_for_clustering)
cluster_centers = kmeans.cluster_centers_

# Выбираем кластер с наименьшим значением Hue
# и не слишком высокой яркостью
hue_values = cluster_centers[:, 0]
value_values = cluster_centers[:, 1]
# Ищем кластер с синим/фиолетовым оттенком (Hue ~ 120-170) и умеренной яркостью
blackberry_cluster = -1
for i in range(K):
    if 100 <= hue_values[i] <= 180 and value_values[i] < 150:
        blackberry_cluster = i
        break

# Если не нашли подходящий кластер, берем кластер с наименьшим Hue
if blackberry_cluster == -1:
    blackberry_cluster = np.argmin(hue_values)

blackberry_mask_kmeans = np.zeros(gray.shape, dtype=np.uint8)
blackberry_mask_kmeans.flat[labels == blackberry_cluster] = 255

# Применяем маску
result_kmeans = cv.bitwise_and(image, image, mask=blackberry_mask_kmeans)

# Отобразим изображения
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.subplot(1, 3, 2)
plt.imshow(blackberry_mask_kmeans, cmap='gray')
plt.title("Маска ежевики (K-means)")
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(result_kmeans, cv.COLOR_BGR2RGB))
plt.title("Выделенная ежевика (K-means)")
plt.show()


## Методы кластеризации. Сдвиг среднего (Mean shift)
# Сглаживаем чтобы уменьшить шум
blur_image = cv.medianBlur(image, 3)
# Выстраиваем пиксели в один ряд и переводим в формат с плавающей точкой
flat_image = np.float32(blur_image.reshape((-1,3)))

# Используем meanshift из библиотеки sklearn
bandwidth = estimate_bandwidth(flat_image, quantile=.06, n_samples=3000)
ms = MeanShift(bandwidth=bandwidth, max_iter=800, bin_seeding=True)
ms.fit(flat_image)
labeled = ms.labels_

# получим количество сегментов
segments = np.unique(labeled)
print('Number of segments: ', segments.shape[0])

# получим средний цвет сегмента
total = np.zeros((segments.shape[0], 3), dtype=float)
count = np.zeros(total.shape, dtype=float)
for i, label in enumerate(labeled):
    total[label] = total[label] + flat_image[i]
    count[label] += 1
avg = total/count
avg = np.uint8(avg)

# Для каждого пикселя проставим средний цвет его сегмента
mean_shift_image = avg[labeled].reshape((image.shape))

# Преобразуем в HSV для выделения ежевики
mean_shift_hsv = cv.cvtColor(mean_shift_image, cv.COLOR_RGB2HSV)

# Создаем маску для ежевики на основе цвета
blackberry_mask_ms = cv.inRange(mean_shift_hsv, blackberry_hsv_lower, blackberry_hsv_upper)

# Улучшаем маску
blackberry_mask_ms = cv.morphologyEx(blackberry_mask_ms, cv.MORPH_CLOSE, kernel, iterations=1)
blackberry_mask_ms = cv.morphologyEx(blackberry_mask_ms, cv.MORPH_OPEN, kernel, iterations=1)

mean_shift_with_mask_image = cv.bitwise_and(image, image, mask=blackberry_mask_ms)

# Построим изображение
plt.figure(figsize=(15,20))
plt.subplot(1, 3, 1)
plt.imshow(cv.cvtColor(image, cv.COLOR_BGR2RGB))
plt.title("Исходное изображение")
plt.subplot(1, 3, 2)
plt.imshow(mean_shift_image, cmap='Set3')
plt.title("Mean Shift сегментация")
plt.subplot(1, 3, 3)
plt.imshow(cv.cvtColor(mean_shift_with_mask_image, cv.COLOR_BGR2RGB))
plt.title("Выделенная ежевика (Mean Shift)")
plt.show()