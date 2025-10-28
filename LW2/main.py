import numpy as np
import cv2 as cv2
import matplotlib.pyplot as plt

def main() -> None:
    image = cv2.imread('./lenna.png')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist = hist.flatten()

    cdf = np.cumsum(hist)
    cdf_normalized = cdf / cdf[-1]
    print(cdf_normalized)
    lut = np.round(255 * cdf_normalized).astype(np.uint8)
    equalized_image = lut[gray_image]

    gs = plt.GridSpec(2, 2)
    plt.figure(figsize=(10, 8))
    plt.subplot(gs[0])
    plt.imshow(gray_image, cmap='gray')
    plt.title('Исходное изображение')

    plt.subplot(gs[1])
    plt.imshow(equalized_image, cmap='gray')
    plt.title('После эквализации')

    plt.subplot(gs[2])
    plt.hist(gray_image.reshape(-1), 256, range=(0, 255))
    plt.title('Гистограмма исходного изображения')
    plt.xlabel('Яркость')
    plt.ylabel('Частота')

    plt.subplot(gs[3])
    plt.hist(equalized_image.reshape(-1), 256, range=(0, 255))
    plt.title('Гистограмма после эквализации')
    plt.xlabel('Яркость')
    plt.ylabel('Частота')

    plt.tight_layout()
    plt.show()
    pass


if __name__ == '__main__':
    main()
