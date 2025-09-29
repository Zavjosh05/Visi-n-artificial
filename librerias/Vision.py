import numpy as np
import cv2 as cv

class Vision:
    def __init__(self):
        return

    def mascaras_kirsch(self,img):
        kirsch_masks = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # Norte
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # Noreste
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # Este
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # Sureste
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # Sur
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # Suroeste
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # Oeste
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])  # Noroeste
        ]

        edges = np.zeros_like(img, dtype=np.uint8)
        for mask in kirsch_masks:
            filtered = cv.filter2D(img, cv.CV_16S, mask)
            abs_filtered = cv.convertScaleAbs(filtered)
            edges = np.maximum(edges, abs_filtered)

        return img

    def sobel_artesanal(self, img):

        if len(img.shape) == 3:
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        # Kernels Sobel
        sobel_x = np.array([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]])

        sobel_y = np.array([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]])

        img_float = img_gray.astype(np.float32)

        grad_x = self.convolucion_manual(img_float, sobel_x)
        grad_y = self.convolucion_manual(img_float, sobel_y)

        magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

        magnitude = self.normalizar_manual(magnitude)

        magnitude = magnitude.astype(np.uint8)

        return magnitude

    def convolucion_manual(self, img, kernel):
        """
        Aplica convolución manualmente sin usar filter2D
        """
        img_h, img_w = img.shape
        kernel_h, kernel_w = kernel.shape

        pad_h = kernel_h // 2
        pad_w = kernel_w // 2

        img_padded = np.zeros((img_h + 2 * pad_h, img_w + 2 * pad_w), dtype=np.float32)
        img_padded[pad_h:pad_h + img_h, pad_w:pad_w + img_w] = img

        output = np.zeros_like(img, dtype=np.float32)

        for i in range(img_h):
            for j in range(img_w):

                region = img_padded[i:i + kernel_h, j:j + kernel_w]

                output[i, j] = np.sum(region * kernel)

        return output

    def normalizar_manual(self, img):
        """
        Normaliza manualmente una imagen al rango [0, 255]
        """
        img_min = np.min(img)
        img_max = np.max(img)

        if img_max == img_min:
            return np.zeros_like(img)

        img_normalized = 255 * (img - img_min) / (img_max - img_min)

        return img_normalized