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

        return edges

    def sobel(self, img):

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

        return img_normalized.astype(np.uint8)

    def roberts(self, img):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img = img.copy()

        kx = np.array([[1, 0], [0, -1]])
        ky = np.array([[0, 1], [-1, 0]])

        gx = self.convolucion_manual(img, kx)
        gy = self.convolucion_manual(img, ky)

        G = np.sqrt((gx ** 2) + (gy ** 2))

        roberts = self.normalizar_manual(G)

        return roberts



    def freichen(self, img):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img = img.copy()

        f1 = np.array([[1, np.sqrt(2), 1], [0, 0, 0], [-1, -np.sqrt(2), -1]], dtype=np.float32) / (2 * np.sqrt(2))
        f2 = np.array([[1, 0, -1], [np.sqrt(2), 0, -np.sqrt(2)], [1, 0, -1]], dtype=np.float32) / (2 * np.sqrt(2))
        f3 = np.array([[0, -1, np.sqrt(2)], [1, 0, -1], [np.sqrt(2), 1, 0]], dtype=np.float32) / (2 * np.sqrt(2))
        f4 = np.array([[np.sqrt(2), -1, 0], [-1, 0, 1], [0, 1, -np.sqrt(2)]], dtype=np.float32) / (2 * np.sqrt(2))
        f5 = np.array([[0, 1, 0], [-1, 0, -1], [0, 1, 0]], dtype=np.float32) / 2
        f6 = np.array([[-1, 0, 1], [0, 0, 0], [1, 0, -1]], dtype=np.float32) / 2
        f7 = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32) / 6
        f8 = np.array([[-2, 1, -2], [1, 4, 1], [-2, 1, -2]], dtype=np.float32) / 6
        f9 = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=np.float32) / 3

        mascaras = [f1, f2, f3, f4, f5, f6, f7, f8, f9]

        resultados = [self.convolucion_manual(img,mascara) for mascara in mascaras]

        energias = [resultado ** 2 for resultado in resultados]

        energia = sum(energias)

        bordenergia = sum(energias[:4])

        angulo = np.sqrt((bordenergia) / energia + 0.000001)

        angulo = self.normalizar_manual(angulo)

        return angulo
