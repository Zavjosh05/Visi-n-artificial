import cv2
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

        if not (len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = np.zeros_like(img, dtype=np.uint8)
        for mask in kirsch_masks:
            filtered = self.convolucion_manual(img,mask)
            abs_filtered = cv.convertScaleAbs(filtered)
            #abs_filtered = self.normalizar_manual(filtered)
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

        angulo = np.arctan2(grad_y, grad_x) * 180 / np.pi

        magnitude = np.sqrt(np.square(grad_x) + np.square(grad_y))

        magnitude = self.normalizar_manual(magnitude)

        return magnitude, angulo

    def convolucion_manual(self, img, kernel):
        """
        Aplica convolución manualmente sin usar filter2D
        """
        if (len(img.shape) != 2):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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

    def prewit(this, img):

        mascara_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
        mascara_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

        convolution_x = this.convolucion_manual(img, mascara_x)
        convolution_y = this.convolucion_manual(img, mascara_y)

        prewit_img = np.sqrt(np.power(convolution_x, 2) + np.power(convolution_y, 2))
        prewit_img = this.normalizar_manual(prewit_img)

        return prewit_img

    def filtro_gaussiano(this, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        varianza = 7
        tam_kernel = 5

        constante = (1 / (2 * np.pi * varianza))

        kernel = np.zeros((tam_kernel, tam_kernel), dtype=np.float32)

        resultado = np.zeros_like(img, dtype=np.float32)

        for i in range(tam_kernel):
            for j in range(tam_kernel):
                kernel[i][j] = constante * pow(np.e, -(((j - 2) + (i - 2)) / (2 * varianza)))

        resultado = this.convolucion_manual(img, kernel)

        resultado = this.normalizar_manual(resultado)

        return resultado

    def canny(this, img):
        suavizado = this.filtro_gaussiano(img)
        bordes, angulo = this.sobel(suavizado)
        supresion = this.supresion_no_max(bordes, angulo)
        umbralizacion = this.umbralizacion_doble(supresion, [40, 100], [120, 255])
        canny_img = this.seguimiento_histéresis(umbralizacion, [40, 100], [120, 255])

        return canny_img

    def supresion_no_max(this, img, angulo):

        resultado = np.zeros_like(img, dtype=np.uint8)

        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if angulo[i][j] >= 0 and angulo[i][j] < 45:
                    val1 = img[i][j - 1]
                    val2 = img[i][j + 1]
                elif angulo[i][j] >= 45 and angulo[i][j] < 90:
                    val1 = img[i - 1][j - 1]
                    val2 = img[i + 1][j + 1]
                elif angulo[i][j] >= 90 and angulo[i][j] < 135:
                    val1 = img[i - 1][j]
                    val2 = img[i + 1][j]
                elif angulo[i][j] >= 135 and angulo[i][j] <= 180:
                    val1 = img[i - 1][j + 1]
                    val2 = img[i + 1][j - 1]

                if img[i][j] >= val1 and img[i][j] >= val2:
                    resultado[i][j] = img[i][j]

        return resultado

    def umbralizacion_doble(this, img, umbral, valores):

        resultado = np.zeros_like(img, dtype=np.uint8)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if img[i][j] >= umbral[0] and img[i][j] < umbral[1]:
                    resultado[i][j] = valores[0]
                elif img[i][j] >= umbral[1]:
                    resultado[i][j] = valores[1]

        return resultado

    def seguimiento_histéresis(this, img, umbral, valores):

        resultado = np.zeros_like(img, dtype=np.uint8)
        resultado_max = np.where(img >= umbral[1])
        resultad_min = np.where((img >= umbral[0]) & (img < umbral[1]))

        resultado[resultado_max] = valores[1]
        resultado[resultad_min] = valores[0]

        for i in range(1, img.shape[0] - 1):
            for j in range(1, img.shape[1] - 1):
                if resultado[i][j] == valores[0]:
                    if np.any(resultado[i - 1:i + 2, j - 1:j + 2] == valores[1]):
                        resultado[i][j] = valores[1]
                    else:
                        resultado[i][j] = 0

        return resultado
