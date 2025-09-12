import cv2
import numpy as np
from librerias.ProcesadorImagen import *

class AjustesDeBrillo:

    def __init__(self):
        self.procesador = ProcesadorImagen()

    def ecualizacion_hipercubica(self, img):
        
        img_gray = self.procesador.convertir_a_grises(img)

        g_min = np.min(img)
        g_max = np.max(img)

        histograma, _ = np.histogram(img_gray, bins=256, range=(0, 255))
        probabilidades = histograma / np.sum(histograma)
        suma_acumulada = np.cumsum(probabilidades)

        cubo_min = np.cbrt(g_min)
        cubo_max = np.cbrt(g_max)

        tabla_transformacion = np.array([
            ((cubo_max - cubo_min) * suma_acumulada[g] + cubo_min) ** 3 for g in range(256)
        ])
        tabla_transformacion = np.clip(tabla_transformacion, 0, 255).astype(np.uint8)
        ecualizada_hipercubica = tabla_transformacion[img_gray]
        return ecualizada_hipercubica


    # # 2. Ecualización del histograma estándar
    def ecualizacion_de_histograma(self, img):
        img_grey = self.procesador.convertir_a_grises(img)
        img_eq = cv2.equalizeHist(img_grey)
        #hist_eq = cv2.calcHist([img_eq], [0], None, [256], [0, 256])
        return img_eq

# # 3. Corrección Gamma 
    def correccion_gamma(self, img):
        img_grey = self.procesador.convertir_a_grises(img)
        gamma = 0.3
        img_gamma = np.power(img_grey/255.0, gamma) * 255
        img_gamma = img_gamma.astype('uint8')
        #hist_gamma = cv2.calcHist([img_gamma], [0], None, [256], [0, 256])
        return img_gamma

# # 4. Expansión lineal de contraste
    def expansion_lineal_de_contraste(self, img):
        img_grey = self.procesador.convertir_a_grises(img)
        min_val = np.min(img_grey)
        max_val = np.max(img_grey)
        img_expanded = ((img_grey - min_val) / (max_val - min_val)) * 255
        img_expanded = img_expanded.astype('uint8')
        #hist_expanded = cv2.calcHist([img_expanded], [0], None, [256], [0, 256])
        return img_expanded

# # 5. Transformación Exponencial
    def transformacion_exponencial(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        c = 1.0
        exponent = 0.05
        img_exp = c * (1 - np.exp(-img_gray * exponent))
        img_exp = (img_exp * 255).astype('uint8')
        #hist_exp = cv2.calcHist([img_exp], [0], None, [256], [0, 256])
        return img_exp

# # 6. Ecualización adaptativa (CLAHE)
    def ecualizacion_adaptativa(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        img_clahe = clahe.apply(img_gray)
        #hist_clahe = cv2.calcHist([img_clahe], [0], None, [256], [0, 256])
        return img_clahe

# 7. Transformación Rayleigh (ajuste de escala)
    def transformacion_rayleigh(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        #img_clahe = clahe.apply(img_gray)
        scale = 50  # Ajustar según necesidad
        rayleigh_cdf = 1 - np.exp(-(np.arange(256)**2)/(2*(scale**2)))
        img_rayleigh = rayleigh_cdf[img_gray] * 255
        img_rayleigh = img_rayleigh.astype('uint8')
        hist_rayleigh = cv2.calcHist([img_rayleigh], [0], None, [256], [0, 256])
        return img_rayleigh