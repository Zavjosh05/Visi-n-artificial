import matplotlib.pyplot as plt
import numpy as np
import cv2

class ProcesadorImagen:
    def __init__(self, ruta=None):
        self.imagen_original = None
        if ruta:
            self.cargar_imagen(ruta)
        self.imagen_grises = None
        self.imagen_umbral = None
        self.ecualizada_hipercubica = None
        self.imagen_suma = None
        self.imagen_resta = None
        self.imagen_multiplicacion = None
    
    def cargar_imagen(self, ruta):
        self.imagen_original = cv2.imread(ruta)
        if self.imagen_original is not None:
            self.imagen_original = cv2.resize(self.imagen_original, (400, 400))
        return self.imagen_original

    def convertir_a_grises(self, img):
        if img is None:
            return None
        
        if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
            return img
        else:
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    def aplicar_binarizacion(self, img, umbral):
        _, imagen_binarizada = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
        return imagen_binarizada

    def calcular_histogramas(self, img):
        imagen_en_gris = self.convertir_a_grises(img)
        
        # Histograma en escala de grises
        fig_gray = plt.figure(figsize=(4, 3))
        plt.hist(imagen_en_gris.ravel(), 256, [0, 256])
        plt.title('Histograma en Escala de Grises')

        # Histograma por canales de color
        fig_color = plt.figure(figsize=(4, 3))
        colores = ('b', 'g', 'r')
        for i, canal in enumerate(colores):
            histograma = cv2.calcHist([img], [i], None, [256], [0, 256])
            plt.plot(histograma, color=canal)
        plt.title('Histograma de la Imagen en Color')
        plt.xlim([0, 256])
        
        return fig_gray, fig_color
    
    def calcular_histograma_gris(self, img):
        imagen_en_gris = self.convertir_a_grises(img)
        
        # Histograma en escala de grises
        fig_gray = plt.figure(figsize=(4, 3))
        plt.hist(imagen_en_gris.ravel(), 256, [0, 256])
        plt.title('Histograma en Escala de Grises')

        return fig_gray