import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy import stats 

#En esta claser iran los filtros de pasa altas y pasa bajas
class Filtros:
    def __init__(self):
        return

    def filtro_promediador(self, imagen_original, kernel):
        n = int(kernel)
        print("aplicando filtro promediador con kernel n  ="  + kernel)
        if imagen_original is None:
            print("Error en la lectura de imagen filtro promediador")
            return None
        else: 
            print("lectura de imagen exitosa imagen filtro promediador")
        imagen_filtrada = cv2.blur(imagen_original, (n,n))
        return imagen_filtrada      
    
    def filtro_pesado(self, imagen_original, n):
        if imagen_original is None:
            return None
        if n <= 1:
            return None
        kernel = np.array([[1, 1, 1], [1, n, 1], [1, 1, 1]]) / (8+n)
        kernel /= np.sum(kernel)
        imagen_promediador_pesado = cv2.filter2D(imagen_original, -1, kernel)
        return imagen_promediador_pesado
    
    def filtro_mediana(self, imagen_original):
        print("aplicando filtro mediana")
        if imagen_original is None:
            return None
        imagen_mediana = cv2.medianBlur(imagen_original,5)
        return imagen_mediana
    
    def filtro_moda(self, img, kernel_size=3):

        if img is None:
            return None 
        pad = kernel_size//2
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        result = np.zeros_like(img)
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                window = padded[i:i+kernel_size, j:j+kernel_size]
                values, counts = np.unique(window, return_counts=True)
                result[i,j] = values[np.argmax(counts)]
        return result

    def filtro_bilateral(self, imagen_original):
        print("aplicando filtro bilateral")
        if imagen_original is None:
            return None
        image_bilateral = cv2.bilateralFilter(imagen_original, 9, 75, 75)
        return image_bilateral
    
    def filtro_max(self, img):
        print("aplicando filtro Maximo")
        kernel_size = 3
        img_maximo = cv2.dilate(img, np.ones((kernel_size,kernel_size)))
        return img_maximo
    
    def filtro_min(self, img):
        print("aplicando filtro Minimo")
        kernel_size = 3 
        img_minimo = cv2.erode(img, np.ones((kernel_size,kernel_size)))
        return img_minimo
    
    def filtro_gaussiano(self, imagen_original):
        print("aplicando filtro gaussiano")
        if imagen_original is None:
            return None
        imagen_gaussiana = cv2.GaussianBlur(imagen_original, (5,5), 1)
        return imagen_gaussiana