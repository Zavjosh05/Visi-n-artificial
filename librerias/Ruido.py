import numpy as np
import cv2

class Ruido:
    def __init__(self, imagen=None):
        self.imagen_original = imagen

    def cargar_imagen(self, ruta):
        self.imagen_original = cv2.imread(ruta)
        if self.imagen_original is not None:
            self.imagen_original = cv2.resize(self.imagen_original, (400, 400))
        return self.imagen_original

    def agregar_ruido_sal_pimienta(self, img, cantidad=0.05):
        if img is None:
            return None
        salida = np.copy(img)
        num_pixeles = int(cantidad * salida.shape[0] * salida.shape[1])
        # Añadir ruido sal
        coords_x = np.random.randint(0, salida.shape[0], num_pixeles)
        coords_y = np.random.randint(0, salida.shape[1], num_pixeles)
        salida[coords_x, coords_y] = 255
        # Añadir ruido pimienta
        coords_x = np.random.randint(0, salida.shape[0], num_pixeles)
        coords_y = np.random.randint(0, salida.shape[1], num_pixeles)
        salida[coords_x, coords_y] = 0
        return salida

    def agregar_ruido_gaussiano(self, img, media=0, sigma=25):
        if img is None:
            return None
        gauss = np.random.normal(media, sigma, img.shape).astype(np.uint8)
        salida = cv2.add(img, gauss)
        return salida