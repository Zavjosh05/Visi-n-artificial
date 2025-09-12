import cv2
class OperacionesLogicas2:

    def __init__(self, imagen1=None, imagen2=None):
        self.imagen1 = imagen1
        self.imagen2 = imagen2
        self.imagen_and = None
        self.imagen_or = None
        self.imagen_xor = None

    def cargar_imagenes(self, ruta1, ruta2):
        self.imagen1 = cv2.imread(ruta1)
        self.imagen2 = cv2.imread(ruta2)
        if self.imagen1 is not None and self.imagen2 is not None:
            self.imagen1 = cv2.resize(self.imagen1, (300, 300))
            self.imagen2 = cv2.resize(self.imagen2, (300, 300))
        return self.imagen1, self.imagen2
    
    def aplicar_suma(self, img1, img2):
        return cv2.add(img1,img2)
    
    def aplicar_resta(self, img1, img2):
        return cv2.subtract(img1,img2)
    
    def aplicar_multiplicacion(self, img1, img2):
        return cv2.multiply(img1,img2)
    
    def aplicar_and(self, img1, img2):
        return cv2.bitwise_and(img1, img2)
    
    def aplicar_or(self, img1, img2):
        return cv2.bitwise_or(img1,img2)
    
    def aplicar_xor(self, img1, img2):
        return cv2.bitwise_xor(img1,img2)
    
    def aplicar_not(self, img):
        return cv2.bitwise_not(img)

    def aplicar_operaciones_logicas(self):
        if self.imagen1 is None or self.imagen2 is None:
            return None, None, None
        self.imagen_and = cv2.bitwise_and(self.imagen1, self.imagen2)
        self.imagen_or = cv2.bitwise_or(self.imagen1, self.imagen2)
        self.imagen_xor = cv2.bitwise_xor(self.imagen1, self.imagen2)
        return self.imagen_and, self.imagen_or, self.imagen_xor


