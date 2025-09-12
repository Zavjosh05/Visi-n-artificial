import cv2
import numpy as np
from skimage.filters import *
from librerias.ProcesadorImagen import *

class Umbralizacion:

    def __init__(self):
        self.procesador = ProcesadorImagen()
# 1. Umbral por Media
    def umbral_media(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        mean_val = np.mean(img_gray)
        _, img_mean = cv2.threshold(img_gray, mean_val, 255, cv2.THRESH_BINARY)
        return img_mean

# 2. Método de Otsu
    def metodo_otsu(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        otsu_thresh, img_otsu = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(img_otsu, cv2.MORPH_CLOSE, kernel)
        return img_otsu

# 3. Multiumbralización (Otsu multinivel)
    def multiumbralizacion(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        thresholds = threshold_multiotsu(img_gray, classes=3)
        regions = np.digitize(img_gray, bins=thresholds)
        img_multi = np.uint8(regions * (255/2))  # Escalar para visualización
        return img_multi

# 4. Entropía de Kapur (similar a Yen)
    def entropia_kapur(self, img):
        img_gray = self.procesador.convertir_a_grises(img=img)
        kapur_thresh = threshold_yen(img_gray)
        _, img_kapur = cv2.threshold(img_gray, kapur_thresh, 255, cv2.THRESH_BINARY)
        return img_kapur

# 5. Umbral por Banda (rango de intensidades)
    def umbral_por_banda(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        lower = 100
        upper = 200
        img_band = cv2.inRange(img_gray, lower, upper)
        return img_band

# 6. Umbral Adaptativo
    def umbral_adaptativo(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        img_adapt = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        return img_adapt

# 7. Mínimo del Histograma (método de Prewitt modificado)
    def minimo_del_histograma(self, img):
        img_gray = self.procesador.convertir_a_grises(img)
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        smooth_hist = cv2.GaussianBlur(hist, (5,1), 0)
        min_pos = np.argmin(smooth_hist[50:200]) + 50  # Evitar extremos
        _, img_minhist = cv2.threshold(img_gray, min_pos, 255, cv2.THRESH_BINARY)
        return img_minhist

# Función para componentes conexas con vecindad-8 (implementación manual ajskasjka equisde)
    def connected_components_8neighbors(binary_img):
        labeled = np.zeros_like(binary_img, dtype=np.int32)
        current_label = 1
        rows, cols = binary_img.shape
    

    def detectar_objetos_vecindad_8(self, img, min_area=100):
        """
        img: imagen binaria CV_8UC1 (0 o 255), sobre la que queremos detectar y contar objetos.
        min_area: umbral mínimo de píxeles para considerar un objeto "válido".
        
        Retorna:
            - output_binary (CV_8UC1): mapa binario donde cada objeto válido está en 255 (blanco).
            - cnt_validos (int): número de objetos de área >= min_area.
        """
        # Verificamos primero que efectivamente sea binaria (un canal, 0/255)
        if not self.verificar_imagen_binaria(img):
            return None, 0

        # 1) Cerrar los contornos con un MORPH_CLOSE (para unir fragmentos de borde)
        kernel_size = 5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
        # 2) Encontrar contornos sobre la imagen cerrada
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 3) Rellenar todos los contornos en una nueva imagen binaria de 1 canal
        filled = np.zeros_like(img)          # mismo tamaño que img, tipo uint8, UN SOLO CANAL
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
        
        # 4) Calcular componentes conexas sobre 'filled' (CV_8UC1)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(filled, connectivity=8)
        
        # 5) Filtrar por área y generar un nuevo mapa binario solo con regiones >= min_area
        output_binary = np.zeros_like(img)   # imagen final de un canal
        cnt_validos = 0
        for i in range(1, num_labels):       # i=0 es el fondo
            area_i = stats[i, cv2.CC_STAT_AREA]
            if area_i >= min_area:
                # Marcar todos los píxeles de esa etiqueta como 255
                output_binary[labels == i] = 255
                cnt_validos += 1

        return output_binary, cnt_validos


        
    def detectar_objetos_vecindad_4(self,img):
        if self.verificar_imagen_binaria(img):
            numero_objetos, objetos = cv2.connectedComponents(img, connectivity=4)
            imagen_resultado = np.zeros((objetos.shape[0], objetos.shape[1],3),dtype=np.uint8)
            colores = [np.random.randint(0,255,size=3).tolist() for _ in range(numero_objetos)]
            colores[0] = [0,0,0]
            for y in range(objetos.shape[0]):
                for x in range(objetos.shape[1]):
                    imagen_resultado[y,x] = colores[objetos[y,x]]

            return imagen_resultado, numero_objetos
        else:
            return None, 0

    # Direcciones de los 8 vecinos
    def vecindad_8(self,img):
        directions = [(-1,-1), (-1,0), (-1,1),
                    (0,-1),          (0,1),
                    (1,-1),  (1,0), (1,1)]
        
        for i in range(400):
            for j in range(400):
                if img[i,j] == 255 and labeled[i,j] == 0:  # Píxel no etiquetado
                    # BFS para etiquetar componente conexa
                    queue = deque()
                    queue.append((i,j))
                    labeled[i,j] = current_label
                    
                    while queue:
                        x, y = queue.popleft()
                        for dx, dy in directions:
                            nx, ny = x + dx, y + dy
                            if (0 <= nx < rows and 0 <= ny < cols and 
                                binary_img[nx,ny] == 255 and labeled[nx,ny] == 0):
                                labeled[nx,ny] = current_label
                                queue.append((nx,ny))
                    
                    current_label += 1
        return labeled, current_label - 1

# Aplicar componentes conexas
    def componentes_conexas(self, img):
        labeled_img, num_objects = connected_components_8neighbors(img)
        return labeled_img

# Colorear componentes para visualización perrila
    def componentes_visualizacion_perrila(self, img):
        colored_components = np.zeros((*labeled_img.shape, 3), dtype=np.uint8)
        for label in range(1, num_objects + 1):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colored_components[labeled_img == label] = color

    #Funcion que devuelve True si es que la imagen es binaria
    def verificar_imagen_binaria(self,img):
        return  np.all((img == 0) | (img == 255))
    
    def analizar_objetos(self, img_binaria, img_original, min_area=200):
        if len(img_original.shape) == 2 or (len(img_original.shape) == 3 and img_original.shape[2] == 1):
            img_color = cv2.cvtColor(img_original, cv2.COLOR_GRAY2BGR)
        else:
            img_color = img_original.copy()

        contours, _ = cv2.findContours(img_binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lista_objetos = []
        id_actual = 1

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_area:
                perimetro = cv2.arcLength(contour, closed=True)
                x, y, w, h = cv2.boundingRect(contour)

                cv2.rectangle(img_color, (x, y), (x + w, y + h), (0, 255, 0), 2)

                M = cv2.moments(contour)
                cx, cy = (x + w // 2, y + h // 2)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                cv2.putText(img_color, f"ID:{id_actual}", (cx - 30, cy - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                cv2.putText(img_color, f"A:{int(area)}", (cx - 30, cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                cv2.putText(img_color, f"P:{int(perimetro)}", (cx - 30, cy + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                lista_objetos.append({
                    'id': id_actual,
                    'area': area,
                    'perimetro': perimetro,
                    'bbox': (x, y, w, h)
                })

                id_actual += 1

        return img_color, lista_objetos
