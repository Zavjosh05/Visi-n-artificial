import cv2
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import ttk

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
            abs_filtered = self.convert_scale_abs_manual(filtered)
            # abs_filtered = cv.convertScaleAbs(filtered)
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

    def convert_scale_abs_manual(self,img, alpha=1.0, beta=0.0):
        """
        Implementacion artesanal de ScaleAbs pq da cosas diferentes Kirsch con normalizar_manual
        """
        scaled = img.astype(np.float32) * alpha + beta
        abs_scaled = np.abs(scaled)
        abs_scaled = np.clip(abs_scaled, 0, 255)

        return abs_scaled.astype(np.uint8)

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

    #####################################
    ### CHINGOASUMADRETODOACINCOPESOS ###
    #####################################

    def harris(self, img, k=0.04, umbral_rel=0.01, tam_ventana=3):
        """
        Implementación mejorada del detector de esquinas de Harris.
        k: parámetro empírico (entre 0.04 y 0.06)
        umbral_rel: umbral relativo respecto al valor máximo de respuesta
        tam_ventana: tamaño de la ventana gaussiana (3x3 o 5x5)
        """

        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        img_gray = img_gray.astype(np.float32)

        # Calcular gradientes usando sobel existente
        magnitud, angulo = self.sobel(img_gray)

        magnitud = magnitud.astype(np.float32)

        Ix = magnitud * np.cos(angulo * np.pi / 180.0)
        Iy = magnitud * np.sin(angulo * np.pi / 180.0)

        Ixx = Ix ** 2
        Iyy = Iy ** 2
        Ixy = Ix * Iy

        if tam_ventana == 3:
            kernel_gauss = np.array([[1, 2, 1],
                                     [2, 4, 2],
                                     [1, 2, 1]], dtype=np.float32)
        else:
            kernel_gauss = np.array([[1, 4, 6, 4, 1],
                                     [4, 16, 24, 16, 4],
                                     [6, 24, 36, 24, 6],
                                     [4, 16, 24, 16, 4],
                                     [1, 4, 6, 4, 1]], dtype=np.float32)

        kernel_gauss /= np.sum(kernel_gauss)

        Sxx = self.convolucion_manual(Ixx, kernel_gauss)
        Syy = self.convolucion_manual(Iyy, kernel_gauss)
        Sxy = self.convolucion_manual(Ixy, kernel_gauss)

        detM = (Sxx * Syy) - (Sxy ** 2)
        traceM = Sxx + Syy
        R = detM - k * (traceM ** 2)

        R_suprimido = self.supresion_no_maxima_harris(R, tam_vecindario=3)

        R_norm = self.normalizar_manual(R_suprimido)

        umbral = umbral_rel * np.max(R_norm)
        esquinas = np.zeros_like(R_norm, dtype=np.uint8)
        esquinas[R_norm > umbral] = 255

        return esquinas

    def supresion_no_maxima_harris(self, R, tam_vecindario=3):
        """
        Supresión no máxima para refinar la detección de esquinas Harris.
        Conserva solo los máximos locales en una vecindad.
        """
        pad = tam_vecindario // 2
        R_padded = np.pad(R, pad, mode='constant', constant_values=0)
        R_suprimido = np.zeros_like(R)

        for i in range(R.shape[0]):
            for j in range(R.shape[1]):
                # Extraer vecindario
                vecindario = R_padded[i:i + tam_vecindario, j:j + tam_vecindario]
                centro = vecindario[pad, pad]

                if centro == np.max(vecindario) and centro > 0:
                    R_suprimido[i, j] = centro

        return R_suprimido

    def analisisPerimetro(self, img):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            perimetro = cv2.arcLength(cnt, closed=True)
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = cnt[0][0]

            texto = f"{i}"
            print(f"Perimetro de {i}: {perimetro}")
            cv2.putText(output, texto, (cx, cy), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 0, 255), 1)

        return output

    def analisisSuperficie(self, img):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = cnt[0][0]

            texto = f"{i}"
            print(f"Superficie de {i}: {area:.2f}")
            cv2.putText(output, texto, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        return output

    def descriptores(self, img):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            perimetro = cv2.arcLength(cnt, True)

            # Circularidad
            if perimetro == 0:
                circularidad = 0
            else:
                circularidad = (4 * np.pi * area) / (perimetro ** 2)

            # Compactidad
            if area != 0:
                compactidad = (perimetro ** 2) / area
            else:
                compactidad = 0

            # Excentricidad
            if len(cnt) >= 5:
                (x, y), (MA, ma), angle = cv2.fitEllipse(cnt)
                excentricidad = np.sqrt(1 - (MA / ma) ** 2) if ma != 0 else 0
            else:
                excentricidad = 0

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = cnt[0][0]

            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            texto = f"{i}"
            print(f"{i}:\tCircularidad: {circularidad:.1f}, Compactidad: {compactidad:.1f}, Excentricidad: {excentricidad:.2f}")
            cv2.putText(output, texto, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)

        return output

    def descriptoresFourier(self, img, M=10000):
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()

        _, thresh = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            z = np.array([complex(p[0][0], p[0][1]) for p in cnt])

            Z = np.fft.fft(z)

            zReducido = np.zeros_like(Z)
            centro = len(Z) // 2
            inicio = max(0, centro - M // 2)
            fin = min(len(Z), centro + M // 2)
            zReducido[inicio:fin] = Z[inicio:fin]

            zReconstruido = np.fft.ifft(zReducido)
            puntos = np.array([[int(p.real), int(p.imag)] for p in zReconstruido])

            cv2.polylines(output, [puntos], isClosed=True, color=(255, 0, 0), thickness=1)

            perimetro = cv2.arcLength(puntos, closed=True)

            print(f"Contorno {i} reconstruido con valor: {perimetro:.2f} usando {M} descriptores de Fourier.")

            momentos = cv2.moments(cnt)
            if momentos["m00"] != 0:
                cx = int(momentos["m10"] / momentos["m00"])
                cy = int(momentos["m01"] / momentos["m00"])
            else:
                cx, cy = cnt[0][0]

            texto = f"{i}"
            cv2.putText(output, texto, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return output

    def template_match(self, img, template, method=cv.TM_CCOEFF_NORMED):
        """Template Matching con OpenCV - devuelve imagen con coincidencia marcada"""

        # Convertir a escala de grises si es necesario
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY) if len(template.shape) == 3 else template.copy()

        h, w = template_gray.shape

        # Aplicar template matching
        result = cv.matchTemplate(img_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        # Determinar mejor ubicacion segun metodo
        top_left = min_loc if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        # Dibujar rectangulo en imagen original
        output = img.copy()
        cv.rectangle(output, top_left, bottom_right, (0, 0, 255), 2)

        return output

    def template_match_with_location(self, img, template, method=cv.TM_CCOEFF_NORMED):
        """Template Matching que devuelve imagen y coordenadas de coincidencia"""

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY) if len(template.shape) == 3 else template.copy()

        h, w = template_gray.shape

        result = cv.matchTemplate(img_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        top_left = min_loc if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

        output = img.copy()
        cv.rectangle(output, top_left, bottom_right, (0, 255, 0), 2)

        # Devolver imagen y coordenadas (x, y, w, h)
        return output, (top_left[0], top_left[1], w, h)

    def harris_corners(self, img, block_size=2, ksize=3, k=0.04, threshold=0.01):
        """Detector de esquinas Harris usando OpenCV"""

        # Convertir a escala de grises
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        gray_float = np.float32(gray)

        # Aplicar Harris
        harris_response = cv.cornerHarris(gray_float, block_size, ksize, k)
        harris_dilated = cv.dilate(harris_response, None)

        # Marcar esquinas en imagen original
        output = img.copy()
        output[harris_dilated > threshold * harris_dilated.max()] = [0, 0, 255]

        return output

    def morphological_skeleton(self, img):

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, copia = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)

        # Crear ventana
        rot = tk.Tk()
        rot.title("Seleccionar forma")
        rot.geometry("400x400")
        rot.configure(bg="#2B2B2B")

        opciones = ["Diamante", "Cruz", "Recto", "Elipse"]

        combo = ttk.Combobox(rot, values=opciones, state="readonly")
        combo.set("Elige una forma")
        combo.pack(pady=20)

        resultado = {"morph": None}

        def seleccionar_opcion():
            opcion = combo.get()
            print("OPCION:", opcion)

            if opcion == "Diamante":
                resultado["morph"] = cv2.MORPH_DIAMOND
            elif opcion == "Cruz":
                resultado["morph"] = cv2.MORPH_CROSS
            elif opcion == "Recto":
                resultado["morph"] = cv2.MORPH_RECT
            elif opcion == "Elipse":
                resultado["morph"] = cv2.MORPH_ELLIPSE

            rot.destroy()

        tk.Button(
            rot,
            text="Aceptar",
            bg="#1F76C2",
            fg="white",
            activebackground="#226EB5",
            relief="flat",
            width=12,
            pady=5,
            bd=0,
            command=seleccionar_opcion
        ).pack(pady=20)

        rot.wait_window()

        morph = resultado["morph"]


        skeleton = np.zeros_like(copia)

        kernel = cv2.getStructuringElement(morph, (3, 3))

        while True:

            erosion = cv2.erode(copia, kernel)
            apertura = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
            contorno = cv2.subtract(erosion, apertura)

            skeleton = cv2.bitwise_or(skeleton, contorno)

            copia = erosion.copy()

            if cv2.countNonZero(copia) == 0:
                break

        return skeleton

    def segmentacion_watershed(self, img):
        imagen = img.copy()
        if len(imagen.shape) == 3:
            gray = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        else:
            gray = imagen.copy()

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        sure_bg = cv2.dilate(opening, kernel, iterations=3)

        dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)

        unknown = cv2.subtract(sure_bg, sure_fg)

        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        markers = cv2.watershed(imagen, markers)
        imagen[markers == -1] = [0, 0, 255]

        return imagen

    def segmentacion_k_means(self, img):
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)

        K = 2

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        centers = np.uint8(centers)
        res = centers[labels.flatten()]
        segmented = res.reshape(img.shape)

        return segmented

    def segmentacion_otsu(self, img):
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap = cv2.convertScaleAbs(lap)

        return lap