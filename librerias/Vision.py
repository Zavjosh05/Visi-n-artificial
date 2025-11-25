import cv2
import numpy as np
import cv2 as cv
import tkinter as tk
import pandas
import sklearn
from tkinter import ttk
import os
import re
from joblib import load, dump
from sklearn.neighbors import KNeighborsClassifier


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
        """
        Calcula TODOS los descriptores (región + perímetro).
        """

        import customtkinter as ctk
        from tkinter import filedialog
        import pandas as pd


        # Copia limpia donde se dibujarán solo contornos/cajas
        imagen_out = img.copy()
        if len(imagen_out.shape) == 2:
            imagen_out = cv2.cvtColor(imagen_out, cv2.COLOR_GRAY2BGR)

        # Imagen gris
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Umbral
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        lista = []  # Tabla de valores

        for i, cnt in enumerate(contours):

            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            perimetro = cv2.arcLength(cnt, True)
            circularidad = (4 * np.pi * area) / (perimetro ** 2 + 1e-6)
            compactidad = (perimetro ** 2) / (area + 1e-6)

            # Excentricidad
            if len(cnt) >= 5:
                (x, y), (a1, a2), angle = cv2.fitEllipse(cnt)
                mayor = max(a1, a2)
                menor = min(a1, a2)
                excentricidad = np.sqrt(1 - (menor / mayor) ** 2)
            else:
                excentricidad = 0

            # Bounding box
            bx, by, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / (h + 1e-6)

            # ---------- DIBUJAR DESCRIPTORES EN LA IMAGEN ----------
            # Contorno
            cv2.drawContours(imagen_out, [cnt], -1, (0, 255, 0), 2)

            # Bounding box
            cv2.rectangle(imagen_out, (bx, by), (bx + w, by + h), (0, 0, 255), 2)

            # ---------- GUARDAR PARA TABLA ----------
            lista.append([
                i, area, perimetro, circularidad, compactidad,
                excentricidad, w, h, aspect_ratio
            ])

        # TABLA
        tabla = ctk.CTkToplevel()
        tabla.title("Tabla de Descriptores")
        tabla.geometry("1100x500")
        tabla.grab_set()

        df = pd.DataFrame(lista, columns=[
            "ID", "Área", "Perímetro", "Circularidad",
            "Compactidad", "Excentricidad", "BBox_W",
            "BBox_H", "Aspect_Ratio"
        ])

        # Búsqueda
        buscar_frame = ctk.CTkFrame(tabla)
        buscar_frame.pack(pady=10)

        lbl_buscar = ctk.CTkLabel(buscar_frame, text="🔍 Buscar:")
        lbl_buscar.grid(row=0, column=0, padx=5)

        entrada_buscar = ctk.CTkEntry(buscar_frame, width=250)
        entrada_buscar.grid(row=0, column=1, padx=5)

        scroll = ctk.CTkScrollableFrame(tabla, width=1050, height=350)
        scroll.pack(pady=10)

        sort_reverse = {col: False for col in df.columns}
        filas_widgets = []

        def dibujar_tabla(data):
            for w in scroll.winfo_children():
                w.destroy()
            filas_widgets.clear()

            # Encabezados clickeables
            for col_idx, col in enumerate(data.columns):
                btn_col = ctk.CTkButton(
                    scroll, text=col, width=100,
                    command=lambda c=col: ordenar_por(c)
                )
                btn_col.grid(row=0, column=col_idx, padx=5, pady=5)

            # Filas
            for r, fila in data.iterrows():
                fila_labels = []
                for c, valor in enumerate(fila):
                    txt = f"{valor:.4f}" if isinstance(valor, float) else str(valor)
                    lbl = ctk.CTkLabel(scroll, text=txt)
                    lbl.grid(row=r + 1, column=c, padx=5, pady=3)
                    fila_labels.append(lbl)
                filas_widgets.append((fila, fila_labels))

        def ordenar_por(col):
            sort_reverse[col] = not sort_reverse[col]
            df_ordenado = df.sort_values(col, ascending=not sort_reverse[col])
            dibujar_tabla(df_ordenado)

        def filtrar(_=None):
            texto = entrada_buscar.get().lower()
            df_filtrado = df[df.apply(lambda row: texto in row.to_string().lower(), axis=1)]
            dibujar_tabla(df_filtrado)

        entrada_buscar.bind("<KeyRelease>", filtrar)

        def exportar_csv():
            ruta = filedialog.asksaveasfilename(
                title="Guardar CSV",
                defaultextension=".csv",
                filetypes=[("CSV", "*.csv")]
            )
            if ruta:
                df.to_csv(ruta, index=False)

        btn_csv = ctk.CTkButton(tabla, text="💾 Exportar CSV", width=200, command=exportar_csv)
        btn_csv.pack(pady=10)

        # Dibujar tabla inicial
        dibujar_tabla(df)

        return imagen_out

    def descriptores_region(self, img):
        """
        Calcula descriptores de REGIÓN (área, circularidad, compactidad, excentricidad).
        """

        # A gris
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Otsu
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            perimetro = cv2.arcLength(cnt, True)

            circularidad = (4 * np.pi * area) / (perimetro ** 2 + 1e-6)
            compactidad = (perimetro ** 2) / (area + 1e-6)

            # Excentricidad corregida
            if len(cnt) >= 5:
                (x, y), (a1, a2), angle = cv2.fitEllipse(cnt)
                mayor = max(a1, a2)
                menor = min(a1, a2)
                excentricidad = np.sqrt(1 - (menor / mayor) ** 2)
            else:
                excentricidad = 0

            # Centroide
            M = cv2.moments(cnt)
            cx = int(M["m10"] / (M["m00"] + 1e-6))
            cy = int(M["m01"] / (M["m00"] + 1e-6))



            print(
                f"[REGION {i}] Area={area:.2f}  Circ={circularidad:.3f}  "
                f"Comp={compactidad:.3f}  Exc={excentricidad:.3f}"
            )

        return output

    def descriptores_perimetro(self, img):
        """
        Calcula descriptores de PERÍMETRO (perímetro, bbox, aspect ratio).
        """

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

        for i, cnt in enumerate(contours):

            area = cv2.contourArea(cnt)
            if area < 50:
                continue

            perimetro = cv2.arcLength(cnt, True)
            x, y, w, h = cv2.boundingRect(cnt)

            # Centroide
            M = cv2.moments(cnt)
            cx = int(M["m10"] / (M["m00"] + 1e-6))
            cy = int(M["m01"] / (M["m00"] + 1e-6))



            print(
                f"[PERIM {i}] Perímetro={perimetro:.2f}  "
                f"Ancho={w}  Alto={h}  Aspect={w / (h + 1e-6):.3f}"
            )

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

            #cv2.polylines(output, [puntos], isClosed=True, color=(255, 0, 0), thickness=1)

            perimetro = cv2.arcLength(puntos, closed=True)

            print(f"Contorno {i} reconstruido con valor: {perimetro:.2f} usando {M} descriptores de Fourier.")

            momentos = cv2.moments(cnt)
            if momentos["m00"] != 0:
                cx = int(momentos["m10"] / momentos["m00"])
                cy = int(momentos["m01"] / momentos["m00"])
            else:
                cx, cy = cnt[0][0]

            texto = f"{i}"
            #cv2.putText(output, texto, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        return output

    def template_match(self, img, template, method=cv.TM_CCOEFF_NORMED):
        """Template Matching con OpenCVdevuelve imagen con coincidencia marcada"""

        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        template_gray = cv.cvtColor(template, cv.COLOR_BGR2GRAY) if len(template.shape) == 3 else template.copy()

        h, w = template_gray.shape

        # Aplicar template matching
        result = cv.matchTemplate(img_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)

        top_left = min_loc if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED] else max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)

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

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        gray_float = np.float32(gray)

        # Aplicar Harris
        harris_response = cv.cornerHarris(gray_float, block_size, ksize, k)
        harris_dilated = cv.dilate(harris_response, None)

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

    def erosion_manual(self, img, kernel=None):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

        if kernel is None:
            kernel = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=np.uint8)

        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        img_padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
        output = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = img_padded[i:i + kh, j:j + kw]
                if np.array_equal(region[kernel == 1], np.full(np.sum(kernel == 1), 255)):
                    output[i, j] = 255

        return output

    def dilatacion_manual(self, img, kernel=None):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)

        if kernel is None:
            kernel = np.array([[1, 1, 1],
                               [1, 1, 1],
                               [1, 1, 1]], dtype=np.uint8)

        kh, kw = kernel.shape
        ph, pw = kh // 2, kw // 2

        img_padded = np.pad(img, ((ph, ph), (pw, pw)), mode="constant", constant_values=0)
        output = np.zeros_like(img)

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                region = img_padded[i:i + kh, j:j + kw]
                if np.any(region[kernel == 1] == 255):
                    output[i, j] = 255

        return output

    def apertura_manual(self, img, kernel=None):
        erosion = self.erosion_manual(img, kernel)
        apertura = self.dilatacion_manual(erosion, kernel)
        return apertura

    def cierre_manual(self, img, kernel=None):
        dilatada = self.dilatacion_manual(img, kernel)
        cierre = self.erosion_manual(dilatada, kernel)
        return cierre

    def gradiente_morfologico(self, img, kernel=None):
        dil = self.dilatacion_manual(img, kernel)
        ero = self.erosion_manual(img, kernel)
        grad = cv.subtract(dil, ero)
        return grad

    def tophat_manual(self, img, kernel=None):
        apertura = self.apertura_manual(img, kernel)
        tophat = cv.subtract(img, apertura)
        return tophat

    def blackhat_manual(self, img, kernel=None):
        cierre = self.cierre_manual(img, kernel)
        blackhat = cv.subtract(cierre, img)
        return blackhat

    def IOU(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        return interArea / float(boxAArea + boxBArea - interArea + 1e-5)

    def template_matching_manual(self, img, template, threshold=0.80, iou_thresh=0.30, max_matches=19):
        """
        Template Matching artesanal con correlación cruzada normalizada (NCC) + máscara + NMS.
        """

        # 1) Grises
        if img is None or template is None:
            return None

        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            out = img.copy()
        else:
            img_gray = img.copy()
            out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        if len(template.shape) == 3:
            tpl_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            tpl_gray = template.copy()

        # Suavizado leve para bajar ruido sin destruir bordes
        img_gray = cv2.GaussianBlur(img_gray, (3, 3), 0)
        tpl_gray = cv2.GaussianBlur(tpl_gray, (3, 3), 0)

        img_f = img_gray.astype(np.float32)
        tpl_f = tpl_gray.astype(np.float32)

        H, W = img_f.shape
        h, w = tpl_f.shape

        # 2) Validación tamaños
        if h > H or w > W:
            # Template más grande que imagen → no hay match posible
            return out

        # 3) Máscara
        _, mask = cv2.threshold(tpl_gray, 50, 1, cv2.THRESH_BINARY)
        mask = mask.astype(np.float32)
        mask_sum = mask.sum() + 1e-6  # evitar división por cero

        # Precalcular stats del template (solo en zonas de máscara)
        t_mean = (tpl_f * mask).sum() / mask_sum
        t_std = np.sqrt(((tpl_f * mask - t_mean) ** 2).sum() / mask_sum) + 1e-6

        # 4) NCC
        corr = np.zeros((H - h + 1, W - w + 1), dtype=np.float32)

        for y in range(H - h + 1):
            for x in range(W - w + 1):
                region = img_f[y:y + h, x:x + w]

                region_masked = region * mask
                r_mean = region_masked.sum() / mask_sum
                r_std = np.sqrt(((region_masked - r_mean) ** 2).sum() / mask_sum) + 1e-6

                num = np.sum((region_masked - r_mean) * (tpl_f * mask - t_mean))
                corr[y, x] = num / (r_std * t_std)

        # ---------- 5) Candidatos por umbral ----------
        ys, xs = np.where(corr >= threshold)
        cajas = [[int(x), int(y), int(w), int(h), float(corr[y, x])] for y, x in zip(ys, xs)]

        # Si nada supera umbral → usar máximo
        if not cajas:
            y_max, x_max = np.unravel_index(np.argmax(corr), corr.shape)
            cv2.rectangle(out, (x_max, y_max), (x_max + w, y_max + h), (0, 0, 255), 2)
            return out

        # Ordenar por score descendente
        cajas.sort(key=lambda b: b[4], reverse=True)

        # 6) NMS (eliminar duplicados/solapados)
        seleccionadas = []
        for box in cajas:
            if len(seleccionadas) >= max_matches:
                break
            ok = True
            for kept in seleccionadas:
                if self.IOU(box, kept) > iou_thresh:
                    ok = False
                    break
            if ok:
                seleccionadas.append(box)

        # 7) Dibujar
        for (x, y, bw, bh, score) in seleccionadas:
            cv2.rectangle(out, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        return out

    def cargarDataset(self, ruta):
        imagenes = []
        etiquetas = []

        for archivo in os.listdir(ruta):
            rutaIMG = os.path.join(ruta, archivo)

            if not archivo.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img = cv2.imread(rutaIMG)
            if img is None:
                continue

            nombre = os.path.splitext(archivo)[0]

            match = re.match(r"[A-Za-z]+", nombre)
            if match:
                etiqueta = match.group(0).lower()
            else:
                print("No se pudo extraer etiqueta de:", archivo)
                continue

            imagenes.append(img)
            etiquetas.append(etiqueta)

        return imagenes, etiquetas

    def entrenarClasificador(self):

        print("🚀 Entrenando clasificador avanzando...")

        from sklearn.ensemble import RandomForestClassifier
        from joblib import dump

        imgs, labels = self.cargarDataset("dataset")

        X = []
        y = []

        for img, label in zip(imgs, labels):

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 0)

            # OTSU invertido
            _, bin_img = cv2.threshold(
                blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            # Engrosar contornos
            kernel = np.ones((3, 3), np.uint8)
            bin_img = cv2.dilate(bin_img, kernel, iterations=2)

            contornos, _ = cv2.findContours(
                bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contornos:

                area = cv2.contourArea(cnt)
                if area < 20:
                    continue

                x, y1, w, h = cv2.boundingRect(cnt)

                perimetro = cv2.arcLength(cnt, True)
                circularidad = (4 * np.pi * area) / (perimetro ** 2 + 1e-6)
                compactacion = (perimetro ** 2) / (area + 1e-6)

                # Excentricidad
                if len(cnt) >= 5:
                    (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
                    mayor = max(a1, a2)
                    menor = min(a1, a2)
                    excentricidad = np.sqrt(1 - (menor / mayor) ** 2)
                else:
                    excentricidad = 0

                aspect_ratio = w / (h + 1e-6)

                # NUEVO DESCRIPTOR 1: SOLIDEZ
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidez = area / (hull_area + 1e-6)

                # NUEVO DESCRIPTOR 2: Nº DE VÉRTICES
                epsilon = 0.01 * perimetro
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                vertices = len(approx)

                # --- NUEVO DESCRIPTOR 3: PERÍMETRO / ÁREA ---
                ratio_per_area = perimetro / (area + 1e-6)

                # Roi normalizado
                roi = bin_img[y1:y1 + h, x:x + w]
                roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_NEAREST)

                # Momentos de Hu
                M = cv2.moments(roi)
                hu = cv2.HuMoments(M).flatten()
                hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

                # Vector de características final
                vector = [
                    area,
                    perimetro,
                    circularidad,
                    compactacion,
                    excentricidad,
                    aspect_ratio,

                    solidez,
                    vertices,
                    ratio_per_area,

                    *hu  # 7 valores
                ]

                X.append(vector)
                y.append(label)

        X = np.array(X)
        y = np.array(y)

        print(f"📌 Total de características por muestra: {X.shape[1]} (deben ser 17)")
        print(f"📌 Total muestras: {len(y)}")

        #Modelo
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
        clf.fit(X, y)

        dump(clf, "modelo.pkl")

        print("✅ Modelo avanzado entrenado correctamente y guardado.")
        return clf

    def clasificador(self, img):

        # Cargar modelo
        try:
            clf = load("modelo.pkl")
        except:
            print("❌ No se pudo cargar modelo.pkl. Entrena el modelo primero.")
            return img

        salida = img.copy()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # OTSU invertido
        _, bin_img = cv2.threshold(
            blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

        # Engrosar contornos para figuras de línea
        kernel = np.ones((3, 3), np.uint8)
        bin_img = cv2.dilate(bin_img, kernel, iterations=2)

        # Encontrar contornos
        contornos, _ = cv2.findContours(
            bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contornos:

            area = cv2.contourArea(cnt)
            if area < 20:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            perimetro = cv2.arcLength(cnt, True)
            circularidad = (4 * np.pi * area) / (perimetro ** 2 + 1e-6)
            compactacion = (perimetro ** 2) / (area + 1e-6)

            # Excentricidad
            if len(cnt) >= 5:
                (_, _), (a1, a2), _ = cv2.fitEllipse(cnt)
                mayor = max(a1, a2)
                menor = min(a1, a2)
                excentricidad = np.sqrt(1 - (menor / mayor) ** 2)
            else:
                excentricidad = 0.0

            aspect_ratio = w / (h + 1e-6)

            # SOLIDEZ
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            solidez = area / (hull_area + 1e-6)

            # VÉRTICES
            epsilon = 0.01 * perimetro
            approx = cv2.approxPolyDP(cnt, epsilon, True)
            vertices = len(approx)

            # RATIO PERÍMETRO/ÁREA
            ratio_per_area = perimetro / (area + 1e-6)

            # ROI normalizada
            roi = bin_img[y:y + h, x:x + w]
            roi = cv2.resize(roi, (128, 128), interpolation=cv2.INTER_NEAREST)

            # Hu Moments normalizados
            M = cv2.moments(roi)
            hu = cv2.HuMoments(M).flatten()
            hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

            # VECTOR FINAL (17 características)
            X = np.array([
                area,
                perimetro,
                circularidad,
                compactacion,
                excentricidad,
                aspect_ratio,

                solidez,
                vertices,
                ratio_per_area,

                *hu
            ]).reshape(1, -1)

            # Predicción
            pred = clf.predict(X)[0]

            # Dibujar resultado
            cv2.rectangle(salida, (x, y), (x + w, y + h), (0, 0, 255), 3)
            cv2.putText(
                salida, pred, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
            )

        return salida

    def identificar_monedas(self, img):
        salida = img.copy()

        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, bin = cv2.threshold(gray, 90, 255, cv2.THRESH_BINARY)
        else:
            bin = img.copy()

        contornos, _ = cv2.findContours(bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        areas = []
        for c in contornos:

            area = cv2.contourArea(c)

            if area < 500 or area > 20000:
                continue

            areas.append(area)

            cv2.drawContours(salida, [c], -1, (0, 255, 0), 4)

            M = cv2.moments(c)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                x, y, w, h = cv2.boundingRect(c)
                cx = x + w // 2
                cy = y + h // 2

            if area > 6000 and area < 10000:
                etiqueta = f"{.50}"
            elif area > 10000 and area < 13000:
                etiqueta = f"{1}"
            elif area > 13000 and area < 16000:
                etiqueta = f"{2}"
            elif area > 16000 and area < 19500:
                etiqueta = f"{5}"

            cv2.putText(salida, etiqueta, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)

        return salida
