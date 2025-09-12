# Importación de librerías
import numpy as np
import cv2

# Importación de librerías visuales
import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk

# Importacion de librerías personales
from librerias.OperacionesLogicas2 import *
from librerias.Ruido import *
from librerias.Filtros_Bajas import *
from librerias.ProcesadorImagen import *
from librerias.FiltrosPasaAltas import *
from librerias.AjustesDeBrillo import *
from librerias.Umbralizacion import *
from librerias.SliderWindow import *
from librerias.VentanaDeDecision import *


# Configuración del tema y apariencia
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class InterfazProcesadorImagenes(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Configuración de la ventana principal
        self.title("Procesador Avanzado de Imágenes")
        self.geometry("1200x1000+0+0")
        self.minsize(1200, 800)

        # Configurar grid principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Variables de instancia (comentadas - reemplaza con tus clases reales)
        self.procesador = ProcesadorImagen()
        self.operaciones_logicas = OperacionesLogicas2()
        self.ruido = Ruido()
        self.filtro = Filtros()
        self.ajustes_brillo = AjustesDeBrillo()
        self.filtros_pasa_altas = FiltrosPasaAltas()
        self.umbralizacion = Umbralizacion()

        self.imagen_1 = None
        self.imagen_2 = None
        self.imagen_display = [None,None]
        self.imagen_1_hist = []
        self.imagen_2_hist = []
        self.imagen_1_indice = 0
        self.imagen_2_indice = 0
        self.indice_actual = 0

        self.crear_interfaz()

    def crear_interfaz(self):
        # Panel lateral izquierdo para controles
        self.sidebar_frame = ctk.CTkScrollableFrame(self, width=280, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, rowspan=4, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(4, weight=1)

        # Logo y título
        self.logo_label = ctk.CTkLabel(
            self.sidebar_frame,
            text="Procesador\nde Imágenes",
            font=ctk.CTkFont(size=20, weight="bold")
        )
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Sección de carga de imágenes
        self.crear_seccion_carga()

        # Sección de procesamiento básico
        self.crear_seccion_procesamiento()

        # Sección de ajustes de brillo
        self.crear_seccion_ajustes_de_brillo()

        # Sección de operaciones lógicas
        self.crear_seccion_logicas()

        # Sección de ruido y filtros
        self.crear_seccion_ruido()

        # Sección de segmentación
        self.crear_seccion_segmentacion()

        # Botón de guardar
        self.crear_seccion_guardar()

        # Panel principal con pestañas
        self.crear_panel_principal()

        # Configurar el selector de tema
        self.crear_selector_tema()

    def selector_de_imagenes(self,choice):
        if choice == "Imagen 1":
            self.indice_actual = 0
            self.mostrar_mensaje("Imagen 1 seleccionada")
        else:
            self.indice_actual = 1
            self.mostrar_mensaje("Imagen 2 seleccionada")

    def crear_seccion_carga(self):
        i = 0
        # Frame para carga de imágenes
        self.carga_frame = ctk.CTkFrame(self.sidebar_frame)
        self.carga_frame.grid(row=1, column=0, padx=(20, 20), pady=(20, 10), sticky="ew")

        # Título de la sección
        self.carga_label = ctk.CTkLabel(
            self.carga_frame,
            text="📁 Cargar Imágenes",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.carga_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        self.ruido_sub_label = ctk.CTkLabel(
            self.carga_frame,
            text="Seleccionar imagen:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.ruido_sub_label.grid(row=1, column=0, padx=20, pady=(5, 5))

        self.combo_selector = ctk.CTkComboBox(
            self.carga_frame,
            values=["Imagen 1","Imagen 2"],
            command=self.selector_de_imagenes
        )
        self.combo_selector.grid(row=2, column=0, padx=20, pady=(5,5))
        
        # Botones de carga

        self.ruido_sub_label = ctk.CTkLabel(
            self.carga_frame,
            text="Imagen 1:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.ruido_sub_label.grid(row=3, column=0, padx=20, pady=(5, 5))

        botones_carga_imagen1 = [
            ("🖼️ Cargar", self.cargar_imagen_1),
            ("🗑️ Eliminar",self.eliminar_imagen_1),
            ("🧊 Restablecer",self.restablecer_imagen_1)
        ]

        for i, (texto, comando) in enumerate(botones_carga_imagen1):
            btn = ctk.CTkButton(
                self.carga_frame,
                text=texto,
                command=comando,
                height=30,
                hover_color="#000000"
            )
            btn.grid(row=i + 4, column=0, padx=20, pady=3, sticky="ew")

        
        self.ruido_sub_label = ctk.CTkLabel(
            self.carga_frame,
            text="Imagen 2:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.ruido_sub_label.grid(row=len(botones_carga_imagen1)+5,
                                   column=0, padx=20, pady=(5, 5))

        botones_carga_imagen2 = [
            ("🖼️ Cargar", self.cargar_imagen_2),
            ("🗑️ Eliminar",self.eliminar_imagen_2),
            ("🧊 Restablecer",self.restablecer_imagen_2)
        ]

        for i, (texto, comando) in enumerate(botones_carga_imagen2):
            btn = ctk.CTkButton(
                self.carga_frame,
                text=texto,
                command=comando,
                height=30,
                hover_color="#000000"
            )
            btn.grid(row=i + len(botones_carga_imagen1) + 6, column=0, padx=20, pady=3, sticky="ew")

    def crear_seccion_procesamiento(self):
        # Frame para procesamiento básico
        self.procesamiento_frame = ctk.CTkFrame(self.sidebar_frame)
        self.procesamiento_frame.grid(row=2, column=0, padx=(20, 20), pady=10, sticky="ew")

        # Título de la sección
        self.procesamiento_label = ctk.CTkLabel(
            self.procesamiento_frame,
            text="⚙️ Procesamiento Básico",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.procesamiento_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Botones de procesamiento
        botones_procesamiento = [
            ("🔳 Escala de Grises", self.convertir_a_grises),
            ("📊 Binarizar", self.aplicar_umbral),
            ("📊 Calcular Histogramas", self.calcular_histogramas)
        ]

        for i, (texto, comando) in enumerate(botones_procesamiento):
            btn = ctk.CTkButton(
                self.procesamiento_frame,
                text=texto,
                command=comando,
                height=30,
                fg_color="#9A721D",
                hover_color="#000000"
            )
            btn.grid(row=i + 1, column=0, padx=20, pady=3, sticky="ew")

        # Espaciado final
        ctk.CTkLabel(self.procesamiento_frame, text="").grid(row=len(botones_procesamiento) + 1, column=0, pady=(0, 15))

    def crear_seccion_ajustes_de_brillo(self):
        # Frame para procesamiento básico
        self.ajustes_frame = ctk.CTkFrame(self.sidebar_frame)
        self.ajustes_frame.grid(row=3, column=0, padx=(20, 20), pady=10, sticky="ew")

        # Título de la sección
        self.ajustes_label = ctk.CTkLabel(
            self.ajustes_frame,
            text="🔦 Ajustes de brillo",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.ajustes_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Botones de procesamiento
        botones_ajustes = [
            ("🔳 Ecualización estandar", self.aplicar_ecualizacion_estandar),
            ("📈 Ecualización Hipercúbica", self.ecualizacion_hipercubica),
            ("📊 Corrección Gamma", self.aplicar_correccion_gamma),
            ("📈 Expansión lineal de contraste", self.aplicar_expansion_lineal),
            ("📊 Transformación exponencial", self.aplicar_transformacion_exponencial),
            ("📈 Ecualización adaptativa",self.aplicar_ecualizacion_adaptativa),
            ("📈 Transformación rayleigh",self.aplicar_transformacion_rayleigh)
        ]

        for i, (texto, comando) in enumerate(botones_ajustes):
            btn = ctk.CTkButton(
                self.ajustes_frame,
                text=texto,
                command=comando,
                height=30,
                fg_color="#445725",
                hover_color="#000000"
            )
            btn.grid(row=i + 1, column=0, padx=20, pady=3, sticky="ew")

        # Espaciado final
        ctk.CTkLabel(self.ajustes_frame, text="").grid(row=len(botones_ajustes) + 1, column=0, pady=(0, 15))
    
    def crear_seccion_logicas(self):
        # Frame para operaciones lógicas
        self.logicas_frame = ctk.CTkFrame(self.sidebar_frame)
        self.logicas_frame.grid(row=4, column=0, padx=(20, 20), pady=10, sticky="ew")

        # Título de la sección
        self.logicas_label = ctk.CTkLabel(
            self.logicas_frame,
            text="🔗 Operaciones Lógicas y\n aritmeticas",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.logicas_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Botón de operaciones lógicas
        botones_logicas = [
            ("🔳 Suma", self.aplicar_suma_gui),
            ("📊 Resta", self.aplicar_resta_gui),
            ("🧮 Multiplicación", self.aplicar_multiplicacion_gui),
            ("🔳 AND", self.aplicar_and_gui),
            ("📊 OR", self.aplicar_or_gui),
            ("🧮 XOR", self.aplicar_xor_gui),
            ("🧮 NOT", self.aplicar_not_gui)
        ]

        for i, (texto, comando) in enumerate(botones_logicas):
            btn = ctk.CTkButton(
                self.logicas_frame,
                text=texto,
                command=comando,
                height=30,
                fg_color="#11500C",
                hover_color="#000000"
            )
            btn.grid(row=i + 1, column=0, padx=20, pady=3, sticky="ew")

    def crear_seccion_ruido(self):
        # Frame para ruido y filtros
        self.ruido_frame = ctk.CTkFrame(self.sidebar_frame)
        self.ruido_frame.grid(row=5, column=0, padx=(20, 20), pady=10, sticky="ew")

        # Título de la sección
        self.ruido_label = ctk.CTkLabel(
            self.ruido_frame,
            text="🔊 Ruido y Filtros",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.ruido_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Subsección de ruido
        self.ruido_sub_label = ctk.CTkLabel(
            self.ruido_frame,
            text="Agregar Ruido:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.ruido_sub_label.grid(row=1, column=0, padx=20, pady=(5, 5))

        ruido_botones = [
            ("🧂 Sal y Pimienta", self.agregar_ruido_sal_pimienta),
            ("📡 Gaussiano", self.agregar_ruido_gaussiano)
        ]

        for i, (texto, comando) in enumerate(ruido_botones):
            btn = ctk.CTkButton(
                self.ruido_frame,
                text=texto,
                command=comando,
                height=30,
                fg_color="#001A61",
                hover_color="#000000"
            )
            btn.grid(row=i + 2, column=0, padx=20, pady=3, sticky="ew")

        # Subsección de filtros
        self.filtros_sub_label = ctk.CTkLabel(
            self.ruido_frame,
            text="Aplicar Filtros Pasa-bajas:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.filtros_sub_label.grid(row=len(ruido_botones) + 2, column=0, padx=20, pady=(10, 5))

        filtros_botones_pb = [
            ("📈 Promediador", self.aplicar_filtro_promediador),
            ("📈 Pesado", self.aplicar_filtro_pesado),
            ("📈 Mediana", self.aplicar_filtro_mediana),
            ("📈 Moda", self.aplicar_filtro_Moda),
            ("📈 Bilateral",self.aplicar_filtro_bilateral),
            ("📈 Max",self.aplicar_filtro_max),
            ("📈 Min",self.aplicar_filtro_min),
            ("📈 Gaussiano", self.aplicar_filtro_gaussiano)
        ]

        for i, (texto, comando) in enumerate(filtros_botones_pb):
            btn = ctk.CTkButton(
                self.ruido_frame,
                text=texto,
                command=comando,
                height=30,
                fg_color= "#0A4B43",
                hover_color="#000000"
            )
            btn.grid(row=i + len(ruido_botones) + 3, column=0, padx=20, pady=3, sticky="ew")

        self.filtros_sub_label = ctk.CTkLabel(
            self.ruido_frame,
            text="Aplicar Filtros Pasa-altas:",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.filtros_sub_label.grid(row=i + len(ruido_botones) + 4, column=0, padx=20, pady=(10, 5))

        filtros_botones_pa = [
            ("📈 Robinson", self.aplicar_filtro_Robinson),
            ("📈 Robert", self.aplicar_filtro_Robert),
            ("📈 Prewitt", self.aplicar_filtro_Prewitt),
            ("📈 Sobel", self.aplicar_filtro_Sobel),
            ("📈 Kirsch",self.aplicar_filtro_Kirch),
            ("📈 Canny",self.aplicar_filtro_Canny),
            ("📈 Op. Laplaciano",self.aplicar_Operador_Laplaciano)
        ]

        for i, (texto, comando) in enumerate(filtros_botones_pa):
            btn = ctk.CTkButton(
                self.ruido_frame,
                text=texto,
                command=comando,
                height=30,
                    fg_color="#29164A",
                hover_color="#000000"
            )
            btn.grid(row=i + len(ruido_botones) + len(filtros_botones_pb) + 5, column=0, padx=20, pady=3, sticky="ew")

        # Espaciado final
        ctk.CTkLabel(self.ruido_frame, text="").grid(row=len(ruido_botones) + len(filtros_botones_pb) + 
                                                     len(filtros_botones_pa) + 5, column=0, pady=(0, 15))

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

    def aplicar_analisis_objetos(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return

        try:
            imagen_binaria = self.imagen_display[self.indice_actual]
            imagen_original = self.imagen_1 if self.indice_actual == 0 else self.imagen_2

            imagen_resultado, datos = self.analizar_objetos(imagen_binaria, imagen_original, min_area=200)

            self.imagen_display[self.indice_actual] = imagen_resultado
            texto = f"📏 Análisis de Objetos\nSe detectaron {len(datos)} objetos\nImagen {self.indice_actual+1}"
            self.mostrar_imagen(self.panel_objetos, imagen_resultado, texto)
            self.tabview.set("🧊 Detección de objetos")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error al analizar objetos: {str(e)}")

    def crear_seccion_segmentacion(self):
        # Frame para segmentación
        self.segmentacion_frame = ctk.CTkFrame(self.sidebar_frame)
        self.segmentacion_frame.grid(row=6, column=0, padx=(20, 20), pady=10, sticky="ew")

        # Título de la sección
        self.segmentacion_label = ctk.CTkLabel(
            self.segmentacion_frame,
            text="✂️ Segmentación",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.segmentacion_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Botones de segmentación
        segmentacion_botones = [
            ("🎯 Umbral Media", self.aplicar_umbral_media),
            ("🎯 Método de Otsu", self.aplicar_filtro_otsu),
            ("🎯 Multiumbralización", self.aplicar_multiubralizacion),
            ("🎯 Entropía Kapur", self.aplicar_entropia_kapur),
            ("🎯 Umbral por banda", self.aplicar_umbral_banda),
            ("🎯 Umbral adaptativo", self.aplicar_umbral_adaptativo),
            ("🎯 Minimo del histograma", self.aplicar_minimo_en_el_histograma),
            ("🎯 Vecindad 4", self.aplicar_vecindad_4),
            ("🎯 Vecindad 8", self.aplicar_vecindad_8),
            ("📏 Análisis de Objetos", self.aplicar_analisis_objetos)
        ]

        for i, (texto, comando) in enumerate(segmentacion_botones):
            btn = ctk.CTkButton(
                self.segmentacion_frame,
                text=texto,
                command=comando,
                height=35,
                fg_color="#631D29",
                hover_color="#000000"
            )
            btn.grid(row=i + 1, column=0, padx=20, pady=5, sticky="ew")

        # Espaciado final
        ctk.CTkLabel(self.segmentacion_frame, text="").grid(row=len(segmentacion_botones) + 1, column=0, pady=(0, 15))

    def crear_seccion_guardar(self):
        # Frame para guardar
        self.guardar_frame = ctk.CTkFrame(self.sidebar_frame)
        self.guardar_frame.grid(row=7, column=0, padx=(20, 20), pady=10, sticky="ew")

        # Título de la sección
        self.guardar_label = ctk.CTkLabel(
            self.guardar_frame,
            text="💾 Guardar Resultado",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.guardar_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Botón de guardar
        self.btn_guardar = ctk.CTkButton(
            self.guardar_frame,
            text="💾 Guardar Imagen Actual",
            command=self.guardar_imagen_actual,
            height=40,
            fg_color="#229954",
            font=ctk.CTkFont(size=14, weight="bold"),
            hover_color="#000000"
        )
        self.btn_guardar.grid(row=1, column=0, padx=20, pady=(5, 15), sticky="ew")

    def crear_panel_principal(self):
        # Frame principal para el contenido
        self.main_frame = ctk.CTkFrame(self, corner_radius=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        # Configurar grid del frame principal
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        # Header frame con altura fija
        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent", height=40)
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))
        self.header_frame.grid_propagate(False)  # Importante para mantener altura fija

        # Botón verde a la izquierda
        self.boton_verde = ctk.CTkButton(
            self.header_frame,
            text="↩️",
            fg_color="green",
            hover_color="#006400",
            text_color="white",
            width=40,
            height=30,
            corner_radius=6,
            command=self.deshacer
        )
        self.boton_verde.pack(side="left")

        # Título absolutamente centrado en el header
        self.main_label = ctk.CTkLabel(
            self.header_frame,
            text="Área de Visualización",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        self.main_label.place(relx=0.5, rely=0.5, anchor="center")

        # Tabview para las diferentes pestañas
        self.tabview = ctk.CTkTabview(self.main_frame, width=250)
        self.tabview.grid(row=1, column=0, padx=20, pady=(10, 20), sticky="nsew")

        # Crear pestañas
        self.tab_basico = self.tabview.add("🔧 Básico")
        self.tab_logicas = self.tabview.add("🔗 Lógicas")
        self.tab_ruido = self.tabview.add("🔊 Ruido/Filtros")
        self.tab_segmentacion = self.tabview.add("✂️ Segmentación")
        self.tab_objetos = self.tabview.add("🧊 Detección de objetos")
        self.tab_histogramas = self.tabview.add("📊 Histogramas")

        # Configurar cada pestaña como scrollable
        self.configurar_pestanas()




    def configurar_pestanas(self):
        # Configurar cada pestaña con scroll
        pestanas = [
            (self.tab_basico, "panel_basico"),
            (self.tab_logicas, "panel_logicas"),
            (self.tab_ruido, "panel_ruido"),
            (self.tab_segmentacion, "panel_segmentacion"),
            (self.tab_objetos,"panel_objetos"),
            (self.tab_histogramas, "panel_histogramas")
        ]

        for tab, nombre_panel in pestanas:
            # Crear frame scrollable para cada pestaña
            panel = ctk.CTkScrollableFrame(tab)
            panel.pack(fill="both", expand=True, padx=10, pady=10)
            setattr(self, nombre_panel, panel)

    def establecer_tabview(self, panel):
        if panel == self.panel_basico:
            self.tabview.set("🔧 Básico")
        elif panel == self.panel_logicas:
            self.tabview.set("🔗 Lógicas")
        elif panel == self.panel_ruido:
            self.tabview.set("🔊 Ruido/Filtros")
        elif panel == self.panel_segmentacion:
            self.tabview.set("✂️ Segmentación")
        elif panel == self.panel_objetos:
            self.tabview.set("panel_objetos")
        elif panel == self.panel_histogramas:
            self.tabview.set("📊 Histogramas")

    def crear_selector_tema(self):
        # Frame para selector de tema
        self.tema_frame = ctk.CTkFrame(self.sidebar_frame)
        self.tema_frame.grid(row=8, column=0, padx=(20, 20), pady=10, sticky="ew")

        # Título
        self.tema_label = ctk.CTkLabel(
            self.tema_frame,
            text="🎨 Tema",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        self.tema_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        # Selector de tema
        self.tema_optionmenu = ctk.CTkOptionMenu(
            self.tema_frame,
            values=["Dark", "Light", "System"],
            command=self.cambiar_tema
        )
        self.tema_optionmenu.grid(row=1, column=0, padx=20, pady=(5, 15), sticky="ew")

    def cambiar_tema(self, nuevo_tema):
        ctk.set_appearance_mode(nuevo_tema)

    # Métodos de funcionalidad (placeholders - implementa según tus clases)

    def cargar_imagen(self):
        ruta = None
        img = None
        ruta = filedialog.askopenfilename(
            title="Seleccionar imagen principal",
            filetypes=[("Archivos de imagen", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )

        try:
            if ruta is not None:
                img = cv2.imread(ruta)
                if img is not None:
                    img = cv2.resize(img, (400,400))
                    return img
        except Exception as e:
                self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def cargar_imagen_1(self):
        self.imagen_1 = self.cargar_imagen()

        if self.imagen_1 is not None:
            self.mostrar_imagen(self.panel_basico, self.imagen_1, "Imagen 1", indicador=False, indicadorDeshacer=True)
            self.tabview.set("🔧 Básico")
            self.imagen_display[0] = self.imagen_1
            self.imagen_1_indice += 1
            self.imagen_1_hist.append(self.imagen_1)
            self.imagen_1_hist.append("Imagen 1")
            self.imagen_1_hist.append(self.panel_basico)
        else:
            self.mostrar_mensaje("❌ Error al cargar la imagen")

    def cargar_imagen_2(self):
        self.imagen_2 = self.cargar_imagen()

        if self.imagen_2 is not None:
            self.mostrar_imagen(self.panel_basico, self.imagen_2, "Imagen 2",indicador=False,indicadorDeshacer=True)
            self.tabview.set("🔧 Básico")
            self.imagen_display[1] = self.imagen_2
            self.imagen_2_indice += 1
            self.imagen_2_hist.append(self.imagen_2)
            self.imagen_2_hist.append("Imagen 2")
            self.imagen_2_hist.append(self.panel_basico)
        else:
            self.mostrar_mensaje("❌ Error al cargar la imagen")

    def deshacer(self):
        if self.indice_actual == 0:
            if self.imagen_1_indice == 0:
                self.mostrar_mensaje("Ya no hay más cambios por deshacer")
                return
            self.imagen_1_indice -= 1
            if self.imagen_1_indice == 0:
                self.limpiar_todas_las_pestanas()
                self.tabview.set("🔧 Básico")
                self.eliminar_imagen_1()
            else:
                for i in range(3):
                    self.imagen_1_hist.pop()
                self.mostrar_imagen(
                    panel=self.imagen_1_hist[(self.imagen_1_indice*3)-1],
                    imagen=self.imagen_1_hist[(self.imagen_1_indice*3)-3],
                    titulo=self.imagen_1_hist[(self.imagen_1_indice*3)-2],
                    indicadorDeshacer=True
                    )
                self.establecer_tabview(self.imagen_1_hist[(self.imagen_1_indice*3)-1])
        else:
            if self.imagen_2_indice == 0:
                self.mostrar_mensaje("Ya no hay más cambios por deshacer")
                return
            self.imagen_2_indice -= 1
            if self.imagen_2_indice == 0:
                self.limpiar_todas_las_pestanas()
                self.tabview.set("🔧 Básico")
                self.eliminar_imagen_2()
            else:
                for i in range(3):
                    self.imagen_2_hist.pop()
                self.mostrar_imagen(
                    panel=self.imagen_2_hist[(self.imagen_2_indice*3)-1],
                    imagen=self.imagen_2_hist[(self.imagen_2_indice*3)-3],
                    titulo=self.imagen_2_hist[(self.imagen_2_indice*3)-2],
                    indicadorDeshacer=True
                    )
                self.establecer_tabview(self.imagen_2_hist[(self.imagen_2_indice*3)-1])

    def verificar_imagen_cargada(self, img):
        if img is None:
            self.mostrar_mensaje(f"⚠️ Por favor cargue la imagen {self.indice_actual+1} primero")
            return False
        else:
            return True

    def restablecer_imagen_1(self):
        if self.verificar_imagen_cargada(self.imagen_1) is False:
            return
        
        self.imagen_1_indice = 0
        self.imagen_1_hist.clear()
        self.limpiar_todas_las_pestanas()
        self.imagen_display[0] = self.imagen_1
        self.mostrar_imagen(self.panel_basico, self.imagen_1, "Imagen 1",indicador=False)
        self.tabview.set("🔧 Básico")


    def restablecer_imagen_2(self):
        if self.verificar_imagen_cargada(self.imagen_2) is False:
            return
        
        self.imagen_2_indice = 0
        self.imagen_2_hist.clear()
        self.limpiar_todas_las_pestanas()
        self.imagen_display[1] = self.imagen_2
        self.mostrar_imagen(self.panel_basico, self.imagen_2, "Imagen 2",indicador=False)
        self.tabview.set("🔧 Básico")
        
    def eliminar_imagen_1(self):
        if self.imagen_1 is None:
            self.mostrar_mensaje("No se ha cargado ninguna imagen")
            return
        else:
            self.imagen_1_indice = 0
            self.imagen_1_hist.clear()
            self.limpiar_todas_las_pestanas()
            self.imagen_1 = None
            self.imagen_display[0] = None
            self.limpiar_pestana("panel_basico")
            self.tabview.set("🔧 Básico")

    def eliminar_imagen_2(self):
        if self.imagen_2 is None:
            self.mostrar_mensaje("No se ha cargado ninguna imagen")
            return
        else:
            self.imagen_2_indice = 0
            self.imagen_2_hist.clear()
            self.limpiar_todas_las_pestanas()
            self.imagen_2 = None
            self.imagen_display[1] = None
            self.limpiar_pestana("panel_basico")
            self.tabview.set("🔧 Básico")

    def convertir_a_grises(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_grises = self.procesador.convertir_a_grises(self.imagen_display[self.indice_actual])

            self.imagen_display[self.indice_actual] = imagen_grises
            self.mostrar_imagen(self.panel_basico, imagen_grises, f"Imagen {self.indice_actual+1} en Escala de Grises")
            self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_umbral(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return

        try:
            imagen_binarizada = self.procesador.convertir_a_grises(self.imagen_display[self.indice_actual])
            slider = SliderWindow(
                title="Seleccionar el umbral",
                min_val=0,
                max_val=255,
                initial_val=127,
                step=1
                )
            if slider.value is not None:
                umbral = slider.value
            else:
                return
            
            imagen_binarizada = self.procesador.aplicar_binarizacion(imagen_binarizada, umbral)

            self.imagen_display[self.indice_actual] = imagen_binarizada
            self.mostrar_imagen(self.panel_basico, imagen_binarizada, f"Imagen {self.indice_actual+1} binarizada\numbral de: {umbral}")
            self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    # Placeholders para otros métodos
    def ecualizacion_hipercubica(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_ecualizada = self.ajustes_brillo.ecualizacion_hipercubica(self.imagen_display[self.indice_actual])
            if imagen_ecualizada is not None:
                self.imagen_display[self.indice_actual] = imagen_ecualizada
                self.mostrar_imagen(self.panel_basico, imagen_ecualizada, f"Ecualización hipercúbica\nImagen {self.indice_actual+1}")
                self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def calcular_histogramas(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return

        self.tabview.set("📊 Histogramas")
        self.limpiar_pestana("panel_histogramas")

        imagen_a_calcular = self.imagen_display[self.indice_actual] 
        

        # Verificar si la imagen es binaria o en escala de grises
        es_gris_o_binaria = len(imagen_a_calcular.shape) == 2 or (
            len(imagen_a_calcular.shape) == 3 and imagen_a_calcular.shape[2] == 1
        )

        # Mostrar solo histograma en escala de grises
        if es_gris_o_binaria:
            hist_gris = self.procesador.calcular_histograma_gris(imagen_a_calcular)

            frame_gris = ctk.CTkFrame(self.panel_histogramas)
            frame_gris.grid(row=0, column=0, padx=30, pady=20, sticky="nsew")

            histograma_gris = FigureCanvasTkAgg(hist_gris, master=frame_gris)
            histograma_gris.draw()
            histograma_gris.get_tk_widget().pack(padx=10, pady=10)

            label_gris = ctk.CTkLabel(frame_gris, text=f"Histograma en Escala de Grises\nImagen {self.indice_actual+1}")
            label_gris.pack(pady=(0, 10))

            self.panel_histogramas.columnconfigure(0, weight=1)
        else:

            hist_gris, hist_color = self.procesador.calcular_histogramas(imagen_a_calcular)
            # Mostrar histograma en escala de grises
            frame_gris = ctk.CTkFrame(self.panel_histogramas)
            frame_gris.grid(row=0, column=0, padx=30, pady=20, sticky="nsew")

            histograma_gris = FigureCanvasTkAgg(hist_gris, master=frame_gris)
            histograma_gris.draw()
            histograma_gris.get_tk_widget().pack(padx=10, pady=10)

            label_gris = ctk.CTkLabel(frame_gris, text=f"Histograma en Escala de Grises\nImagen {self.indice_actual+1}")
            label_gris.pack(pady=(0, 10))

            # Mostrar histograma a color
            frame_color = ctk.CTkFrame(self.panel_histogramas)
            frame_color.grid(row=0, column=1, padx=30, pady=20, sticky="nsew")

            histograma_color = FigureCanvasTkAgg(hist_color, master=frame_color)
            histograma_color.draw()
            histograma_color.get_tk_widget().pack(padx=10, pady=10)

            label_color = ctk.CTkLabel(frame_color, text=f"Histograma a Color (RGB)\nImagen {self.indice_actual+1}")
            label_color.pack(pady=(0, 10))

            self.panel_histogramas.columnconfigure(0, weight=1)
            self.panel_histogramas.columnconfigure(1, weight=1)


    def aplicar_ecualizacion_estandar(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_ecualizacion = self.ajustes_brillo.ecualizacion_de_histograma(img=self.imagen_display[self.indice_actual])
            if imagen_ecualizacion is not None:
                self.imagen_display[self.indice_actual] = imagen_ecualizacion
                self.mostrar_imagen(self.panel_basico, imagen_ecualizacion, f"Ecualización estandar\nImagen {self.indice_actual+1}")
                self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_correccion_gamma(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_gamma = self.ajustes_brillo.correccion_gamma(img=self.imagen_display[self.indice_actual])
            if imagen_gamma is not None:
                self.imagen_display[self.indice_actual] = imagen_gamma
                self.mostrar_imagen(self.panel_basico, imagen_gamma, f"Corrección gamma\nImagen {self.indice_actual+1}")
                self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_expansion_lineal(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_expansion = self.ajustes_brillo.expansion_lineal_de_contraste(img=self.imagen_display[self.indice_actual])
            if imagen_expansion is not None:
                self.imagen_display[self.indice_actual] = imagen_expansion
                self.mostrar_imagen(self.panel_basico, imagen_expansion, f"Expansión líneal\nImagen {self.indice_actual+1}")
                self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_transformacion_exponencial(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_transformacion = self.ajustes_brillo.transformacion_exponencial(img=self.imagen_display[self.indice_actual])
            if imagen_transformacion is not None:
                self.imagen_display[self.indice_actual] = imagen_transformacion
                self.mostrar_imagen(self.panel_basico, imagen_transformacion, f"Transformación exponencial\nImagen {self.indice_actual+1}")
                self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_ecualizacion_adaptativa(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_ecualizacion = self.ajustes_brillo.ecualizacion_adaptativa(img=self.imagen_display[self.indice_actual])
            if imagen_ecualizacion is not None:
                self.imagen_display[self.indice_actual] = imagen_ecualizacion
                self.mostrar_imagen(self.panel_basico, imagen_ecualizacion, f"Ecualización adaptativa\nImagen {self.indice_actual+1}")
                self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_transformacion_rayleigh(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_ecualizacion = self.ajustes_brillo.transformacion_rayleigh(img=self.imagen_display[self.indice_actual])
            if imagen_ecualizacion is not None:
                self.imagen_display[self.indice_actual] = imagen_ecualizacion
                self.mostrar_imagen(self.panel_basico, imagen_ecualizacion, f"Transformación rayleigh\nImagen {self.indice_actual+1}")
                self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")


    def agregar_ruido_sal_pimienta(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_ruido = self.ruido.agregar_ruido_sal_pimienta(img=self.imagen_display[self.indice_actual])
            if imagen_ruido is not None:
                self.imagen_display[self.indice_actual] = imagen_ruido
                self.mostrar_imagen(self.panel_ruido, imagen_ruido, f"Ruido sal y pimienta\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def agregar_ruido_gaussiano(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_ruido = self.ruido.agregar_ruido_gaussiano(img=self.imagen_display[self.indice_actual])
            if imagen_ruido is not None:
                self.imagen_display[self.indice_actual] = imagen_ruido
                self.mostrar_imagen(self.panel_ruido, imagen_ruido, 
                                    f"Ruido gaussiano\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_suma_gui(self):
        if self.imagen_1 is None and self.imagen_2 is None:
            self.mostrar_mensaje("Se requiere alguna de las dos imagenes este cargada")
            return
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        ventana_suma = VentanaDeDecision(
            title="Suma",
            mainText="Elija el tipo de suma que se desea realizar",
            firstButton="Suma entre\ndos imagenes",
            secondButton="Suma por\nun escalar",
            command1=self.aplicar_suma_dos_imagenes,
            command2=self.aplicar_suma_escalar
            )
        ventana_suma.lift
        ventana_suma.focus_force()
        ventana_suma.grab_set() 

    def aplicar_suma_dos_imagenes(self):
        if self.imagen_1 is None or self.imagen_2 is None:
            self.mostrar_mensaje("Se necesita cargar las dos imagenes")
            return
        try:
            imagen_suma = self.operaciones_logicas.aplicar_suma(self.imagen_display[0],self.imagen_display[1])
            if imagen_suma is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_suma,
                                    f"Operación suma\nEntre dos imagenes\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
        
    def aplicar_suma_escalar(self):
        try:
            ver = True
            text_ventana = "Ingrese el escalar (de 0 a 255)"
            while ver:
                dialog = ctk.CTkInputDialog(text=text_ventana, title="Escalar")
                val = int(dialog.get_input())
                if val >= 0 and val <= 255:
                    ver = False
                else: 
                    text_ventana = "Ingrese un escalar valido (de 0 a 255)"
            
            imagen_suma = self.operaciones_logicas.aplicar_suma(self.imagen_display[self.indice_actual],val)
            if imagen_suma is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_suma,
                                    f"Operación suma\nCon escalar\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
        
                
    def aplicar_resta_gui(self):
        if self.imagen_1 is None and self.imagen_2 is None:
            self.mostrar_mensaje("Se requiere alguna de las dos imagenes este cargada")
            return
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        ventana_resta = VentanaDeDecision(
            title="Resta",
            mainText="Elija el tipo de resta que se desea realizar",
            firstButton="Resta entre\ndos imagenes",
            secondButton="Resta por\nun escalar",
            command1=self.aplicar_resta_dos_imagenes,
            command2=self.aplicar_resta_escalar
            )
        ventana_resta.lift
        ventana_resta.focus_force()
        ventana_resta.grab_set() 

    def aplicar_resta_dos_imagenes(self):
        if self.imagen_1 is None or self.imagen_2 is None:
            self.mostrar_mensaje("Se necesita cargar las dos imagenes")
            return
        try:
            imagen_resta = self.operaciones_logicas.aplicar_resta(self.imagen_display[0],self.imagen_display[1])
            if imagen_resta is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_resta,
                                    f"Operación resta\nEntre dos imagenes\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_resta_escalar(self):
        try:
            ver = True
            text_ventana = "Ingrese el escalar (de 0 a 255)"
            while ver:
                dialog = ctk.CTkInputDialog(text=text_ventana, title="Escalar")
                val = int(dialog.get_input())
                if val >= 0 and val <= 255:
                    ver = False
                else: 
                    text_ventana = "Ingrese un escalar valido (de 0 a 255)"
            
            imagen_resta = self.operaciones_logicas.aplicar_resta(self.imagen_display[self.indice_actual],val)
            if imagen_resta is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_resta,
                                    f"Operación resta\nCon escalar\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_multiplicacion_gui(self):
        if self.imagen_display[0] is None or self.imagen_display[1] is None:
            self.mostrar_mensaje("Se necesita cargar las dos imagenes")
        else:
            imagen_mult = self.operaciones_logicas.aplicar_multiplicacion(self.imagen_display[0],self.imagen_display[1])
            if imagen_mult is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_mult,f"Operación multiplicación\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")

    def aplicar_and_gui(self):
        if self.imagen_display[0] is None or self.imagen_display[1] is None:
            self.mostrar_mensaje("Se necesita cargar las dos imagenes")
        else:
            imagen_and = self.operaciones_logicas.aplicar_and(self.imagen_display[0],self.imagen_display[1])
            if imagen_and is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_and,f"Operación AND\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")


    def aplicar_or_gui(self):
        if self.imagen_display[0] is None or self.imagen_display[1] is None:
            self.mostrar_mensaje("Se necesita cargar las dos imagenes")
        try:
            imagen_or = self.operaciones_logicas.aplicar_or(self.imagen_display[0],self.imagen_display[1])
            if imagen_or is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_or,f"Operación OR\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_xor_gui(self):
        if self.imagen_display[0] is None or self.imagen_display[1] is None:
            self.mostrar_mensaje("Se necesita cargar las dos imagenes")
        try:
            imagen_xor = self.operaciones_logicas.aplicar_xor(self.imagen_display[0],self.imagen_display[1])
            if imagen_xor is not None:
                self.mostrar_imagen(self.panel_logicas,imagen_xor,f"Operación XOR\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_not_gui(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_not = self.operaciones_logicas.aplicar_not(self.imagen_display[self.indice_actual])
            if imagen_not is not None:
                self.imagen_display[self.indice_actual] = imagen_not
                self.mostrar_imagen(self.panel_logicas,imagen_not,
                                    f"Operación NOT\nImagen {self.indice_actual+1}")
                self.tabview.set("🔗 Lógicas")
            else:
                self.mostrar_mensaje("Error al generar la imagen")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_promediador(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            dialog = ctk.CTkInputDialog(text="Inserte el tamaño del kernel", title="kernel")
            text = dialog.get_input()
            if text is None:
                return 
            print("imput ingresado" + text )
            imagen_filtrada = self.filtro.filtro_promediador(self.imagen_display[self.indice_actual],text)
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro promediador\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro promediador")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
                
    def aplicar_filtro_pesado(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            dialog = ctk.CTkInputDialog(text="Inserte el grado de suavizado\nTiene que ser mayor a 1", title="kernel")
            text = dialog.get_input()
            n = int(text)
            if text is None:
                self.mostrar_mensaje("Error al recibir el dato")
            if n <= 1:
                self.mostrar_mensaje("El valor insertado debe ser mayor a 1")
            imagen_filtrada = self.filtro.filtro_pesado(self.imagen_display[self.indice_actual],n)
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro pesado\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro pesado")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_mediana(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtro.filtro_mediana(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Mediana\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Mediana")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_filtro_Moda(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtro.filtro_moda(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,
                                    f"Filtro Moda\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Moda")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_filtro_bilateral(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtro.filtro_bilateral(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Bilateral\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Bilateral")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_max(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtro.filtro_max(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.imagen_display[self.indice_actual] = imagen_filtrada
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Maximo\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Maximo ")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_min(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtro.filtro_min(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro minimo\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro minimo ")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_filtro_gaussiano(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtro.filtro_gaussiano(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,
                                    f"Filtro Gaussiano\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Gaussiano")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    
    def aplicar_filtro_Robinson(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtros_pasa_altas.filtro_robinson(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Operador Robinson\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Operador Robinson")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_Robert(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtros_pasa_altas.operador_robert(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,
                                    f"Filtro Operador Robert\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Operador Robert")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_Prewitt(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtros_pasa_altas.operador_prewitt(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,
                                    f"Filtro Operador Prewitt\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Operador Prewitt")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_Sobel(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtros_pasa_altas.operador_sobel(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Operador Sobel\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Operador Sobel")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_Kirch(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtros_pasa_altas.operador_kirsch(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Operador Kirsch\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Operador Kirsch")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_filtro_Canny(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtros_pasa_altas.operador_canny(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Operador Canny\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Operador Canny")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")
    
    def aplicar_Operador_Laplaciano(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        try:
            imagen_filtrada = self.filtros_pasa_altas.operador_laplaciano(self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_ruido,imagen_filtrada,f"Filtro Operador Laplaciano\nImagen {self.indice_actual+1}")
                self.tabview.set("🔊 Ruido/Filtros")
               
            else:
                self.mostrar_mensaje("Error al generar la imagen Filtro Operador Laplaciano")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_umbral_media(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            umbral_media = self.umbralizacion.umbral_media(img=self.imagen_display[self.indice_actual])
            if umbral_media is not None:
                self.mostrar_imagen(self.panel_segmentacion, umbral_media, f"Umbral media\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar el umbral media sobre la imagen {self.indice_actual+1}")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_filtro_otsu(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_filtrada = self.umbralizacion.metodo_otsu(img=self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_filtrada, f"Filtro de Otsu\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar el filtro de otsu a la imagen {self.indice_actual+1}")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_multiubralizacion(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_filtrada = self.umbralizacion.multiumbralizacion(img=self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_filtrada, f"Multiubralizacion\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar multiubralizacion a la imagen {self.indice_actual+1}")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_entropia_kapur(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_filtrada = self.umbralizacion.entropia_kapur(img=self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_filtrada, f"Entropia de Kapur\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar la entropia de Kapur a la imagen {self.indice_actual+1}")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_umbral_banda(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_filtrada = self.umbralizacion.umbral_por_banda(img=self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_filtrada, f"Umbral por banda\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar el umbral por banda a la imagen {self.indice_actual+1}")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_umbral_adaptativo(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_filtrada = self.umbralizacion.umbral_adaptativo(img=self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_filtrada, f"Umbral adaptativo\nImagen {self.indice_actual}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar el umbral adaptativo a la imagen {self.indice_actual+1}")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_minimo_en_el_histograma(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_filtrada = self.umbralizacion.minimo_del_histograma(img=self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_filtrada, f"Minimo en el histograma\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar el minimo en el histograma a la imagen {self.indice_actual+1}")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_segmentacion_filtro_Robert(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_filtrada = self.umbralizacion.filtro_Robert(img=self.imagen_display[self.indice_actual])
            if imagen_filtrada is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_filtrada, f"Filtro de Roberts\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar el filtro de Roberts a la imagen {self.indice_actual+1}")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_vecindad_4(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_v4, cantidad_objetos = self.umbralizacion.detectar_objetos_vecindad_4(img=self.imagen_display[self.indice_actual])
            if imagen_v4 is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_v4, f"Vecindad 4\n{cantidad_objetos} objetos detectados\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar vecindad 8 a la imagen {self.indice_actual+1}\nRecomendación: utlizar una imagen binarizada")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def aplicar_vecindad_8(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return
        
        try:
            imagen_v8, cantidad_objetos = self.umbralizacion.detectar_objetos_vecindad_8(img=self.imagen_display[self.indice_actual])
            if imagen_v8 is not None:
                self.mostrar_imagen(self.panel_segmentacion, imagen_v8, f"Vecindad 8\n{cantidad_objetos} objetos detectados\nImagen {self.indice_actual+1}")
                self.tabview.set("✂️ Segmentación")
            else:
                self.mostrar_mensaje(f"Error al aplicar vecindad 8 a la imagen {self.indice_actual+1}\nRecomendación: Utilizar una imagen binarizada")

        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def guardar_imagen_actual(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return

        ruta = filedialog.asksaveasfilename(
            title="Guardar imagen",
            defaultextension=".jpg",
            filetypes=[
                ("JPEG", "*.jpg"),
                ("PNG", "*.png"),
                ("BMP", "*.bmp"),
                ("TIFF", "*.tiff"),
                ("Todos los archivos", "*.*")
            ]
        )

        if ruta:
            try:
                cv2.imwrite(ruta, self.imagen_display[self.indice_actual])
                self.mostrar_mensaje(f"✅ Imagen guardada en {ruta}")
            except Exception as e:
                self.mostrar_mensaje(f"❌ Error al guardar: {str(e)}")

    def mostrar_imagen(self, panel, imagen, titulo, indicador=True, indicadorDeshacer=False):
        # Limpiar panel

        if indicador is True:
            self.imagen_display[self.indice_actual] = imagen
        if indicadorDeshacer is False:
            if self.indice_actual == 0:
                self.imagen_1_indice += 1
                self.imagen_1_hist.append(imagen)
                self.imagen_1_hist.append(titulo)
                self.imagen_1_hist.append(panel)
            else:
                self.imagen_2_indice += 1
                self.imagen_2_hist.append(imagen)
                self.imagen_2_hist.append(titulo)
                self.imagen_2_hist.append(panel)

        for widget in panel.winfo_children():
            widget.destroy()

        try:
            # Convertir imagen de OpenCV a formato RGB para mostrar
            if len(imagen.shape) == 3:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB)
            else:
                imagen_rgb = cv2.cvtColor(imagen, cv2.COLOR_GRAY2RGB)

            # Frame contenedor
            frame_contenedor = ctk.CTkFrame(panel)
            frame_contenedor.pack(padx=20, pady=20, fill="both", expand=True)

            # Título
            titulo_label = ctk.CTkLabel(
                frame_contenedor,
                text=titulo,
                font=ctk.CTkFont(size=18, weight="bold")
            )
            titulo_label.pack(pady=(15, 10))

            # Información de dimensiones
            altura, anchura = imagen_rgb.shape[:2]
            info_label = ctk.CTkLabel(
                frame_contenedor,
                text=f"📏 Dimensiones: {anchura} x {altura} píxeles",
                font=ctk.CTkFont(size=12)
            )
            info_label.pack(pady=(0, 10))

            # Redimensionar imagen para mostrar
            max_width, max_height = 600, 400
            factor_width = max_width / anchura if anchura > max_width else 1
            factor_height = max_height / altura if altura > max_height else 1
            factor = min(factor_width, factor_height)

            if factor < 1:
                nueva_anchura = int(anchura * factor)
                nueva_altura = int(altura * factor)
                imagen_redimensionada = cv2.resize(imagen_rgb, (nueva_anchura, nueva_altura))
            else:
                imagen_redimensionada = imagen_rgb

            # Convertir a PIL y mostrar
            img_pil = Image.fromarray(imagen_redimensionada)
            #img_tk = ImageTk.PhotoImage(image=img_pil)
            img_tk =  ctk.CTkImage(img_pil, size=(400,400))

            # Label para mostrar la imagen
            imagen_label = ctk.CTkLabel(frame_contenedor, image=img_tk, text="")
            imagen_label.image = img_tk  # Mantener referencia
            imagen_label.pack(padx=15, pady=15)

        except Exception as e:
            error_label = ctk.CTkLabel(
                panel,
                text=f"❌ Error al mostrar imagen: {str(e)}",
                font=ctk.CTkFont(size=14)
            )
            error_label.pack(pady=50)

    def mostrar_mensaje(self, mensaje):
        # Crear ventana de mensaje personalizada
        dialog = ctk.CTkToplevel(self)
        dialog.geometry("400x200")
        dialog.title("Información")
        dialog.transient(self)
        dialog.grab_set()

        # Centrar la ventana
        dialog.geometry("+%d+%d" % (self.winfo_rootx() + 50, self.winfo_rooty() + 50))

        # Contenido del diálogo
        label = ctk.CTkLabel(
            dialog,
            text=mensaje,
            font=ctk.CTkFont(size=14),
            wraplength=350
        )
        label.pack(pady=40, padx=20)

        # Botón OK
        btn_ok = ctk.CTkButton(
            dialog,
            text="OK",
            command=dialog.destroy,
            width=100
        )
        btn_ok.pack(pady=20)

    def limpiar_frame(self,frame):
        for widget in frame.winfo_children:
            widget.destroy()

    def limpiar_pestana(self, nombre_panel: str):
        panel = getattr(self, nombre_panel, None)

        if panel is not None:
            for widget in panel.winfo_children():
                widget.destroy()
        else:
            self.mostrar_mensaje("Error al eliminar frame")

    def limpiar_todas_las_pestanas(self):
        self.limpiar_pestana("panel_basico")
        self.limpiar_pestana("panel_logicas")
        self.limpiar_pestana("panel_ruido")
        self.limpiar_pestana("panel_segmentacion")
        self.limpiar_pestana("panel_objetos")
        self.limpiar_pestana("panel_histogramas")


if __name__ == "__main__":
    app = InterfazProcesadorImagenes()
    app.mainloop()