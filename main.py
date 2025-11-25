import functools

import matplotlib.pyplot as plt
import customtkinter as ctk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import filedialog, messagebox
import os
from PIL import Image, ImageTk
from joblib import dump

from botones import botones_procesamiento, botones_ajustes_de_brillo, subsecciones_operaciones_aritmeticas_y_logicas, \
    botones_filtros_pasa_altas
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
from librerias.Vision import *
from librerias.SeccionDinamica import *
import botones as btn


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

        # Configuracion grid principal
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.procesador = ProcesadorImagen()
        self.operaciones_logicas = OperacionesLogicas2()
        self.ruido = Ruido()
        self.filtro = Filtros()
        self.ajustes_brillo = AjustesDeBrillo()
        self.filtros_pasa_altas = FiltrosPasaAltas()
        self.umbralizacion = Umbralizacion()
        self.vision = Vision()

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
        self.crear_panel_principal()

        botones_procesamiento = [(texto,getattr(self,comando)) for texto, comando in btn.botones_procesamiento]
        botones_ajustes_de_brillo = [(texto,getattr(self,comando)) for texto, comando in btn.botones_ajustes_de_brillo]
        subsecciones_operaciones_aritmeticas_y_logicas = [(sub, [(txt, getattr(self, comando)) for txt, comando in botones], color) for sub, botones, color in btn.subsecciones_operaciones_aritmeticas_y_logicas]
        botones_ruido = [(texto,getattr(self,comando)) for texto, comando in btn.botones_ruido]
        botones_filtros_pasa_bajas = [(texto,getattr(self,comando)) for texto, comando in btn.botones_filtros_pasa_altas]
        botones_filtros_pasa_altas = [(texto,getattr(self,comando)) for texto, comando in btn.botones_filtros_pasa_bajas]
        #botones_segmentacion = [(texto,getattr(self,comando)) for texto, comando in btn.botones_segmentacion]

        botones_vision = self.cargar_botones(btn.botones_vision)
        botones_vision_dos = self.cargar_botones(btn.botones_vision_dos)



        # Sección de carga de imágenes
        self.crear_seccion_carga()

        seccion_procesamiento = SeccionDinamica(
            master=self.sidebar_frame,
            titulo="⚙️ Procesamiento Básico",
            botones=botones_procesamiento,
            default_color="#9A721D",
            hover_color="#000000"
        )
        seccion_procesamiento.grid(row=2, column=0, padx=20, pady=(10, 5))

        # seccion_ajustes_de_brillo = SeccionDinamica(
        #     master=self.sidebar_frame,
        #     titulo="🔦 Ajustes de brillo",
        #     botones=botones_ajustes_de_brillo,
        #     default_color="#445725",
        #     hover_color="#000000"
        # )
        # seccion_ajustes_de_brillo.grid(row=3, column=0, padx=20, pady=(20, 10))

        # seccion_operaciones = SeccionDinamica(
        #     master=self.sidebar_frame,
        #     titulo="🔗 Operaciones",
        #     subsecciones=subsecciones_operaciones_aritmeticas_y_logicas
        # )
        # seccion_operaciones.grid(row=4, column=0, padx=20, pady=(20, 10))

        # seccion_ruido = SeccionDinamica(
        #     master=self.sidebar_frame,
        #     titulo="🔊 Ruido",
        #     botones=botones_ruido,
        #     default_color="#001A61",
        #     hover_color="#000000"
        # )
        # seccion_ruido.grid(row=5, column=0, padx=20, pady=(20, 10))

        # seccion_filtros_pasa_bajas = SeccionDinamica(
        #     master=self.sidebar_frame,
        #     titulo="Filtros pasa_baja",
        #     botones=botones_filtros_pasa_bajas,
        #     default_color="#0A4B43",
        #     hover_color="#000000"
        # )
        # seccion_filtros_pasa_bajas.grid(row=6, column=0, padx=20, pady=(20, 10))

        # seccion_filtros_pasa_altas = SeccionDinamica(
        #     master=self.sidebar_frame,
        #     titulo="Filtros pasa_altas",
        #     botones=botones_filtros_pasa_altas,
        #     default_color="#0A4B43",
        #     hover_color="#000000"
        # )
        # seccion_filtros_pasa_altas.grid(row=7, column=0, padx=20, pady=(20, 10))

        seccion_vision = SeccionDinamica(
            master=self.sidebar_frame,
            titulo="Vision",
            botones=botones_vision,
            default_color="#0A4B43",
            hover_color="#000000",
        )
        seccion_vision.grid(row=3, column=0, padx=20, pady=(10, 10))

        seccion_vision_dos = SeccionDinamica(
            master=self.sidebar_frame,
            titulo="Vision 2",
            botones=botones_vision_dos,
            default_color="#0A4B43",
            hover_color="#000000",
        )
        seccion_vision_dos.grid(row=4, column=0, padx=20, pady=(10, 10))

        #seccion_segmentacion = SeccionDinamica(
        #     master=self.sidebar_frame,
        #     titulo="✂️ Segmentación",
        #     botones=botones_segmentacion,
        #     default_color="#631D29",
        #     hover_color="#000000"
        # )
        # seccion_segmentacion.grid(row=8, column=0, padx=20, pady=(20, 10))

        # Botón de guardar
        self.crear_seccion_guardar()

        # Panel principal con pestañas
        #self.crear_panel_principal()

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

    def crear_seccion_guardar(self):
        # Frame para guardar
        self.guardar_frame = ctk.CTkFrame(self.sidebar_frame)
        self.guardar_frame.grid(row=9, column=0, padx=(20, 20), pady=10, sticky="ew")

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
            self.tabview.set("🧊 Detección de objetos")
        elif panel == self.panel_histogramas:
            self.tabview.set("📊 Histogramas")

    def crear_selector_tema(self):
        # Frame para selector de tema
        self.tema_frame = ctk.CTkFrame(self.sidebar_frame)
        self.tema_frame.grid(row=10, column=0, padx=(20, 20), pady=10, sticky="ew")

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
                    # img = cv2.resize(img, (400,400))
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
            self.mostrar_imagen(self.panel_basico, imagen_binarizada, f"Imagen {self.indice_actual+1} binarizada\nUmbral de: {umbral}")
            self.tabview.set("🔧 Básico")
        except Exception as e:
            self.mostrar_mensaje(f"❌ Error: {str(e)}")

    def otsu(self):
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return

        try:
            imagen_gris = self.procesador.convertir_a_grises(self.imagen_display[self.indice_actual])

            umbral, imagen_binarizada = cv2.threshold(imagen_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            self.imagen_display[self.indice_actual] = imagen_binarizada
            self.mostrar_imagen(
                self.panel_basico,
                imagen_binarizada,
                f"Imagen {self.indice_actual + 1} binarizada con Otsu\nUmbral de: {umbral}"
            )
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

    def wrap_funcion(self, funcion):
        def wrapper(*args, **kwargs):
            resultado = funcion(*args, **kwargs)
            
            # Para template_match_with_location que devuelve (imagen, coordenadas)
            # Solo necesitamos la imagen para mostrar en funciones normales
            if isinstance(resultado, tuple) and len(resultado) == 2:
                if hasattr(resultado[0], 'shape'):  # Si el primer elemento es una imagen
                    return resultado[0]  # Devolver solo la imagen
            elif isinstance(resultado, tuple):
                return resultado[0]
            
            return resultado
        return wrapper

    def cargar_botones(self, botones_config):
        """
        Convierte la configuración de botones planos en funciones dinámicas.
        """
        botones_resultado = []

        for texto, funcion_path, tab, panel, mensaje in botones_config:
            partes = funcion_path.split(".")
            metodo = partes[-1]
            ruta_modulos = partes[:-1]

            try:
                obj = functools.reduce(lambda o, attr: getattr(o, attr), ruta_modulos, self)
                funcion_original = getattr(obj, metodo)
            except AttributeError as e:
                print(f"❌ No se pudo resolver '{funcion_path}' -> {e}")
                continue

            # Envolver la función para limpiar el retorno
            funcion_envuelta = self.wrap_funcion(funcion_original)

            # Determinar si es template matching
            es_template = "template" in metodo.lower()
            requiere_obj = "vecindad" in metodo.lower()
    
            botones_resultado.append(
                (
                    texto,
                    lambda f=funcion_envuelta, fo=funcion_original, t=mensaje,
                           tb=tab, p=panel, r=requiere_obj, et=es_template:
                    self.aplicar_funcion_generica(f, fo, p, t, tabview=tb,
                                                  requiere_objetos=r, es_template=et)
                )
            )

        return botones_resultado

    def cargar_subsecciones(self, subsecciones_config):
        """
        Convierte la configuración de subsecciones (con varios grupos de botones)
        en estructuras listas para usar con SeccionDinamica.
        Aplica el wrapper automáticamente a las funciones.
        """
        subsecciones_resultado = []

        for sub_titulo, botones, color in subsecciones_config:
            botones_convertidos = []

            for texto, funcion_path, tab, panel, mensaje in botones:
                partes = funcion_path.split(".")
                metodo = partes[-1]
                ruta_modulos = partes[:-1]

                try:
                    obj = functools.reduce(lambda o, attr: getattr(o, attr), ruta_modulos, self)
                    funcion = getattr(obj, metodo)
                except AttributeError as e:
                    print(f"❌ No se pudo resolver '{funcion_path}' -> {e}")
                    continue

                # Aplicar wrapper
                funcion_envuelta = self.wrap_funcion(funcion)
                requiere_obj = "vecindad" in metodo.lower()

                botones_convertidos.append(
                    (
                        texto,
                        lambda f=funcion_envuelta, t=mensaje, tb=tab, p=panel, r=requiere_obj:
                        self.aplicar_funcion_generica(f, p, t, tabview=tb, requiere_objetos=r)
                    )
                )

            subsecciones_resultado.append((sub_titulo, botones_convertidos, color))

        return subsecciones_resultado
    
    def aplicar_funcion_generica(self, funcion_envuelta, funcion_original, panel, titulo,
                                tabview=None, actualizar=True, requiere_objetos=False, es_template=False):
        """
        Aplica cualquier funcion de procesamiento y muestra el resultado.
        """
        if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
            return

        try:
            panel_obj = getattr(self, panel)

            # Caso especial para Template Matching - requiere imagen y template
            if es_template:
                if self.imagen_1 is None or self.imagen_2 is None:
                    self.mostrar_mensaje("Para Template Matching se necesitan ambas imagenes cargadas")
                    return

                # Usar la imagen seleccionada como imagen principal
                if self.indice_actual == 0:
                    imagen_principal = self.imagen_display[0]
                    template = self.imagen_display[1]
                else:
                    imagen_principal = self.imagen_display[1] 
                    template = self.imagen_display[0]

                # Usar la funcion ORIGINAL (no la envuelta) para pasar ambos parametros
                resultado = funcion_original(img=imagen_principal, template=template)

                # Manejar diferentes tipos de retorno
                titulo_final = titulo
                if isinstance(resultado, tuple) and len(resultado) > 1:
                    imagen_procesada = resultado[0]
                    
                    # Si es template_match_with_location, mostrar coordenadas en el titulo
                    if "location" in funcion_original.__name__.lower() and len(resultado) == 2:
                        coords = resultado[1]
                        titulo_final = f"{titulo}\nCoordenadas: x={coords[0]}, y={coords[1]}, w={coords[2]}, h={coords[3]}"
                else:
                    imagen_procesada = resultado

                if imagen_procesada is not None:
                    self.mostrar_imagen(
                        panel_obj,
                        imagen_procesada,
                        f"{titulo_final}\nImagen {self.indice_actual + 1}"
                    )
                    if tabview:
                        self.tabview.set(tabview)
                else:
                    self.mostrar_mensaje(f"Error al aplicar {titulo}")

            # Caso especial para Harris no actualizar la imagen original
            elif "harris" in funcion_original.__name__.lower():
                imagen_procesada = funcion_envuelta(img=self.imagen_display[self.indice_actual])
                if imagen_procesada is not None:
                    self.mostrar_imagen(
                        panel_obj,
                        imagen_procesada,
                        f"{titulo}\nImagen {self.indice_actual + 1}"
                    )
                    if tabview:
                        self.tabview.set(tabview)
                else:
                    self.mostrar_mensaje(f"Error al aplicar {titulo} a la imagen {self.indice_actual + 1}")

            # Caso para funciones que devuelven (imagen, cantidad_objetos)
            elif requiere_objetos:
                resultado = funcion_envuelta(img=self.imagen_display[self.indice_actual])
                if isinstance(resultado, tuple) and len(resultado) == 2:
                    imagen_procesada, cantidad_objetos = resultado
                    if imagen_procesada is not None:
                        if actualizar:
                            self.imagen_display[self.indice_actual] = imagen_procesada

                        self.mostrar_imagen(
                            panel_obj,
                            imagen_procesada,
                            f"{titulo}\n{cantidad_objetos} objetos detectados\nImagen {self.indice_actual + 1}"
                        )
                        if tabview:
                            self.tabview.set(tabview)
                    else:
                        self.mostrar_mensaje(f"Error al aplicar {titulo} a la imagen {self.indice_actual + 1}")
                else:
                    self.mostrar_mensaje(f"Error: La funcion no devolvio el formato esperado (imagen, cantidad)")

            # Caso general para otras funciones
            else:
                imagen_procesada = funcion_envuelta(img=self.imagen_display[self.indice_actual])
                if imagen_procesada is not None:
                    if actualizar:
                        self.imagen_display[self.indice_actual] = imagen_procesada

                    self.mostrar_imagen(
                        panel_obj,
                        imagen_procesada,
                        f"{titulo}\nImagen {self.indice_actual + 1}"
                    )
                    if tabview:
                        self.tabview.set(tabview)
                else:
                    self.mostrar_mensaje(f"Error al aplicar {titulo} a la imagen {self.indice_actual + 1}")

        except Exception as e:
            self.mostrar_mensaje(f"Error: {str(e)}")

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
                imagen_redimensionada = imagen_rgb
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

def cargarDataset(ruta):
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

def entrenarClasificador():

    imgs, labels = cargarDataset('dataset')

    X = []
    y = []

    for img, label in zip(imgs, labels):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).flatten()

        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

        X.append(hu)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    from sklearn.neighbors import KNeighborsClassifier
    clf = KNeighborsClassifier(5000)
    clf.fit(X, y)

    return clf


if __name__ == "__main__":

    app = InterfazProcesadorImagenes()
    app.mainloop()