def aplicar_ajuste_generico(self, funcion, panel, titulo, actualizar=True):
    """
    Aplica cualquier función de ajuste de brillo/contraste y muestra el resultado.

    :param funcion: función a ejecutar (ej: self.ajustes_brillo.correccion_gamma)
    :param panel: panel donde mostrar el resultado
    :param titulo: texto a mostrar en el título
    :param actualizar: si debe actualizar la imagen en memoria
    """
    if self.verificar_imagen_cargada(self.imagen_display[self.indice_actual]) is False:
        return

    try:
        imagen_procesada = funcion(img=self.imagen_display[self.indice_actual])
        if imagen_procesada is not None:
            if actualizar:
                self.imagen_display[self.indice_actual] = imagen_procesada

            self.mostrar_imagen(
                panel,
                imagen_procesada,
                f"{titulo}\nImagen {self.indice_actual+1}"
            )
            self.tabview.set("🔧 Básico")
        else:
            self.mostrar_mensaje(f"Error al aplicar {titulo} a la imagen {self.indice_actual+1}")

    except Exception as e:
        self.mostrar_mensaje(f"❌ Error: {str(e)}")

# botones_config.py

# =====================
# Ajustes de brillo
# =====================
ajustes_botones = [
    ("🔆 Corrección Gamma", "ajustes_brillo.correccion_gamma"),
    ("📊 Ecualización Adaptativa", "ajustes_brillo.ecualizacion_adaptativa"),
]


import botones_config as cfg

# Resolver funciones de ajustes
ajustes_btns = []
for texto, funcion_path in cfg.ajustes_botones:
    modulo, metodo = funcion_path.split(".")
    funcion = getattr(getattr(self, modulo), metodo)

    ajustes_btns.append(
        (texto, lambda f=funcion, t=texto:
            self.aplicar_ajuste_generico(f, self.panel_basico, t))
    )

# Crear sección Ajustes
self.ajustes_frame = SeccionDinamica(
    master=self.sidebar_frame,
    titulo="🔧 Básico",
    botones=ajustes_btns,
    default_color="#004477"
)
self.ajustes_frame.grid(row=4, column=0, padx=20, pady=10, sticky="ew")
