import customtkinter as ctk

class SeccionDinamica(ctk.CTkFrame):
    def __init__(self, master, titulo="Sección",
                 botones=None, subsecciones=None,
                 font_size=16, boton_height=30,
                 default_color="#229954", hover_color="#000000",
                 *args, **kwargs):
        super().__init__(master, *args, **kwargs)

        # Título de la sección principal
        self.titulo_label = ctk.CTkLabel(
            self,
            text=titulo,
            font=ctk.CTkFont(size=font_size, weight="bold")
        )
        self.titulo_label.grid(row=0, column=0, padx=20, pady=(15, 10))

        fila_actual = 1

        # === Botones directos (sin subsección) ===
        if botones:
            for texto, comando in botones:
                btn = ctk.CTkButton(
                    self,
                    text=texto,
                    command=comando,
                    height=boton_height,
                    fg_color=default_color,
                    hover_color=hover_color
                )
                btn.grid(row=fila_actual, column=0, padx=20, pady=5, sticky="ew")
                fila_actual += 1

        # === Subsecciones ===
        if subsecciones:
            for sub_titulo, botones_sub, color in subsecciones:
                # Subtítulo
                if sub_titulo:  # opcional
                    sub_label = ctk.CTkLabel(
                        self,
                        text=sub_titulo,
                        font=ctk.CTkFont(size=font_size, weight="bold")
                    )
                    sub_label.grid(row=fila_actual, column=0, padx=20, pady=(10, 5))
                    fila_actual += 1

                # Botones de la subsección
                for texto, comando in botones_sub:
                    btn = ctk.CTkButton(
                        self,
                        text=texto,
                        command=comando,
                        height=boton_height,
                        fg_color=color if color else default_color,
                        hover_color=hover_color
                    )
                    btn.grid(row=fila_actual, column=0, padx=20, pady=3, sticky="ew")
                    fila_actual += 1

        # Espaciado final
        ctk.CTkLabel(self, text="").grid(row=fila_actual, column=0, pady=(0, 15))

    def cargar_subsecciones(self, subsecciones_config, panel):
        """
        Convierte la configuración de subsecciones en botones dinámicos.

        :param subsecciones_config: lista con (titulo_subseccion, lista_botones, color)
        :param panel: panel donde mostrar los resultados
        :return: lista de subsecciones con botones convertidos
        """
        subsecciones_resultado = []

        for sub_titulo, botones, color in subsecciones_config:
            botones_convertidos = []
            for texto, funcion_path, tab in botones:
                modulo, metodo = funcion_path.split(".")
                funcion = getattr(getattr(self, modulo), metodo)

                requiere_obj = "vecindad" in metodo.lower()  # ejemplo para casos especiales
                botones_convertidos.append(
                    (texto, lambda f=funcion, t=texto, tb=tab, r=requiere_obj:
                        self.aplicar_funcion_generica(f, panel, t, tabview=tb, requiere_objetos=r))
                )

            subsecciones_resultado.append((sub_titulo, botones_convertidos, color))

        return subsecciones_resultado