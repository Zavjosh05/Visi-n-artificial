import customtkinter as ctk
from librerias.SeccionDinamica import SeccionDinamica

class DemoApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Demo SeccionDinamica")
        self.geometry("400x600")

        # Sidebar principal
        self.sidebar_frame = ctk.CTkFrame(self, width=250)
        self.sidebar_frame.pack(side="left", fill="y", padx=10, pady=10)

        # ===============================
        # 1. Sección con botones simples
        # ===============================
        self.guardar_frame = SeccionDinamica(
            master=self.sidebar_frame,
            titulo="💾 Guardar Resultado",
            botones=[("Guardar Imagen", self.guardar_imagen)],
            default_color="#229954"
        )
        self.guardar_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # ===============================
        # 2. Sección con subsecciones
        # ===============================
        self.ruido_frame = SeccionDinamica(
            master=self.sidebar_frame,
            titulo="🔊 Ruido y Filtros",
            subsecciones=[
                ("Agregar Ruido:", [
                    ("🧂 Sal y Pimienta", self.ruido_sal_pimienta),
                    ("📡 Gaussiano", self.ruido_gaussiano)
                ], "#001A61"),
                ("Filtros Pasa-bajas:", [
                    ("📈 Promediador", self.filtro_promediador),
                    ("📈 Mediana", self.filtro_mediana)
                ], "#0A4B43"),
                ("Filtros Pasa-altas:", [
                    ("📈 Sobel", self.filtro_sobel),
                    ("📈 Canny", self.filtro_canny)
                ], "#29164A")
            ]
        )
        self.ruido_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        # ===============================
        # 3. Sección en modo mixto
        # ===============================
        self.procesamiento_frame = SeccionDinamica(
            master=self.sidebar_frame,
            titulo="⚙️ Procesamiento",
            botones=[("▶️ Ejecutar Todo", self.ejecutar_todo)],
            subsecciones=[
                ("Preprocesamiento:", [
                    ("🔧 Normalizar", self.preprocesar_normalizar),
                    ("🔧 Escalar", self.preprocesar_escalar)
                ], "#444444")
            ]
        )
        self.procesamiento_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

    # ===============================
    # Funciones de ejemplo
    # ===============================
    def guardar_imagen(self): print("✅ Imagen guardada")
    def ruido_sal_pimienta(self): print("✅ Ruido Sal y Pimienta agregado")
    def ruido_gaussiano(self): print("✅ Ruido Gaussiano agregado")
    def filtro_promediador(self): print("✅ Filtro Promediador aplicado")
    def filtro_mediana(self): print("✅ Filtro Mediana aplicado")
    def filtro_sobel(self): print("✅ Filtro Sobel aplicado")
    def filtro_canny(self): print("✅ Filtro Canny aplicado")
    def ejecutar_todo(self): print("▶️ Ejecutando todo el pipeline...")
    def preprocesar_normalizar(self): print("🔧 Normalizando datos...")
    def preprocesar_escalar(self): print("🔧 Escalando datos...")

if __name__ == "__main__":
    app = DemoApp()
    app.mainloop()
