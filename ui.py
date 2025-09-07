import customtkinter as ctk

# ===============================
#  Paleta de colores 
# ===============================
COLORES = {
    "fondo_principal": "#605B56",
    "sidebar": "#837A75",
    "boton": "#ACC18A",
    "texto": "#DAFEB7",
    "acento": "#F2FBE0"
}
# ===============================

def aplicar_filtro(nombre):
    print(f"Aplicando filtro: {nombre}")

# Diccionario  de filtros
filtros = {
    "Canny": lambda: aplicar_filtro("Canny"),
    "Sobel": lambda: aplicar_filtro("Sobel"),
    "Otsu": lambda: aplicar_filtro("Otsu"),
    "Bilateral": lambda: aplicar_filtro("Bilateral"),
}

# Configuración inicial de  app
ctk.set_appearance_mode("dark") 
ctk.set_default_color_theme("green")  

root = ctk.CTk()
root.title("Proyecto")
root.geometry("900x600")
root.configure(fg_color=COLORES["fondo_principal"])

# Sidebar 
sidebar = ctk.CTkFrame(root, fg_color=COLORES["sidebar"], corner_radius=10)
sidebar.pack(side="left", fill="y", padx=10, pady=10)

# Generar botones 
for nombre, funcion in filtros.items():
    boton = ctk.CTkButton(
        sidebar, 
        text=nombre, 
        command=funcion,
        fg_color=COLORES["boton"],
        hover_color=COLORES["acento"],
        text_color=COLORES["texto"],
        corner_radius=8
    )
    boton.pack(fill="x", pady=5)

# Área central  imagen
area_imagen = ctk.CTkLabel(
    root, 
    text="Aquí se mostrará la imagen",
    fg_color=COLORES["sidebar"],
    text_color=COLORES["texto"],
    corner_radius=10
)
area_imagen.pack(expand=True, fill="both", padx=10, pady=10)

root.mainloop()
