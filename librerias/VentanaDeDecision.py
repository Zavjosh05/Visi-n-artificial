import customtkinter as ctk

class VentanaDeDecision(ctk.CTkToplevel):
    def __init__(self, master=None, title="Nada", mainText="Nada", 
                 firstButton="Nada", secondButton="Nada", command1=None, command2=None):
        super().__init__(master)
        self.title(title)
        self.geometry("300x150")
        self.resizable(False, False)

        self.label = ctk.CTkLabel(self, text=mainText)
        self.label.pack(pady=20)

        # Frame contenedor para los botones
        botones_frame = ctk.CTkFrame(self)
        botones_frame.pack(pady=10)

        # Botones en una sola fila
        self.boton1 = ctk.CTkButton(botones_frame, text=firstButton, command=lambda: self.ejecutar_y_cerrar(command1))
        self.boton1.pack(side="left", padx=10)

        self.boton2 = ctk.CTkButton(botones_frame, text=secondButton, command=lambda: self.ejecutar_y_cerrar(command2))
        self.boton2.pack(side="left", padx=10)

    def ejecutar_y_cerrar(self, funcion):
        if funcion:
            funcion()
        self.destroy()
