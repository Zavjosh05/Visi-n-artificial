import customtkinter as ctk

class SliderWindow(ctk.CTkToplevel):
    def __init__(self, master=None, title="Selecciona un valor", min_val=0, max_val=100, initial_val=50, step=1):
        super().__init__(master)
        self.title(title)
        self.geometry("300x160")
        self.resizable(False, False)

        self.value = None  # Valor final que devolverá el diálogo

        # Etiqueta del título
        self.label = ctk.CTkLabel(self, text=title)
        self.label.pack(pady=(10, 5))

        # Mostrar valor actual
        self.slider_value = ctk.StringVar(value=str(initial_val))
        self.value_label = ctk.CTkLabel(self, textvariable=self.slider_value)
        self.value_label.pack()

        # Slider
        steps = int((max_val - min_val) / step)
        self.slider = ctk.CTkSlider(self, from_=min_val, to=max_val, number_of_steps=steps, command=self.update_value)
        self.slider.set(initial_val)
        self.slider.pack(pady=10)

        # Botón para confirmar
        self.confirm_button = ctk.CTkButton(self, text="Aceptar", command=self.confirm)
        self.confirm_button.pack(pady=(5, 10))

        # Modal
        self.grab_set()
        self.wait_window()

    def update_value(self, value):
        self.slider_value.set(f"{float(value):.2f}")

    def confirm(self):
        self.value = float(self.slider.get())
        self.destroy()
