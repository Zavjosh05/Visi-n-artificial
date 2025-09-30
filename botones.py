#Row: 2, fg_color="#9A721D",hover_color="#000000"
botones_procesamiento = [
    ("🔳 Escala de Grises", "convertir_a_grises"),
    ("📊 Binarizar", "aplicar_umbral"),
    ("📊 Calcular Histogramas", "calcular_histogramas")
]

#Row: 3,fg_color="#445725",hover_color="#000000"
botones_ajustes_de_brillo = [
    ("🔳 Ecualización estandar", "aplicar_ecualizacion_estandar"),
    ("📈 Ecualización Hipercúbica", "ecualizacion_hipercubica"),
    ("📊 Corrección Gamma", "aplicar_correccion_gamma"),
    ("📈 Expansión lineal de contraste", "aplicar_expansion_lineal"),
    ("📊 Transformación exponencial", "aplicar_transformacion_exponencial"),
    ("📈 Ecualización adaptativa","aplicar_ecualizacion_adaptativa"),
    ("📈 Transformación rayleigh","aplicar_transformacion_rayleigh")
]

#Row: 4
subsecciones_operaciones_aritmeticas_y_logicas = [
    ("Aritmeticas", [
        ("🔳 Suma", "aplicar_suma_gui"),
        ("📊 Resta", "aplicar_resta_gui"),
        ("🧮 Multiplicación", "aplicar_multiplicacion_gui"),
    ],"#11500C"),
    ("Lógicas", [
            ("🔳 AND", "aplicar_and_gui"),
            ("📊 OR", "aplicar_or_gui"),
            ("🧮 XOR", "aplicar_xor_gui"),
            ("🧮 NOT", "aplicar_not_gui")
    ],"#11500C")
]

#Row: 5,  fg_color="#001A61",hover_color="#000000"
botones_ruido = [
    ("🧂 Sal y Pimienta", "agregar_ruido_sal_pimienta"),
    ("📡 Gaussiano", "agregar_ruido_gaussiano")
]

#Row: 6, fg_color= "#0A4B43",hover_color="#000000"
botones_filtros_pasa_bajas = [
    ("📈 Promediador", "aplicar_filtro_promediador"),
    ("📈 Pesado", "aplicar_filtro_pesado"),
    ("📈 Mediana", "aplicar_filtro_mediana"),
    ("📈 Moda", "aplicar_filtro_Moda"),
    ("📈 Bilateral","aplicar_filtro_bilateral"),
    ("📈 Max","aplicar_filtro_max"),
    ("📈 Min","aplicar_filtro_min"),
    ("📈 Gaussiano", "aplicar_filtro_gaussiano")
]

#Row: 7, fg_color= "#0A4B43", hover_color="#000000"
botones_filtros_pasa_altas = [
    ("📈 Robinson", "aplicar_filtro_Robinson"),
    ("📈 Robert", "aplicar_filtro_Robert"),
    ("📈 Prewitt", "aplicar_filtro_Prewitt"),
    ("📈 Sobel", "aplicar_filtro_Sobel"),
    ("📈 Kirsch","aplicar_filtro_Kirch"),
    ("📈 Canny","aplicar_filtro_Canny"),
    ("📈 Op. Laplaciano","aplicar_Operador_Laplaciano")
]

#Row: 8, fg_color="#631D29",hover_color="#000000"
botones_segmentacion = [
    ("🎯 Umbral Media", "aplicar_umbral_media"),
    ("🎯 Método de Otsu", "aplicar_filtro_otsu"),
    ("🎯 Multiumbralización", "aplicar_multiubralizacion"),
    ("🎯 Entropía Kapur", "aplicar_entropia_kapur"),
    ("🎯 Umbral por banda", "aplicar_umbral_banda"),
    ("🎯 Umbral adaptativo", "aplicar_umbral_adaptativo"),
    ("🎯 Minimo del histograma", "aplicar_minimo_en_el_histograma"),
    ("🎯 Vecindad 4", "aplicar_vecindad_4"),
    ("🎯 Vecindad 8", "aplicar_vecindad_8"),
    ("📏 Análisis de Objetos", "aplicar_analisis_objetos")
]

#Row: 8 - sintaxis: (nombre del boton, clase(delcarada anteriormente).metodo,tabview,panel,texto que quieran que aparezca)
botones_vision = [
    ("Mascaras de Kirsch","vision.mascaras_kirsch","✂️ Segmentación","panel_segmentacion","Mascaras de kirsch"),
    ("Sobel","vision.sobel","✂️ Segmentación","panel_segmentacion","Sobel"),
    ("Operador de Roberts","vision.roberts","✂️ Segmentación","panel_segmentacion","Operador de Roberts"),
    ("Método de Frei-chen","vision.freichen","✂️ Segmentación","panel_segmentacion","Método de Frei-chen"),
    ("Canny","vision.canny","✂️ Segmentación","panel_segmentacion","Canny"),
    ("Prewitt","vision.prewit","✂️ Segmentación","panel_segmentacion","Prewitt"),
]