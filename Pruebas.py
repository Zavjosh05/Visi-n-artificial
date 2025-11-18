import cv2
from librerias.Vision import *  # Asegúrate de que Vision.py esté en la misma carpeta

def main():
    # Cargar imágenes
    imagen = cv2.imread("imgs/img_mario.jpg")
    template = cv2.imread("imgs/template_mario.jpg")

    if imagen is None:
        print("Error: no se encontró 'imagen.png'")
        return
    if template is None:
        print("Error: no se encontró 'template.png'")
        return

    # Crear objeto Vision
    v = Vision()

    # Ejecutar template matching manual
    resultado = v.template_matching_manual(imagen, template)

    # Mostrar resultado
    cv2.imshow("Resultado Template Matching Manual", resultado)

    print("Presiona cualquier tecla para salir...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if __name__ == "__main__":
        main()