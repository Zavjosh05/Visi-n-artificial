import os
import cv2
import numpy as np

os.makedirs("dataset", exist_ok=True)

def crear_canvas():
    return np.ones((300, 300, 3), dtype=np.uint8) * 255

def ruido(img):
    return img

def generar_circulo(i):
    img = crear_canvas()
    cv2.circle(img, (150,150), 80, (0,0,0), 4)
    cv2.imwrite(f"dataset/circulo_{i}.png", img)

def generar_cuadrado(i):
    img = crear_canvas()
    cv2.rectangle(img, (70,70), (230,230), (0,0,0), 4)
    cv2.imwrite(f"dataset/cuadrado_{i}.png", img)

def generar_triangulo(i):
    img = crear_canvas()
    pts = np.array([[150,50],[250,250],[50,250]], np.int32)
    cv2.polylines(img, [pts], True, (0,0,0), 4)
    cv2.imwrite(f"dataset/triangulo_{i}.png", img)

def generar_estrella(i):
    img = crear_canvas()
    pts = np.array([
        [150,40],[180,120],[260,120],
        [200,160],[220,240],
        [150,190],[80,240],
        [100,160],[40,120],[120,120]
    ], np.int32)
    cv2.polylines(img, [pts], True, (0,0,0), 4)
    cv2.imwrite(f"dataset/estrella_{i}.png", img)

for i in range(1, 21):
    generar_circulo(i)
    generar_cuadrado(i)
    generar_triangulo(i)
    generar_estrella(i)

print("Dataset generado en carpeta dataset/")
