import numpy as np
import cv2 as cv

class Vision:
    def __init__(self):
        return

    def mascaras_kirsch(self,img):
        kirsch_masks = [
            np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # Norte
            np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # Noreste
            np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # Este
            np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # Sureste
            np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # Sur
            np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # Suroeste
            np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # Oeste
            np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])  # Noroeste
        ]

        edges = np.zeros_like(img, dtype=np.uint8)
        for mask in kirsch_masks:
            filtered = cv.filter2D(img, cv.CV_16S, mask)
            abs_filtered = cv.convertScaleAbs(filtered)
            edges = np.maximum(edges, abs_filtered)

        return img