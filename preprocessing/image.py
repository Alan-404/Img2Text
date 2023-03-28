import numpy as np
import cv2 as cv


class ImageProcessor:
    def __init__(self) -> None:
        pass

    def process(self, items: list):
        images = []
        for item in  items:
            image = cv.imread(item)
            image = cv.resize(image, (512, 512))
            images.append(np.array(image))

        return np.array(images)