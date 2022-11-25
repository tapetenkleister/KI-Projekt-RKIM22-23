from __future__ import annotations
import cv2
import os
import sys
import numpy as np
from typing import TypeVar
from basic_class import BasicImageClass

Image = TypeVar('Image')
ImageLabel = TypeVar('ImageLabel')
AspectRatio = TypeVar('AspectRatio')


class AspectRatioGenerator(BasicImageClass):
    def __init__(self, dir_path: str = "data", scale_fact: float = 0.1) -> None:
        super().__init__(dir_path, scale_fact)

    def process_images(self, show_debug_info: bool = False, max_num_images: int = sys.maxsize, load_folder: list[str] = ["all"]) -> tuple[list[int, int, int, int], list[AspectRatio]]:
        """Looks for brown objects in the image and draws a boundingbox around the biggest one. The coordinates of the boundingbox will be returned as well as a list of the aspect ratio of the box.

        Parameters
        ----------
        show_debug_info : bool, optional
            Show the different processing steps of the image, by default False
        max_num_images : int, optional
            Defines the maximum number of images to load, by default sys.maxsize (i.e. all of them)
        load_folder : list, optional
            Defines a list of folders from which the images are loaded, by default ["all"]. 
            More than one folder can be loaded by adding the foldername to the list, e.g. load_folder = ['closed_seal_broken', 'closed_sealed']

        Returns
        -------
        tuple[list[int, int, int, int], list[AspectRatio]]
            A List of the boundingbox coordinates (x, y, w, h) in pixels. A List of the aspect ratios of the found brown object.
        """
        # Load all the images
        images, _labels = self._load_images(
            max_num_images=max_num_images, load_folder=load_folder)

        # Converts the BGR color space of the image to the HSV color space
        hsv = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
               for image in images.copy()]

        # Threshold of brown in HSV space
        lower_brown = np.array([1, 30, 0])
        upper_brown = np.array([40, 255, 150])

        # Find brown shades inside the image and display them as white in front of a black background
        mask = [cv2.inRange(image, lower_brown, upper_brown)
                for image in hsv.copy()]

        # Dilate the image
        kernel_3 = np.ones((3, 3), np.uint8)
        dilation = [cv2.dilate(image, kernel_3, iterations=3)
                    for image in mask.copy()]

        # Dilate and erode the image to close small holes inside the object
        kernel_5 = np.ones((5, 5), np.uint8)
        closing = [cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_5, iterations=6)
                   for image in dilation.copy()]

        # Search the image for contours
        contours = [cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for image in closing.copy()]

        # Get the biggest contour inside the image
        biggest_contours = [max(contour, key=cv2.contourArea)
                            for contour in contours.copy()]

        # Create a black canvas and draw all found contours onto it
        black_canvas = np.zeros(
            (mask[0].shape[0], mask[0].shape[1], 3), dtype=np.uint8)
        contour_pic = [cv2.drawContours(
            black_canvas.copy(), contour, -1, (0, 255, 75), 2) for contour in contours.copy()]

        # Build a bounding box around the biggest contour found in the image
        x_y_w_h = [cv2.boundingRect(contour)
                   for contour in biggest_contours.copy()]

        # Calculate aspect ratio
        aspect_ratio = [(tuple[2] / tuple[3]) for tuple in x_y_w_h]

        # Draw the bounding box
        bounding_boxes = [cv2.rectangle(contour_pic[idx], (x, y), (
            x + w, y + h), (255, 255, 255), 1) for idx, (x, y, w, h) in enumerate(x_y_w_h.copy())]

        # Show the processing steps of the image
        if show_debug_info:
            for i in range(len(images)):
                cv2.imshow('input', images[i])
                cv2.imshow('mask', mask[i])
                cv2.imshow('dilation', dilation[i])
                cv2.imshow('closing', closing[i])
                cv2.imshow('bounding_box', bounding_boxes[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return x_y_w_h, aspect_ratio


if __name__ == '__main__':
    aspect_ratio_gen = AspectRatioGenerator()
    x_y_w_h, aspect_ratio = aspect_ratio_gen.process_images(
        show_debug_info=True)
