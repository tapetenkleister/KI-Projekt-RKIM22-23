from __future__ import annotations
import cv2
import os
import numpy as np
from typing import TypeVar
from basic_class import BasicImageClass

Image = TypeVar('Image')
ImageLabel = TypeVar('ImageLabel')


class VerticalLineGenerator(BasicImageClass):
    """seal_broken
    """

    def __init__(self, dir_path: str = "data/pictures_tobi_w_timo/pictures", scale_fact: float = 0.1) -> None:
        super().__init__(dir_path, scale_fact)

    def _filter_vertical_lines(self, image: Image = None) -> Image:
        """Filters out horizontal lines that dont have another horizontal line below them.

        Parameters
        ----------
        image : Image, optional
            binary image (beware of the np.logical_and), by default None

        Returns
        -------
        Image
            binary image, filtered
        """
        _height, width = image.shape

        for col_nb in range(width-1):
            image[:, col_nb] = image[:, col_nb] & image[:, col_nb+1]
        return image

    def process_images(self, show_debug_info: bool = False) -> list[list[int]]:
        # Load the images
        images, _labels = self._load_images(
            load_folder=['closed_seal_broken', 'closed_sealed'])
        height, _width, _colour_channels = images[0].shape

        # Converts the BGR color space of the image to the GRAY color space
        gray = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                for image in images.copy()]

        # Threshold for Canny-edge-detection
        lower_thresh = 50
        upper_thresh = 140

        # Use the Canny-edge-detection algorithm to detect edges in the image
        canny = [cv2.Canny(image, lower_thresh, upper_thresh, L2gradient=True)
                 for image in gray.copy()]

        # Create kernel to only take horizontal lines in the image into account
        h_line_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 7))

        # Morph the image with the kenel
        opening = [cv2.morphologyEx(image, cv2.MORPH_OPEN, h_line_kernel, iterations=2)
                   for image in canny.copy()]

        # Create kernel for dilation task
        hh_line_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 2*height))

        # Dilate horizontal lines for the whole image width
        dilation = [cv2.dilate(image, hh_line_kernel, iterations=1)
                    for image in opening.copy()]

        # Filter out horizontal lines that dont have another line below them.
        filtered = [self._filter_vertical_lines(
            image) for image in dilation.copy()]

        # Create the feature list
        feature_list = [image[0, :] for image in filtered.copy()]

        # Show the processing steps of the image
        if show_debug_info:
            for i in range(len(images)):
                cv2.imshow('input', images[i])
                cv2.imshow('canny', canny[i])
                cv2.imshow('opening', opening[i])
                cv2.imshow('filtered', dilation[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return feature_list


if __name__ == '__main__':
    vertical_line_gen = VerticalLineGenerator()
    vertical_line_gen.process_images(show_debug_info=True)
