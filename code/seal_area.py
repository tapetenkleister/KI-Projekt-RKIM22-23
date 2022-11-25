from __future__ import annotations
import cv2
import os
import sys
import numpy as np
import pandas as pd
from typing import TypeVar
from basic_class import BasicImageClass
from aspect_ratio import AspectRatioGenerator

Image = TypeVar('Image')
ImageLabel = TypeVar('ImageLabel')


class SealContourAreaGenerator(BasicImageClass):
    """Calculates the area of the biggest visible contour inside a image of the bottle neck. To truely get the contour area of the label on the bottle neck, it needs to be in frame.
    If no contour is found the area will default to float(0.0). That way it can be differentiated between 'closed_sealed' (label visible) and 'closed_sealed' (label not visible),
    'closed_seal_broken', 'open_broken', 'broken'.

    Parameters
    ----------
    BasicImageClass :
        inherits the _load_images- and _crop_image-methode
    """

    def __init__(self, dir_path: str = "data/pictures_tobi_w_timo/pictures", scale_fact: float = 0.1) -> None:
        super().__init__(dir_path, scale_fact)

    def process_images(self, show_debug_info: bool = False, max_num_images: int = sys.maxsize, load_folder: list[str] = ["all"]) -> list[float]:
        """Crops to the bottle neck inside the image, by using the bounding box coordinates from the AspectRatioGenerator-class. After that
        all shades of brown and black get filtered out. The area of the biggest remaining contour is calculated and appended to a list, which will be returned
        by this method. If no contour is present a float(0.0) is appended to the list.

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
        list[float]
            A List of the contour areas of the labels.
        """
        aspect_ration_gen = AspectRatioGenerator()

        x_y_w_h = aspect_ration_gen.process_images(
            max_num_images=max_num_images, load_folder=load_folder)[0]
        # Because the _load_image-methode was run in the AspectRatioGenerator.process_images-methode
        # we can save some time by writing: (instead of self._load_images)
        images = aspect_ration_gen.images_inh
        labels = aspect_ration_gen.labels_inh
        # It also ensures, that the same images are loaded.

        # Crop only the bottle necks, where the label is to be expected
        crpd_images = [self._crop_image(image, x=x_y_w_h[idx][0], y=x_y_w_h[idx][1]+0.1*x_y_w_h[idx][3], w=x_y_w_h[idx][2], h=0.3*x_y_w_h[idx][3])
                       for idx, image in enumerate(images.copy())]

        # Calculate the area of the croped images
        crpd_areas = [image.shape[0] * image.shape[1]
                      for image in crpd_images.copy()]

        # Converts the BGR color space of the image to the HSV color space
        hsv_images = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                      for image in crpd_images.copy()]

        # Threshold of everything but brown in HSV space
        lower_brown = np.array([15, 80, 0])
        upper_brown = np.array([179, 255, 255])

        # Find brown shades inside the image and display them as white in front of a black background
        masked_images = [cv2.inRange(image, lower_brown, upper_brown)
                         for image in hsv_images.copy()]

        # Erode and then dilate the image to remove small points outside the object
        kernel_7 = cv2.getStructuringElement(
            cv2.MORPH_RECT, (7, 7))
        opening = [cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel_7, iterations=1)
                   for image in masked_images.copy()]

        # Dilate and erode the image to close small holes inside the object
        closing = [cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_7, iterations=2)
                   for image in opening.copy()]

        # Search the image for contours
        contours = [cv2.findContours(
            image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] for image in closing.copy()]

        # Get the biggest contour inside the image
        biggest_contours = [max(contour, key=cv2.contourArea)
                            if contour else np.ndarray([]) for contour in contours.copy()]

        # Create a black canvas and draw all found contours onto it
        """ black canvas needs to resize accordingly """
        black_canvas = np.zeros(
            (masked_images[0].shape[0], masked_images[0].shape[1], 3), dtype=np.uint8)
        contour_images = [cv2.drawContours(
            black_canvas.copy(), [contour], -1, (0, 255, 75), 1) if contour.shape else black_canvas for contour in biggest_contours.copy()]

        # Calculate the contour-area
        areas = [cv2.contourArea(
            contour) if contour.shape else 0.0 for contour in biggest_contours.copy()]

        # Normalize area
        areas_normalized = [area / crpd_areas[idx]
                            for idx, area in enumerate(areas)]

        # Create dataframe
        area_df = pd.DataFrame(data=zip(areas_normalized, labels),
                               columns=["area", "label"])

        # Plot dataframe
        self._displot_df(area_df, x="area")

        # Show the processing steps of the image
        if show_debug_info:
            for i in range(len(images)):
                cv2.imshow('input_images', images[i])
                cv2.imshow('crpd_images', crpd_images[i])
                cv2.imshow('masked_images', masked_images[i])
                cv2.imshow('opening', opening[i])
                cv2.imshow('closing', closing[i])
                cv2.imshow('contour_images', contour_images[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        return areas_normalized


if __name__ == '__main__':
    seal_contour_area_gen = SealContourAreaGenerator()
    seal_contour_area_gen.process_images()
    # seal_contour_area_gen.process_images(show_debug_info=False, load_folder=[
    #     'closed_seal_broken', 'closed_sealed'])
