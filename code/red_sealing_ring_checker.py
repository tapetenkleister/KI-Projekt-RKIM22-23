from __future__ import annotations
import cv2
import numpy as np
import sys
from basic_class import BasicImageClass
from aspect_ratio import AspectRatioGenerator


class RedSealingRingChecker(BasicImageClass):
    '''
    Handles a list images, searching for red pixels on the upper end of the pre-calculated bounding boxes representing a beer bottle.
    '''

    def __init__(self, dir_path: str = "data/pictures_tobi_w_timo/pictures", scale_fact: float = 0.1) -> None:
        super(RedSealingRingChecker, self).__init__(dir_path, scale_fact)

    def process_images(self, box_height_percent: float = 0.075, threshold: int = 6, max_num_images: int = sys.maxsize, show_debug_info: bool = False) -> list[int]:
        """Searches for red pixels on the upper end of a bounding box in an image. If the the red pixels in the region of interest
        surpasses a preset threshold an integer 1 is returned. Otherwise a interger 0 is returned.
        ----------
        box_height_percent : float, optional
            Percent of the boxheight used for the ROI, by default 0.075 (7.5 percent) of the height of the bounding box.
        threshold : int, optional
            Threshold that needs to be surpassed by the mean of the ROI, by default 6.
        max_num_images : int, optional
            Sets the maximum number of images loaded into the class to speed up the debugging process, by default sys.maxsize (all pictures).
        show_debug_info : bool, optional
            Show the different processing steps of the image, by default False

        Returns
        -------
        list[int]
            A List of integers. Value 1 indicating a red sealing ring found. Value 0 representing the absence of red pixels in the ROI or
            not enough red pixels in the ROI.
        """

        aspect_ratio_gen = AspectRatioGenerator()

        x_y_w_h = aspect_ratio_gen.process_images(
            max_num_images=max_num_images)[0]

        # Load all the images
        images = aspect_ratio_gen.images_inh

        # Converts the BGR color space of the image to the HSV color space
        hsv = [cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
               for image in images.copy()]

        # Create a red color mask
        lower_red = np.array([0, 136, 111])
        upper_red = np.array([6, 255, 200])

        # Find red shades inside the image and display them as white in front of a black background
        masks = [cv2.inRange(image, lower_red, upper_red)
                 for image in hsv.copy()]

        # Erode the image
        kernel2 = np.ones((2, 2), np.uint8)
        eroding = [cv2.erode(mask, kernel2, iterations=1)
                   for mask in masks.copy()]

        # Perform a Closing action on every image
        kernel3 = np.ones((3, 3), np.uint8)
        closing = [cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
                   for img in eroding.copy()]

        # Dilate every image
        kernel5 = np.ones((5, 5), np.uint8)
        dilation = [cv2.dilate(img, kernel5, iterations=1)
                    for img in closing.copy()]

        # Initialise the list later to be returned
        sealing_ring_found = []

        i = 0
        for pixels, (x, y, w, h) in zip(dilation, x_y_w_h):

            # Check the avarage of the pixels around the given upper edge of the box

            if show_debug_info:
                cv2.imshow("Red Areas", pixels)
                cv2.imshow("Input", images[i])
                cv2.imshow("Mask", masks[i])
                cv2.imshow("Erosion", eroding[i])
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                i += 1

            roi = pixels[int(y):int(y+(box_height_percent*h)), int(x):int(x+w)]

            if np.mean(roi) >= threshold:
                sealing_ring_found.append(1)
                if show_debug_info:
                    print("Ring found!")
            else:
                sealing_ring_found.append(0)

        # Return list
        return sealing_ring_found


if __name__ == '__main__':
    aspect_ratio_gen = RedSealingRingChecker()
    x_y_w_h, aspect_ratio = aspect_ratio_gen.process_images(
        show_debug_info=True)
