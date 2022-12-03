from __future__ import annotations
import cv2
import os
import sys
import numpy as np
from typing import TypeVar
from feature_extraction import BeerBottle

def aspect_ratio_extract(image : np.ndarray) :
    """Looks for brown objects in the image and draws a boundingbox around the biggest one. 
    The coordinates of the boundingbox will be returned as well as a list of the aspect ratio of the box.

    Parameters
    ----------
    image : np.ndarray
        The image to which the code applies aspect ratio and bounding box.
    Returns
    -------
    list[int, int, int, int], list[AspectRatio]
        Two returns: A List of the boundingbox coordinates (x, y, w, h) in pixels. 
                     A List of the aspect ratios of the found brown object.
    """
    

    # Converts the BGR color space of the image to the HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
          

    # Threshold of brown in HSV space
    lower_brown = np.array([1, 30, 0])
    upper_brown = np.array([40, 255, 150])

    # Find brown shades inside the image and display them as white in front of a black background
    mask = cv2.inRange(image, lower_brown, upper_brown)
            

    # Dilate the image
    kernel_3 = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(image, kernel_3, iterations=3)
                

    # Dilate and erode the image to close small holes inside the object
    kernel_5 = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_5, iterations=6)
               

    # Search the image for contours
    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Get the biggest contour inside the image
    biggest_contours = max(contours, key=cv2.contourArea)
                        

    # Create a black canvas and draw all found contours onto it
    black_canvas = np.zeros((mask[0].shape[0], mask[0].shape[1], 3), dtype=np.uint8)
    contour_pic = cv2.drawContours(black_canvas.copy(), contours, -1, (0, 255, 75), 2)

    # Build a bounding box around the biggest contour found in the image
    x_y_w_h = cv2.boundingRect(biggest_contours)
                

    # Calculate aspect ratio
    aspect_ratio = (x_y_w_h[2] / x_y_w_h[3])

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


