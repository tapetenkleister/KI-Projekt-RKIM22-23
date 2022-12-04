from __future__ import annotations
import cv2
import os
import numpy as np
from typing import TypeVar

    

def _filter_vertical_lines(image = None) :
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

def vertical_line_extract(image : np.ndarray, debug: bool = False) -> list[int]:
    
    height, _width, _colour_channels = image.shape

    # Converts the BGR color space of the image to the GRAY color space
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
           

    # Threshold for Canny-edge-detection
    lower_thresh = 50
    upper_thresh = 140

    # Use the Canny-edge-detection algorithm to detect edges in the image
    canny = cv2.Canny(gray, lower_thresh, upper_thresh, L2gradient=True)
               

    # Create kernel to only take horizontal lines in the image into account
    h_line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 7))

    # Morph the image with the kenel
    opening = cv2.morphologyEx(canny, cv2.MORPH_OPEN, h_line_kernel, iterations=2)
               

    # Create kernel for dilation task
    hh_line_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, 2*height))

    # Dilate horizontal lines for the whole image width
    dilation = cv2.dilate(opening, hh_line_kernel, iterations=1)
                

    # Filter out horizontal lines that dont have another line below them.
    filtered = _filter_vertical_lines(dilation) 

    # Create the feature list
    feature_list = len(filtered[0, :])

    # Show the processing steps of the image
    if debug:
        cv2.imshow('input', image)
        cv2.imshow('canny', canny)
        cv2.imshow('opening', opening)
        cv2.imshow('filtered', dilation)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return feature_list


