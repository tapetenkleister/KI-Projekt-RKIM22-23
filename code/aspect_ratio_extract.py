from __future__ import annotations
import cv2
import numpy as np

def aspect_ratio_extract(image : np.ndarray, debug : bool) :
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
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
            

    # Dilate the image
    kernel_3 = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(mask, kernel_3, iterations=3)
                

    # Dilate and erode the image to close small holes inside the object
    kernel_5 = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel_5, iterations=6)
               

    # Search the image for contours
    contours = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Get the biggest contour inside the image
    biggest_contours = max(contours, key=cv2.contourArea)
                        

    # Create a black canvas and draw all found contours onto it
    black_canvas = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    contour_pic = cv2.drawContours(black_canvas.copy(), contours, -1, (0, 255, 75), 2)

    # Build a bounding box around the biggest contour found in the image
    x_y_w_h = cv2.boundingRect(biggest_contours)
    x,y,w,h = x_y_w_h            

    # Calculate aspect ratio
    aspect_ratio = (w / h)

    # Draw the bounding box
    bounding_boxes = cv2.rectangle(contour_pic, (x, y), (x + w, y + h), (255, 255, 255), 1) 

    # Show the processing steps of the image
    if debug:
        cv2.imshow('input', image)
        cv2.imshow('mask', mask)
        cv2.imshow('dilation', dilation)
        cv2.imshow('closing', closing)
        cv2.imshow('bounding_box', bounding_boxes)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x_y_w_h, aspect_ratio


