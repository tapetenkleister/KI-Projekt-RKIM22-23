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
          
    blur = cv2.blur(hsv,(7,7))
    # Threshold of dark objects in HSV space H S V
    lower_brown = np.array([1, 20, 0])
    upper_brown = np.array([180, 255, 95])

    # Find brown shades inside the image and display them as white in front of a black background
    mask = cv2.inRange(hsv, lower_brown, upper_brown)
            

    # Dilate the image
    kernel_3 = np.ones((3, 3), np.uint8)
    #dilation = cv2.dilate(mask, kernel_3, iterations=3)
                
    


    # Dilate and erode the image to close small holes inside the object
    kernel_5 = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_5, iterations=7)
    
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel_3, iterations=1)          

    # Search the image for contours
    contours = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

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

    
    # Show the processing steps of the image
    def showInMovedWindow(winname, img, x, y):
        cv2.namedWindow(winname,cv2.WINDOW_NORMAL)        # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)
        cv2.resizeWindow(winname, 300,800)
        cv2.imshow(winname,img)
        

    
    if debug:
        # Draw the bounding box
        bounding_boxes = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 1) 

        #cv2.imshow('input', image)
        showInMovedWindow('1mask', mask,0,10)
        showInMovedWindow('2closing', closing,305,10)
        showInMovedWindow('3opening', opening,610,10)
        showInMovedWindow('4bounding_box', bounding_boxes,920,10)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return x_y_w_h, aspect_ratio