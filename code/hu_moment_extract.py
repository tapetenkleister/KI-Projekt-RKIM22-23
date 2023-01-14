from __future__ import annotations
import cv2
from math import copysign, log10
import numpy as np

def hu_rearrange(moment_array):
    """
    Rearrange the Hu Moments to be in the same range and have positive sign
    Parameters:
        moment_array (list): List of Hu Moments 
    Returns:
        moment_array (list): List of rearranged Hu Moments
    """
    for i in range(0, 7):
        moment_array[i] =abs( -1 * \
            copysign(1.0, moment_array[i]) * log10(abs(moment_array[i])))
    return moment_array

def hu_moment_extract(image, x_y_w_h : list[int,int,int,int],top_part : float = 0.0, debug: bool = False) -> list[float, float, float, float, float, float, float]:
    """
    Returns 7 HU values already scaled to a similar number region and absolute value
    Parameters:
    ----------
    image : numpy array
        The image on which Hu Moments are to be extracted
    x_y_w_h : list[int,int,int,int]
        Information necessary for cropping out just the bottle
    top_part : float, optional
        Percent of the image to be cropped, by default 0.0
    debug: bool, optional
        Show the different processing steps of the image, by default False
    Returns
    -------
    list[float, float, float, float, float, float, float]
        A List of 7 Hu values for each processed image.
    """
    x,y,w,h = x_y_w_h
    
    #crop the extraction to a specified part of the image e.g. 0.1 is looking at the top 10%
    #of an image
    if top_part != 0:
        cropped = image[y:y+(int(h*top_part)), x:x+w]
    else:     
        cropped = image[y:y+h, x:x+w]

    # Converts the BGR color space of the image to Greyscale for thresholding
    grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    dst = cv2.threshold(grey, 130, 255, cv2.THRESH_BINARY)[1]

    # Calculate Moments
    moments = cv2.moments(dst)

    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
 
    # Call function to logaritmize the values to be in the same number range and make all positive signed
    hu_moment_list = hu_rearrange(huMoments)
    hu_moment_list = np.asarray(hu_moment_list, dtype=object)
    hu_moment_list.flatten()
    hu_moment_list = np.squeeze(hu_moment_list, axis=None)                    

    # Show the processing steps of the image with a function to arrange the windows
    def showInMovedWindow(winname, img, x, y):
        cv2.namedWindow(winname,cv2.WINDOW_NORMAL)      # Create a named window
        cv2.moveWindow(winname, x, y)   # Move it to (x,y)
        cv2.resizeWindow(winname, 300,800)
        cv2.imshow(winname,img)
        
    if debug:
        showInMovedWindow('Input', image,0,10)
        showInMovedWindow('Greyscale', grey,305,10)
        showInMovedWindow('Thresholded', dst,610,10)
        print(hu_moment_list)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return hu_moment_list
