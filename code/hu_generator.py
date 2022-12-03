
from __future__ import annotations
import cv2
from math import copysign, log10
import numpy as np



def hu_rearrange(moment_array):
    for i in range(0, 7):
        moment_array[i] =abs( -1 * \
            copysign(1.0, moment_array[i]) * log10(abs(moment_array[i])))
    return moment_array

def hu_moment_extract(image, x_y_w_h : list[int,int,int,int],top_part : float = 0.0, debug: bool = False) -> list[float, float, float, float, float, float, float]:
    """Returns 7 HU values already scaled to a similar number region
    ----------
    x_y_w_h : list[int,int,int,int]
        Information necessary for cropping out just the bottle
    show_debug_info : bool, optional
        Show the different processing steps of the image, by default False
    Returns
    -------
    list[float, float, float, float, float, float, float]
        A List of 7 Hu values for each processed image.
    """
    x,y,w,h = x_y_w_h


    if top_part != 0:
        cropped = image[y:y+(int(h*top_part)), x:x+w]
        if debug:
            print("Prozent crop",top_part)
    else:     
        cropped = image[y:y+h, x:x+w]
        

    # Converts the BGR color space of the image to Greyscale for thresholding
    grey = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
            

    binary = []
    dst = cv2.threshold(grey, 120, 255, cv2.THRESH_BINARY)[1]
         

    # Calculate Moments
    moments = cv2.moments(dst)
                
    # Calculate Hu Moments
    huMoments = cv2.HuMoments(moments)
                
    new = huMoments.copy()
    # call function to logaritmize the values to be in the same number range and make all positive signed


    hu_moment_list = hu_rearrange(new)
    hu_moment_list = np.asarray(hu_moment_list, dtype=object)
    hu_moment_list.flatten()
    hu_moment_list = np.squeeze(hu_moment_list, axis=None)                    




    # Show the processing steps of the image
    if debug:
        
        cv2.namedWindow("input")        # Create a named window
        cv2.moveWindow("input", 0, 30)  # Move it to (40,30)
        cv2.imshow('input', image)

        cv2.namedWindow("grey")        # Create a named window
        cv2.moveWindow("grey", 500, 30)  # Move it to (40,30)
        cv2.imshow('grey', grey)

        cv2.namedWindow("binary")        # Create a named window
        cv2.moveWindow("binary", 1000, 30)  # Move it to (40,30)
        cv2.imshow("binary", dst)
        print(hu_moment_list)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return hu_moment_list


# if __name__ == '__main__':
#     hu_gen = HU_Generator()
#     hu, labels = hu_gen.hu_moment_process(
#         show_debug_info=True)
#     # print(np.ndim(hu))
#     test = np.asarray(hu, dtype=object)
#     test.flatten()
#     test = np.squeeze(test, axis=None)

#     print("Dimensions: ", np.ndim(test))
#     print(labels)
#     np.savetxt("test_hu.csv", test, delimiter=",")

    