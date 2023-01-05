
from __future__ import annotations
import cv2
import os
from math import copysign, log10
import numpy as np

def cropper(image,x_y_w_h,top_part,folder,number):
        """Saves a cropped to the bottle version of the image

        Args:
            x_y_w_h (_type_): Information needed for cropping. 
        """        
        x,y,w,h = x_y_w_h


        if top_part != 0:
            cropped = image[y:y+(int(h*top_part)), x:x+w]
            
        else:     
            cropped = image[y:y+h, x:x+w]
        path='data_cropped/'+folder+'/'    
        cv2.imwrite(os.path.join(path , str(number)+'.jpg'), cropped)