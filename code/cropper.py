from __future__ import annotations
import cv2
import os

def cropper(image,x_y_w_h,top_part,folder,number,path):
        """
        Crops an image based on the x_y_w_h coordinates provided and saves the cropped image to a specified path.

        Parameters:
            image (np.ndarray): The image to be cropped
            x_y_w_h (list[int,int,int,int]): Information needed for cropping, contains the x, y, width and height coordinates of the crop
            top_part (float): the top part of the image to be cropped
            folder (str): The folder name where the image is saved
            number (int): The number of the image
            path (str): The path where the image is saved
        
        """        
        x,y,w,h = x_y_w_h

        if top_part != 0:
            cropped = image[y:y+(int(h*top_part)), x:x+w]
            
        else:     
            cropped = image[y:y+h, x:x+w]
        path=path+'/'+folder+'/'    
        cv2.imwrite(os.path.join(path , str(number)+'.jpg'), cropped)
