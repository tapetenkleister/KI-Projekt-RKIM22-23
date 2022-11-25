
from __future__ import annotations
import cv2
import csv
from typing import TypeVar
from basic_class import BasicImageClass
from math import copysign,log10
import numpy as np
from aspect_ratio import AspectRatioGenerator
Image = TypeVar('Image')
ImageLabel = TypeVar('ImageLabel')

def hu_rearrange(moment_array):
        for i in range(0,7):
            moment_array[i] = -1* copysign(1.0, moment_array[i]) * log10(abs(moment_array[i]))
        return moment_array


class HU_Cap(BasicImageClass):
    def __init__(self, dir_path: str = "data/pictures_tobi_w_timo/pictures", scale_fact: float = 0.1) -> None:
        super().__init__(dir_path, scale_fact)
    

    def hu_moment_process(self, show_debug_info: bool = False) -> list[float, float, float, float, float, float, float]:
        """Returns 7 HU values already scaled to a similar number region from the top image part
        ----------
        show_debug_info : bool, optional
            Show the different processing steps of the image, by default False
        Returns
        -------
        list[float, float, float, float, float, float, float]
            A List of 7 Hu values for each processed image from the top of the bottle in order to detect the shape of the cap
        """
        # Load all the images wanted
        images, _labels = self._load_images()

        aspect_ratio_gen = AspectRatioGenerator()
        x_y_w_h = aspect_ratio_gen.process_images(
        show_debug_info=False)[0]

        #print("Postion und Höhe breite",x_y_w_h)
        # Cropping an image  
        x = [position[0]
            for position in x_y_w_h.copy()]
        y = [position[1]
            for position in x_y_w_h.copy()]
        w = [position[2]
            for position in x_y_w_h.copy()]
        h = [position[3]
            for position in x_y_w_h.copy()]
       


        #print("Positionen",x,y)
        #print("Breite/Höhe",w,h)
 
        images = [image[y[count]:int(h[count]*0.3)+y[count], x[count]:x[count]+w[count]]
            for count, image in enumerate(images.copy())]


        # Converts the BGR color space of the image to Greyscale for thresholding

        grey = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            for image in images.copy()]

        binary=[]
        dst = [cv2.threshold(image, 120, 255, cv2.THRESH_BINARY)[1]
            for image in grey]

        # Calculate Moments 
        moments = [cv2.moments(image) 
            for image in dst]
        #Calculate Hu Moments 
        huMoments = [cv2.HuMoments(values)
            for values in moments]
        new = huMoments.copy()    
        #call function to logaritmize the values to be in the same number range 
        hu_moment_list = [hu_rearrange(moments)
            for moments in new]

        # Show the processing steps of the image
        if show_debug_info:
            for i in range(len(images)):
                cv2.namedWindow("input")        # Create a named window
                cv2.moveWindow("input", 0,30)  # Move it to (40,30)
                cv2.imshow('input', images[i])


                


                cv2.namedWindow("grey")        # Create a named window
                cv2.moveWindow("grey", 500,30)  # Move it to (40,30)
                cv2.imshow('grey',grey[i])

                cv2.namedWindow("binary")        # Create a named window
                cv2.moveWindow("binary", 1000,30)  # Move it to (40,30)
                cv2.imshow("binary",dst[i])
                #print("nummer:",i)
                #print(hu_moment_list[i])
                
                cv2.waitKey(0)
                cv2.destroyAllWindows()
                
        return hu_moment_list,_labels


if __name__ == '__main__':
    hu_gen_cap = HU_Cap()
    hu,labels = hu_gen_cap.hu_moment_process(
        show_debug_info=False)
    #print(np.ndim(hu))
    test = np.asarray(hu, dtype=object)
    test.flatten()
    test = np.squeeze(test, axis=None)
   
    #print("Dimensions: ",np.ndim(test))
    print(labels)
    np.savetxt("test_hu_cap1.csv", test, delimiter=",")
    
    #print(test)
    with open('test_hu_cap1.csv','r') as out:
        csv_out=csv.writer(out)
        csv_out.writerow(['hu0','hu1','hu2','hu3','hu4','hu5','hu6','label'])
        for row in test:
            csv_out.writerow(row)