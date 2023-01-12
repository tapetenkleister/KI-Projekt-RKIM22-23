from __future__ import annotations
import cv2
import os
import datetime
import sys
import csv
from typing import TypeVar
from cropper import cropper
from aspect_ratio_extract import aspect_ratio_extract


class TestBeerImages():
    """TestBeerImages class used to crop the images used for testing e.g. in the CNN and save them for the CNN to predict
    """

    def __init__(self, dir_path: str = "test_images", scale_fact: float = 0.1) -> None:
        self._dir_path = dir_path
        self._scale_fact = scale_fact
        self._label_dict = {
            'broken': 0, 'closed_seal_broken': 1, 'closed_sealed': 2, 'open_broken': 3}

    def processing(self, max_num_images: int = sys.maxsize, load_folder: list[str] = ["all"],debug : bool = False) :
        """Load the images inside the directory (dir_path), crop and save them for the CNN to use. The target folder is
        the dir_path+'_cropped'
        The images have to be seperated into subdirectories, which have to be labeled by the classification of the image ('broken', 'closed_seal_broken', 'closed_sealed', 'open_broken').

        Parameters
        ----------
        max_num_images : int, optional
            Defines the maximum number of images to load, by default sys.maxsize (i.e. all of them)
        load_folder : list, optional
            Defines a list of folders from which the images are loaded, by default ["all"]. 
            More than one folder can be loaded by adding the foldername to the list, e.g. load_folder = ['closed_seal_broken', 'closed_sealed']
        debug : bool
            Turns on debug messages for each image
        Returns
        -------
        """

        folder_stop = False
        nb_image = 0

        folder_list = os.listdir(
            self._dir_path) if load_folder[0] == "all" else load_folder

        for labeled_folder in folder_list:
            if True:
                    print('Folder: ',labeled_folder)

            for image_path in os.listdir(self._dir_path + '/' + labeled_folder):
                try:
                    
                    print('Image No:',nb_image+1)
                    image = cv2.imread(self._dir_path + '/' +
                                    labeled_folder + '/' + image_path)
               
                    height, width, _colour_channels = image.shape
                    image = cv2.resize(
                        image, (int(width*self._scale_fact), int(height*self._scale_fact)), interpolation=cv2.INTER_AREA)
                    scaled_height, scaled_width, _colour_channels = image.shape

                    # call functions to extract a feature from a single image
                    x_y_w_h,aspect_ratio = aspect_ratio_extract(image,debug=False)
                    cropper(image, x_y_w_h,top_part=0.0,folder=labeled_folder,number=nb_image, path=self._dir_path+'_cropped')
                    
                    
                        
                #stopping condition based on the given argument max_num_images   
                    if nb_image>=max_num_images-1:
                        folder_stop = True
                        break
                    nb_image += 1
                except Exception as e:
                    print(e)
                    print("Error at image no:",nb_image+1,image_path)
                    break
            if folder_stop:
                print('Finished with maximum number of images')
                break
            else: 
                print('Finished because running out of images or error')      
     
    

test = TestBeerImages(dir_path = "test_images",scale_fact= 0.5   )
test.processing(debug= False)

