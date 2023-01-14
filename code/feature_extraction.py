from __future__ import annotations
import cv2
import os
import datetime
import sys
import csv
from typing import TypeVar
from aspect_ratio_extract import aspect_ratio_extract
from hu_moment_extract import hu_moment_extract

from detect_red_seal import detect_red_seal
Image = TypeVar('Image')
ImageLabel = TypeVar('ImageLabel')
Current_Date = datetime.datetime.today().strftime ('%d_%b_%Y_%H_%M_%S')

class BeerBottle():
    """Base class used to create a generator class specific to every feature. Classes inheriting from this class are ensured to use the same
    order of data points.
    """

    def __init__(self, dir_path: str = "code/data", scale_fact: float = 0.5) -> None:
        self._dir_path = dir_path
        self._scale_fact = scale_fact
        self._label_dict = {
            'broken': 0, 'closed_seal_broken': 1, 'closed_sealed': 2, 'open_broken': 3}

    def processing(self, max_num_images: int = sys.maxsize, load_folder: list[str] = ["all"],debug : bool = False) :
        """
        Load the images inside the directory (dir_path) and extract feature from each image by calling functions.
        Afterwards the features are written to feature_list.csv. The images have to be seperated into subdirectories, which have to be labeled by the classification of the image ('broken', 'closed_seal_broken', 'closed_sealed', 'open_broken').

        Parameters
        ----------
        max_num_images : int, optional
            Defines the maximum number of images to load, by default sys.maxsize (i.e. all of them)
        load_folder : list, optional
            Defines a list of folders from which the images are loaded, by default ["all"]. 
            More than one folder can be loaded by adding the foldername to the list, e.g. load_folder = ['closed_seal_broken', 'closed_sealed']
        debug : bool
            Turns on debug messages for each image
        Returns :
            Nothing
        -------
        """
        #Initialize  folder structure and CSV creation
        folder_stop = False
        feature_csv = open('analysis/feature_list.csv', 'w')
        feature_row = []
        #define the header and features that end up in the CSV
        header = ['Aspect Ratio','Seal Position', 'Hu_0', 'Hu_1', 'Hu_2', 'Cap_Hu_0', 'Cap_Hu_1', 'Cap_Hu_2','Label']
        nb_image = 0

        folder_list = os.listdir(
            self._dir_path) if load_folder[0] == "all" else load_folder

        #Iterate throuh every folder inside the specified image classes folder
        for labeled_folder in folder_list:
            if True:
                    print('Folder: ',labeled_folder)

            for image_path in os.listdir(self._dir_path + '/' + labeled_folder):
                try:
                    #load image
                    print('Image No:',nb_image+1)
                    image = cv2.imread(self._dir_path + '/' +
                                    labeled_folder + '/' + image_path)

                    #resize the image to a smaller scale for faster calculation
                    height, width, _ = image.shape
                    image = cv2.resize(
                        image, (int(width*self._scale_fact), int(height*self._scale_fact)), interpolation=cv2.INTER_AREA)

                    # call functions to extract a feature from the image
                    x_y_w_h,aspect_ratio = aspect_ratio_extract(image,debug=False)
                    hu_moment_list = hu_moment_extract(image,x_y_w_h, top_part=0.0, debug=False)
                    cap_hu_moment_list = hu_moment_extract(image, x_y_w_h, top_part=0.3, debug=False)
                    seal_position = detect_red_seal(image, x_y_w_h,top_part=0.5, debug=False)

                    #append all features to the row that is to be added
                    feature_row.append([aspect_ratio, seal_position, hu_moment_list[0],hu_moment_list[1],hu_moment_list[2],
                        cap_hu_moment_list[0],cap_hu_moment_list[1],cap_hu_moment_list[2],labeled_folder])
       
                    #stopping condition based on the given argument max_num_images   
                    if nb_image>=max_num_images-1:
                        folder_stop = True
                        break
                    nb_image += 1
                #error handling if e.g. the size of an image is to small of an extraction doesn't work
                except Exception as e:
                    print(e)
                    print("Error at image no:",nb_image+1,image_path)
                    break
            #stop condition
            if folder_stop:
                print('Finished with maximum number of images')
                break
            else: 
                print('Finished because running out of images or error')    

        #write all extracted features into the csv file for further examination
        write = csv.writer(feature_csv,delimiter=',')   
        write.writerow(header)    
        write.writerows(feature_row)    
        feature_csv.close()

        #rename the file with current date
        os.rename(r'analysis/feature_list.csv',r'analysis/feature_list_' + str(Current_Date) + '.csv')


#calling the class to iterate through the given folder dir_path
#to extract training features: 'code/data_cropped'
#to extract test features: 'code/test_images_cropped'

test = BeerBottle(dir_path = 'code/data_cropped',scale_fact= 0.5)
test.processing(max_num_images=2000, debug= False)



