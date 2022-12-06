from __future__ import annotations
import cv2
import os
import datetime
import sys
import csv
from typing import TypeVar
import matplotlib.pyplot as plt
import seaborn as sns
from aspect_ratio_extract import aspect_ratio_extract
from hu_moment_extract import hu_moment_extract
from vertical_line_extract import vertical_line_extract
Image = TypeVar('Image')
ImageLabel = TypeVar('ImageLabel')
Current_Date = datetime.datetime.today().strftime ('%d_%b_%Y_%H_%M_%S')

class BeerBottle():
    """Base class used to create a generator class specific to every feature. Classes inheriting from this class are ensured to use the same
    order of data points.
    """

    def __init__(self, dir_path: str = "data", scale_fact: float = 0.1) -> None:
        self._dir_path = dir_path
        self._scale_fact = scale_fact
        self._label_dict = {
            'broken': 0, 'closed_seal_broken': 1, 'closed_sealed': 2, 'open_broken': 3}

    def processing(self, max_num_images: int = sys.maxsize, load_folder: list[str] = ["all"],debug : bool = False) :
        """Load the images inside the directory (dir_path) and extract feature from each image by calling functions.
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
        Returns
        -------
        """

        folder_stop = False
        feature_csv = open('analysis/feature_list.csv', 'w')
        feature_row = []

        header = ['Aspect Ratio', 'Hu_0', 'Hu_1', 'Hu_2', 'Cap_Hu_0', 'Cap_Hu_1', 'Cap_Hu_2','Label']
        nb_image = 0

        folder_list = os.listdir(
            self._dir_path) if load_folder[0] == "all" else load_folder

        for labeled_folder in folder_list:
            if True:
                    print('Folder: ',labeled_folder)

            for image_path in os.listdir(self._dir_path + '/' + labeled_folder):
                try:
                    if True:
                        print('Image No:',nb_image+1)
                    image = cv2.imread(self._dir_path + '/' +
                                    labeled_folder + '/' + image_path)
               
                    height, width, _colour_channels = image.shape
                    image = cv2.resize(
                        image, (int(width*self._scale_fact), int(height*self._scale_fact)), interpolation=cv2.INTER_AREA)
                    scaled_height, scaled_width, _colour_channels = image.shape

                    # call functions to extract a feature from a single image
                    x_y_w_h,aspect_ratio = aspect_ratio_extract(image,debug=False)
                    hu_moment_list = hu_moment_extract(image,x_y_w_h, top_part=0.0, debug=False)
                    cap_hu_moment_list = hu_moment_extract(image, x_y_w_h, top_part=0.3, debug=False)
                    






                    #append all features to the row that is to be added
                    feature_row.append([aspect_ratio,hu_moment_list[0],hu_moment_list[1],hu_moment_list[2],
                        cap_hu_moment_list[0],cap_hu_moment_list[1],cap_hu_moment_list[2],labeled_folder])
                    


                    
                        
                #stopping condition based on the given argument max_num_images   
                    if nb_image>=max_num_images-1:
                        folder_stop = True
                        break
                    nb_image += 1
                except:
                    print("Error at image no:",nb_image+1)
                    break
            if folder_stop:
                break

        #write all extracted features into the csv file for further examination
        write = csv.writer(feature_csv,delimiter=',')   
        write.writerow(header)    
        write.writerows(feature_row)    
        feature_csv.close()







    def _crop_image(self, image: Image, x: int, y: int, w: int, h: int) -> Image:
        """Crops an input image to the given parameters.

        Parameters
        ----------
        image : Image
            The image to crop
        x : int
            x-coordinate of the upper-left corner of the new image border
        y : int
            y-coordinate of the upper-left corner of the new image border
        w : int
            width to crop (direction: from left to right)
        h : int
            height to crop (direction: from up to down)

        Returns
        -------
        Image
            Cropped image
        """
        image = image[int(y):int(y+h), int(x):int(x+w)]
        return image

    def _pairplot_df(self, dataframe) -> None:
        """Pairplots a given dataframe

        Parameters
        ----------
        data_frame : DataFrame
            Dataframe consisting of atleast two columns, one of them needs to be called "label" and must contain the labels.
        """
        sns.set_theme(style="ticks")
        plot = sns.pairplot(dataframe, hue="label", palette="tab10")
        plt.show()

    def _scatterplot_df(self, dataframe,  x: str = "label", y: str = None, hue: str = "label") -> None:
        """Scatterplots a given dataframe

        Parameters
        ----------
        dataframe : DataFrame
            Dataframe consisting of two columns, one of them needs to be called "label" and must contain the labels.
        """
        sns.set_theme(style="ticks")
        plot = sns.scatterplot(dataframe, y=y, x=x,
                               hue=hue, palette="tab10")
        plt.show()

    def _boxplot_df(self, dataframe, x: str = "label", y: str = None, hue: str = "label") -> None:
        """Boxplots a given dataframe

        Parameters
        ----------
        dataframe : DataFrame
            Dataframe consisting of two columns, one of them needs to be called "label" and must contain the labels.
        """
        sns.set_theme(style="ticks")
        plot = sns.boxplot(dataframe, y=y, x=x,
                           hue=hue, palette="tab10")
        plt.show()

    def _displot_df(self, dataframe, x: str = None, hue: str = "label") -> None:
        """Distplots a given dataframe

        Parameters
        ----------
        dataframe : DataFrame
            Dataframe consisting of two columns, one of them needs to be called "label" and must contain the labels.
        """
        sns.set_theme(style="ticks")
        plot = sns.displot(dataframe, x=x, hue=hue,
                           palette="tab10", col="label", multiple="stack")
        plt.show()



test = BeerBottle(scale_fact= 0.8)
test.processing(max_num_images=1000, debug= False)

os.rename(r'analysis/feature_list.csv',r'analysis/feature_list_' + str(Current_Date) + '.csv')


