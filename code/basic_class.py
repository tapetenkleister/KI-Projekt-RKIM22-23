from __future__ import annotations
import cv2
import os
import sys
from typing import TypeVar
import matplotlib.pyplot as plt
import seaborn as sns

Image = TypeVar('Image')
ImageLabel = TypeVar('ImageLabel')


class BasicImageClass():
    """Base class used to create a generator class specific to every feature. Classes inheriting from this class are ensured to use the same
    order of data points.
    """

    def __init__(self, dir_path: str = "data/pictures_tobi_w_timo/pictures", scale_fact: float = 0.1) -> None:
        self._dir_path = dir_path
        self._scale_fact = scale_fact
        self._label_dict = {
            'broken': 0, 'closed_seal_broken': 1, 'closed_sealed': 2, 'open_broken': 3}

    def _load_images(self, max_num_images: int = sys.maxsize, load_folder: list[str] = ["all"]) -> tuple[list[Image], list[ImageLabel]]:
        """Load the images inside the directory (dir_path). The images have to be seperated into subdirectories, which have to be labeled by the classification of the image ('broken', 'closed_seal_broken', 'closed_sealed', 'open_brocken').

        Parameters
        ----------
        max_num_images : int, optional
            Defines the maximum number of images to load, by default sys.maxsize (i.e. all of them)
        load_folder : list, optional
            Defines a list of folders from which the images are loaded, by default ["all"]. 
            More than one folder can be loaded by adding the foldername to the list, e.g. load_folder = ['closed_seal_broken', 'closed_sealed']

        Returns
        -------
        tuple[list[Images], list[ImageLabel]]
            A List of all images. A List of all imagelabels
        """
        self.images_inh = []
        self.labels_inh = []
        nb_image = 0

        folder_list = os.listdir(
            self._dir_path) if load_folder[0] == "all" else load_folder

        for labeled_folder in folder_list:
            for image_path in os.listdir(self._dir_path + '/' + labeled_folder):
                image = cv2.imread(self._dir_path + '/' +
                                   labeled_folder + '/' + image_path)
                nb_image += 1
                height, width, _colour_channels = image.shape
                image = cv2.resize(
                    image, (int(width*self._scale_fact), int(height*self._scale_fact)), interpolation=cv2.INTER_AREA)
                self.images_inh.append(image)
                self.labels_inh.append(self._label_dict[labeled_folder])
                if max_num_images <= nb_image:
                    return self.images_inh, self.labels_inh
        return self.images_inh, self.labels_inh

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
