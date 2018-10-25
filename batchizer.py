import os
from random import shuffle

import cv2
import numpy as np

from utils import change_channel, gray_normalizer


class Batchizer(object):
    """
    list the images filename and read labels.csv,
    shuffle them at each epoch and yield `batch_size` of images
    """

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size

        # check if CSV files are exist
        if not os.path.isfile(data_path):
            raise FileNotFoundError

        # load the records into memory
        self.data_list = []
        with open(data_path, "r") as f:
            for line in f:
                #  values: [ img_path, x, y, w, h , a]
                values = line.strip().split(",")
                self.data_list.append([values[0],  # image path
                                       values[1],  # x
                                       values[2],  # y
                                       values[3],  # w
                                       values[4],  # h
                                       values[5]])  # a

        self.n_batches = int(np.ceil(len(self.data_list) / self.batch_size))

    def __len__(self):
        return len(self.data_list)

    def batches(self, ag, lbl_len=4, num_c=1,
                zero_mean=False):

        # infinitely do ....
        while True:
            # before each epoch, shuffle data
            shuffle(self.data_list)

            images = []
            labels = []
            img_names = []

            # for all records in data list
            for row in self.data_list:
                # read the image and ground truth
                image = cv2.imread(row[0], cv2.IMREAD_GRAYSCALE)
                label = np.asarray(row[1:], dtype=np.float32)

                # add noise to images and corresponding label
                if ag is not None:
                    image, label = ag.addNoise(image, label)

                # discard unused labels
                label = label[0:lbl_len]

                labels.append(label)

                # zero mean the image
                if zero_mean:
                    image = gray_normalizer(image)

                # change to desired num_channel
                image = change_channel(image, num_c)

                images.append(image)

                img_names.append(row[0])
                if len(images) == self.batch_size:
                    yield images, labels, img_names

                    # empty the list for next yield
                    images = []
                    labels = []
                    img_names = []

            # just yield reminded data
            if len(images) > 0:
                yield images, labels, img_names
