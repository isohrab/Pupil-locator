import os
from random import shuffle
from PIL import Image
import numpy as np
import cv2


class Batchizer(object):
    """
    list the images filename and read labels.csv,
    shuffle them at each epoch and yield `batch_size` of images
    """

    def __init__(self, data_path, batch_size):
        self.batch_size = batch_size

        if not os.path.isfile(data_path):
            raise FileNotFoundError

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

    def batches(self, ag):
        # before each epoch, shuffle data
        while True:
            shuffle(self.data_list)

            images = []
            labels = []
            img_names = []
            for row in self.data_list:
                image = cv2.imread(row[0], cv2.IMREAD_GRAYSCALE)
                label = np.asarray(row[1:5], dtype=np.float32)
                # add noise to images and corresponding label
                ag_img, ag_lbl = ag.addNoise(image, label)
                images.append(np.expand_dims(np.array(ag_img), -1))
                labels.append(ag_lbl)
                img_names.append(row[0])
                if len(images) == self.batch_size:
                    yield images, labels, img_names
                    images = []
                    labels = []
                    img_names = []

            # just yield reminded data
            if len(images) > 0:
                yield images, labels, img_names
