import os
from random import shuffle
from PIL import Image
import numpy as np


class Batchizer(object):
    """
    list the images filename and read labels.csv,
    shuffle them at each epoch and yield `batch_size` of images
    """

    def __init__(self, root_path, batch_size, setname):
        self.batch_size = batch_size

        if not os.path.isdir(root_path):
            raise FileNotFoundError

        part_folders = [os.path.join(root_path, f)
                        for f in os.listdir(root_path)
                        if os.path.isdir(os.path.join(root_path, f))]

        self.images_fn = []
        for p in part_folders:
            self.images_fn.extend([os.path.join(p, f)
                                   for f in os.listdir(p)
                                   if f.endswith(".jpg")])

        # sort the list file name to match the labels
        self.images_fn = sorted(self.images_fn)

        labels_fn = setname+"_labels.csv"
        labels_path = os.path.join(root_path, labels_fn)

        if not os.path.isfile(labels_path):
            raise FileNotFoundError

        self.labels = []
        with open(labels_path, "r") as f:
            for line in f:
                #  values: [ line #, x, y, w, h , a]
                values = line.strip().split(",")
                self.labels.append([values[1],  # x
                                    values[2],  # y
                                    values[3],  # w
                                    values[4],  # h
                                    values[5]])  # a

        self.n_batches = len(self.images_fn)/self.batch_size
        print(self.n_batches)

    def __shuffler(self):
        data = list(zip(self.images_fn, self.labels))
        shuffle(data)
        self.images_fn, self.labels = zip(*data)

    def batches(self):
        # before each epoch, shuffle data
        self.__shuffler()

        images = []
        labels = []
        for i, fn in enumerate(self.images_fn):
            image = Image.open(fn, 'r')
            # todo: add noise here
            label = self.labels[i]
            images.append(np.expand_dims(np.array(image), -1))
            labels.append(label)
            if len(images) == self.batch_size:
                yield images, labels
                images = []
                labels = []

        # just yield reminded data
        if len(images) > 0:
            yield images, labels
