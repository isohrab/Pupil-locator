import os
import numpy as np
import cv2

def check_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def rf(low, high):
    """
    return a random float number between [low, high)
    :param low: lower bound
    :param high: higher bound (excluded)
    :return: a float number between [low, high)
    """
    return np.random.uniform(low, high)


def ri(low, high):
    """
    return a random int number between [low, high)
    :param low: lower bound
    :param high: higher bound (excluded)
    :return: an int number between [low, high)
    """
    return np.random.randint(low, high)

def anotator(img, label):
    """
    draw a yellow + on input image based label
    :param img: input image
    :param label: location of pupil
    :return: anotated pupil location
    """
    w, h = img.shape
    rgb = np.zeros(shape=(w, h, 3), dtype=np.uint8)
    rgb[:, :, 0] = img  # TODO: a better way!
    rgb[:, :, 1] = img
    rgb[:, :, 2] = img

    l1xs = int(label[0] - label[2]/2)
    l1ys = int(label[1])
    l1xe = int(label[0] + label[2]/2)
    l1ye = int(label[1])

    l2xs = int(label[0])
    l2ys = int(label[1] - label[3]/2)
    l2xe = int(label[0])
    l2ye = int(label[1] + label[3]/2)

    rgb = cv2.line(rgb, (l1xs, l1ys), (l1xe, l1ye), (255, 255, 0), 1)
    rgb = cv2.line(rgb, (l2xs, l2ys), (l2xe, l2ye), (255, 255, 0), 1)

    return rgb
