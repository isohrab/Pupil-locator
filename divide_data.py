import csv
from random import shuffle
from xml.etree import ElementTree

from utils import *


def process_data(original_path, portion):
    """
    process original data with respective labels, create a csv file include the image_path and its label
    :param original_path: path to the original data
    :param portion: portion of train, valid, test set as array [0.85,0.15]
    :return:
    """

    # list all bmp files and labels
    # load all image names into the list
    location_folders = [os.path.join(original_path, f)
                        for f in os.listdir(original_path)
                        if os.path.isdir(os.path.join(original_path, f))]

    # loop over location folders
    subjects_folders = []
    for l in location_folders:
        subjects_folders.extend([os.path.join(l, f)
                                 for f in os.listdir(l)
                                 if os.path.isdir(os.path.join(l, f))])

    # loop over images and save their path in the lists
    images_list = []
    labels_list = []
    for s in subjects_folders:
        images_list.extend([os.path.join(s, f)
                            for f in os.listdir(s)
                            if f.endswith(".bmp")])
        labels_list.extend([os.path.join(s, f)
                            for f in os.listdir(s)
                            if f.endswith(".xml")])

    # sort both lists to match labels with the images
    images_list = sorted(images_list)
    labels_list = sorted(labels_list)

    assert len(images_list)==len(labels_list)

    # contain both images and labels in one list
    data = []
    for i, img in enumerate(images_list):
        xml_path = labels_list[i]
        e = ElementTree.parse(xml_path).getroot()
        x = np.float32(e[0].text)
        y = np.float32(e[1].text)
        w = np.float32(e[2].text)
        h = np.float32(e[3].text)
        a = np.float32(e[4].text)

        if x <= 0 or x >= 192:
            print("label for {0} is out of bound".format(img))
            continue

        if y <= 0 or y >= 192:
            print("label for {0} is out of bound".format(img))
            continue

        data.append([img, x, y, w, h, a])

    # shuffle data
    shuffle(data)

    # based portion values, make train, validation, test set
    data_len = len(data)
    train_len = int(np.ceil(data_len * portion[0]))

    train_data = data[:train_len]
    valid_data = data[train_len:]

    assert data_len == (len(train_data)+len(valid_data))

    # save all lists to data/ folder
    saveCSV(train_data, "train_data.csv", "data/")
    saveCSV(valid_data, "valid_data.csv", "data/")

    print("There are {0} images in train set".format(len(train_data)))
    print("There are {0} images in validation set".format(len(valid_data)))


def saveCSV(data_list, output_name, save_path):
    # save a list into a CSV file
    check_dir(save_path)
    p = os.path.join(save_path, output_name)

    with open(p, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_list)

    print("{0} has been successfully saved on {1}".format(output_name, save_path))

if __name__ == "__main__":
    process_data("data/Original-data", [0.9, 0.1])
    print("done...")