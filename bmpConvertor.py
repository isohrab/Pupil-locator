import os
from random import shuffle
from PIL import Image
from xml.etree import ElementTree
import csv


def get_filenames(root_path):
    # list all bmp files and labels
    # load all image names into the list
    location_folders = [os.path.join(root_path, f)
                        for f in os.listdir(root_path)
                        if os.path.isdir(os.path.join(root_path, f))]

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
    return images_list, labels_list


def bmp2jpg(filename_list, setname):
    # save all jpeg images to trainset or testset
    save_path = 'data/' + setname + '/'
    folder_index = 0

    if not os.path.exists(save_path):
        os.mkdir(save_path)
        # os.mkdir(save_path + str(folder_index))
    images_in_folder = 5000

    counter = 0
    for p in filename_list:
        if counter % images_in_folder == 0:
            folder_index += 1
            if not os.path.exists('data/{0}/{1}'.format(setname, folder_index)):
                os.mkdir('data/{0}/{1}'.format(setname, folder_index))

        counter += 1

        im = Image.open(p)
        im.save('data/{0}/{1}/{2}.jpg'.format(setname, folder_index, counter), 'JPEG')



def saveCSV(data_list, output_name, save_path):
    # save a list into a CSV file
    if os.path.isdir(save_path):
        p = os.path.join(save_path, output_name)
    else:
        raise IOError

    data_csv = xml2list(data_list)
    with open(p, "w") as f:
        writer = csv.writer(f)
        writer.writerows(data_csv)


def xml2list(data_list):
    labels = []
    counter = 1
    for l in data_list:
        e = ElementTree.parse(l).getroot()
        x = e[0].text
        y = e[1].text
        w = e[2].text
        h = e[3].text
        a = e[4].text
        labels.append([counter, x, y, w, h, a])
        counter += 1
    return labels

if __name__ == "__main__":
    # list of sorted BMP files and XML files
    bmp_list, xml_list = get_filenames('Original-data/')

    # shuffle data and split XX% of data for test set
    data = list(zip(bmp_list, xml_list))
    shuffle(data)
    shuffled_bmp_list, shuffled_xml_list = zip(*data)

    # just to check some files
    print(shuffled_bmp_list[88886])
    print(shuffled_xml_list[88886])
    print(shuffled_bmp_list[88887])
    print(shuffled_xml_list[88887])
    print(shuffled_bmp_list[88888])
    print(shuffled_xml_list[88888])
    print(shuffled_bmp_list[88889])
    print(shuffled_xml_list[88889])

    # divide data to train set and test set
    ratio = round(len(shuffled_bmp_list) * 0.1)

    train_bmp_fn = shuffled_bmp_list[:-ratio]
    train_xml_fn = shuffled_xml_list[:-ratio]
    print(len(train_bmp_fn))

    test_bmp_fn = shuffled_bmp_list[-ratio:]
    test_xml_fn = shuffled_xml_list[-ratio:]
    print(len(test_bmp_fn))

    # save train jpeg files
    bmp2jpg(train_bmp_fn, "train")

    # save labels to csv file inside the train folder
    saveCSV(train_xml_fn, 'train_labels.csv', 'data/train/')

    # save test data inside the data/test folder
    bmp2jpg(test_bmp_fn, 'test')

    saveCSV(test_xml_fn, 'test_labels.csv', 'data/test')

    print("Done...")
