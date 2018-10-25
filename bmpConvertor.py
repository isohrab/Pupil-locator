import os

from PIL import Image
from tqdm import tqdm


# read all BMP images and convert them to jpg images. it will save 14 Gb space
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
    for s in subjects_folders:
        images_list.extend([os.path.join(s, f)
                            for f in os.listdir(s)
                            if f.endswith(".bmp")])

    # sort both lists to match labels with the images
    images_list = sorted(images_list)
    return images_list


def bmp2jpg(bmp_path):
    # save all jpeg images to trainset or testset
    path = bmp_path.split("/")
    name = path[-1].split(".")
    jpg_name = name[0]
    save_path = path[0] + "/" + path[1] + "/" + path[2] + "/" + jpg_name + ".jpg"
    im = Image.open(bmp_path)
    im.save(save_path, 'JPEG')


if __name__ == "__main__":
    # list of sorted BMP files and XML files
    bmp_list = get_filenames('Original-data/')

    for bmp in tqdm(bmp_list):
        bmp2jpg(bmp)

    print("Done...")
